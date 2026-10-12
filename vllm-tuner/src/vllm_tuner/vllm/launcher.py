from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from vllm_tuner.core.models import TrialConfig
from vllm_tuner.helper.logging import get_logger


logger = get_logger()

_SIGTERM_TIMEOUT = 5
_DEFAULT_STARTUP_TIMEOUT = 300
_HEALTH_CHECK_INTERVAL = 5


def _suppress_crash_reporter() -> None:
    """Disable core dumps to suppress macOS crash-reporter dialogs."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


class VLLMLauncher:
    """Manages vLLM server subprocess lifecycle.

    Uses process groups for reliable cleanup (SIGTERM → SIGKILL escalation).
    """

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        log_callback: Any | None = None,
    ):
        self._model = model
        self._host = host
        self._port = port
        self._process: subprocess.Popen | None = None
        self._pid: int | None = None
        self._pgid: int | None = None
        self._log_lines: list[str] = []
        self._log_callback = log_callback
        self._reader_thread: threading.Thread | None = None

    def build_command(self, trial_config: TrialConfig) -> list[str]:
        """Build the vLLM serve command from trial parameters.

        Handles vLLM CLI conventions:
        - Boolean flags use ``--flag`` / ``--no-flag`` (BooleanOptionalAction)
        - Integer args must not have decimal points (``32`` not ``32.0``)
        """
        cmd = [
            "vllm",
            "serve",
            self._model,
            "--host",
            self._host,
            "--port",
            str(self._port),
        ]
        all_params = {**trial_config.static_parameters, **trial_config.parameters}
        for key, value in all_params.items():
            cli_key = f"--{key.replace('_', '-')}"

            # Boolean flags: --flag / --no-flag (vLLM uses BooleanOptionalAction)
            str_val = str(value).lower()
            if str_val in ("true", "false"):
                if str_val == "true":
                    cmd.append(cli_key)
                else:
                    cmd.append(f"--no-{key.replace('_', '-')}")
                continue

            # Integer-like floats: 32.0 → 32
            if isinstance(value, float) and value == int(value):
                cmd.extend([cli_key, str(int(value))])
            else:
                cmd.extend([cli_key, str(value)])

        logger.debug("Built vLLM command: {}", " ".join(cmd))
        return cmd

    def start(self, trial_config: TrialConfig) -> None:
        """Start vLLM server as a subprocess in a new process group."""
        if self._process is not None:
            logger.warning("VLLMLauncher: server already running (pid={}), stopping first", self._pid)
            self.stop()

        cmd = self.build_command(trial_config)
        logger.info("VLLMLauncher: starting vLLM server: {}", " ".join(cmd))

        self._log_lines = []
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            preexec_fn=_suppress_crash_reporter if sys.platform == "darwin" else None,
        )
        self._pid = self._process.pid
        self._pgid = os.getpgid(self._pid)
        logger.info("VLLMLauncher: server started (pid={}, pgid={})", self._pid, self._pgid)

        # Start a background thread to continuously read stdout
        self._reader_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        """Stop the running vLLM server (SIGTERM, then SIGKILL after timeout)."""
        if self._process is None and self._pid is None:
            logger.debug("VLLMLauncher: no server to stop")
            return

        logger.info("VLLMLauncher: stopping server (pid={}, pgid={})", self._pid, self._pgid)

        try:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=_SIGTERM_TIMEOUT)
                    logger.info("VLLMLauncher: server stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("VLLMLauncher: SIGTERM timed out, sending SIGKILL")
                    self._kill_process_group()
            elif self._pid is not None:
                # Process handle lost but PID known — use process group kill
                self._kill_process_group()
        finally:
            # Join reader thread after process is terminated (pipe will close)
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=5)
                self._reader_thread = None
            self._process = None
            self._pid = None
            self._pgid = None

    def _kill_process_group(self) -> None:
        """Kill the entire process group with SIGKILL."""
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, signal.SIGKILL)
            logger.info("VLLMLauncher: process group {} killed", self._pgid)
        except ProcessLookupError:
            logger.debug("VLLMLauncher: process group {} already terminated", self._pgid)
        except PermissionError:
            logger.error("VLLMLauncher: no permission to kill process group {}", self._pgid)

    def health_check(self) -> bool:
        """Single health check against /health endpoint."""
        url = f"{self.server_url}/health"
        try:
            with urlopen(url, timeout=5) as response:
                return response.status == 200
        except (URLError, OSError):
            return False

    def wait_until_ready(self, timeout: float = _DEFAULT_STARTUP_TIMEOUT) -> bool:
        """Poll /health endpoint until server is ready or timeout is reached."""
        logger.info(
            "VLLMLauncher: waiting for server at {} (timeout={}s)",
            self.server_url,
            timeout,
        )
        deadline = time.monotonic() + timeout
        consecutive_failures = 0
        max_failures_after_ready = 3

        while time.monotonic() < deadline:
            # Check if process has crashed
            if self._process is not None and self._process.poll() is not None:
                # Give the reader thread a moment to consume remaining output
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=2)
                logger.error(
                    "VLLMLauncher: server process exited with code {}",
                    self._process.returncode,
                )
                # Log captured output so the user can see why vLLM failed
                for line in self._log_lines[-20:]:
                    logger.error("VLLMLauncher: server output: {}", line)
                return False

            if self.health_check():
                logger.info("VLLMLauncher: server is ready at {}", self.server_url)
                return True

            consecutive_failures += 1
            if consecutive_failures > max_failures_after_ready:
                logger.debug("VLLMLauncher: health check failed ({} consecutive)", consecutive_failures)

            time.sleep(_HEALTH_CHECK_INTERVAL)

        logger.error("VLLMLauncher: server failed to start within {}s", timeout)
        return False

    def read_logs(self) -> list[str]:
        """Return collected log lines from the server stdout/stderr."""
        return list(self._log_lines)

    def _read_stdout_loop(self) -> None:
        """Continuously read stdout lines in a background thread.

        Runs until the pipe is closed (process exits or stdout is exhausted).
        Each line is appended to ``_log_lines`` and forwarded to ``_log_callback``.
        """
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                self._log_lines.append(stripped)
                if self._log_callback is not None:
                    self._log_callback(stripped)
        except (ValueError, OSError):
            pass

    @property
    def server_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def pid(self) -> int | None:
        return self._pid
