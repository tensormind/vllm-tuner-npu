"""Rich Live terminal dashboard for vLLM tuning runs.

Provides a btop-style full-screen terminal UI with:
- Status bar with model/study info and phase indicator
- GPU/system resource meters (when available)
- Trial progress with sparkline history
- Scrolling vLLM server log window
- Live results table with best-trial highlight
- Baseline vs best comparison panel
"""

from __future__ import annotations

import shutil
import threading
import time
from contextlib import contextmanager
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vllm_tuner.core.models import BenchmarkResult, StudyConfig, TrialResult, TrialStatus


_MAX_LOG_LINES = 50
_REFRESH_RATE = 4  # Hz

# Sparkline characters
_SPARK = "▁▂▃▄▅▆▇█"


def _spark(values: list[float], width: int = 20) -> Text:
    """Render a list of floats as a sparkline."""
    if not values:
        return Text("", style="dim")
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    span = hi - lo if hi != lo else 1.0
    txt = Text()
    for v in tail:
        idx = int((v - lo) / span * (len(_SPARK) - 1))
        txt.append(_SPARK[idx], style="green")
    return txt


def _bar(fraction: float, width: int = 20, fill_style: str = "green", empty_style: str = "dim") -> Text:
    """Render a horizontal usage bar ████░░░░."""
    filled = int(fraction * width)
    txt = Text()
    txt.append("█" * filled, style=fill_style)
    txt.append("░" * (width - filled), style=empty_style)
    return txt


def _fmt_metric(val: float | None, unit: str = "", prec: int = 1) -> str:
    if val is None:
        return "—"
    return f"{val:.{prec}f}{unit}"


class LiveDashboard:
    """Full-screen Rich Live dashboard (btop-style) for tuning runs."""

    def __init__(self, study_config: StudyConfig):
        self._config = study_config
        self._console = Console(force_terminal=True)
        self._live: Live | None = None

        # State
        self._phase = "initializing"
        self._phase_detail = ""
        self._server_logs: list[str] = []
        self._trial_results: list[_TrialRow] = []
        self._current_trial: int | None = None
        self._baseline_result: BenchmarkResult | None = None
        self._start_time = time.monotonic()
        self._best_tokens: float = 0.0

        # Sparkline histories
        self._throughput_history: list[float] = []
        self._latency_history: list[float] = []
        self._tokens_history: list[float] = []

        # Hardware metrics (populated by monitor callback)
        self._gpu_util: float | None = None
        self._gpu_mem_used: float | None = None
        self._gpu_mem_total: float | None = None
        self._gpu_temp: int | None = None

    # ── Lifecycle ──────────────────────────────────────────

    @contextmanager
    def live_context(self):
        """Context manager — enters alternate screen, runs Live display."""
        from vllm_tuner.helper.logging import restore_console, suppress_console

        self._start_time = time.monotonic()
        is_interactive = self._console.is_terminal
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=_REFRESH_RATE,
            screen=is_interactive,
            transient=is_interactive,
        )
        suppress_console()
        try:
            with self._live:
                self._start_tick_thread()
                yield self
        finally:
            self._live = None
            restore_console()

    def _start_tick_thread(self) -> None:
        """Start background thread that refreshes the display every second for the live timer."""

        def _tick() -> None:
            while self._live is not None:
                time.sleep(1)
                self._refresh()

        t = threading.Thread(target=_tick, daemon=True)
        t.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_layout())

    # ── Hardware monitoring callback ───────────────────────

    def on_gpu_metrics(
        self,
        utilization: float,
        mem_used_gb: float,
        mem_total_gb: float,
        temperature: int | None = None,
    ) -> None:
        """Update GPU stats from hardware monitor."""
        self._gpu_util = utilization
        self._gpu_mem_used = mem_used_gb
        self._gpu_mem_total = mem_total_gb
        self._gpu_temp = temperature

    # ── Event callbacks ────────────────────────────────────

    def on_study_start(self) -> None:
        self._phase = "baseline"
        self._phase_detail = "Running baseline benchmark..."
        self._refresh()

    def on_baseline_start(self) -> None:
        self._phase = "baseline"
        self._phase_detail = "Running baseline with default parameters"
        self._refresh()

    def on_baseline_complete(self, result: BenchmarkResult) -> None:
        self._baseline_result = result
        self._phase = "optimizing"
        self._phase_detail = ""
        self._refresh()

    def on_trial_start(self, trial_number: int, params: dict[str, Any]) -> None:
        self._current_trial = trial_number
        self._phase = "optimizing"
        self._phase_detail = f"Trial #{trial_number}"
        self._server_logs.clear()
        self._refresh()

    def on_server_starting(self, cmd: str) -> None:
        self._phase_detail = "Starting vLLM server"
        self._refresh()

    def on_server_log(self, line: str) -> None:
        self._server_logs.append(line)
        if len(self._server_logs) > _MAX_LOG_LINES * 2:
            self._server_logs = self._server_logs[-_MAX_LOG_LINES:]
        self._refresh()

    def on_server_ready(self) -> None:
        self._phase_detail = "Server ready — running benchmark"
        self._refresh()

    def on_server_failed(self, error: str) -> None:
        self._phase_detail = f"Server failed: {error}"
        self._refresh()

    def on_benchmark_start(self, trial_number: int) -> None:
        self._phase_detail = f"Trial #{trial_number}: benchmarking..."
        self._refresh()

    def on_benchmark_error(self, trial_number: int, error: str) -> None:
        self._server_logs.append(f"[BENCHMARK ERROR] Trial #{trial_number}: {error}")
        self._refresh()

    def on_trial_complete(self, result: TrialResult) -> None:
        row = _TrialRow.from_result(result)
        self._trial_results.append(row)
        self._current_trial = None
        # Update sparkline histories
        if result.benchmark:
            bm = result.benchmark
            self._throughput_history.append(bm.throughput_req_per_sec)
            self._latency_history.append(bm.p95_latency_ms)
            self._tokens_history.append(bm.output_tokens_per_sec)
            if bm.output_tokens_per_sec > self._best_tokens:
                self._best_tokens = bm.output_tokens_per_sec
        # Surface error in log panel
        if result.error_message:
            self._server_logs.append(f"[ERROR] Trial #{result.trial_number}: {result.error_message}")
        self._refresh()

    def on_study_complete(self) -> None:
        self._phase = "complete"
        self._phase_detail = "Optimization complete"
        self._refresh()

    # ── Full-screen layout ─────────────────────────────────

    def _build_layout(self) -> Layout:
        """Build btop-style fullscreen layout using Rich Layout."""
        layout = Layout()

        # ┌─── header (3 lines) ──────────────────────────────┐
        # ├─── top ────────────────────────────────────────────┤
        # │  metrics (left)    │  results table (right)        │
        # ├─── logs (full width) ──────────────────────────────┤
        # ├─── footer (progress) ──────────────────────────────┤
        # └───────────────────────────────────────────────────-┘
        has_compare = bool(self._baseline_result and self._trial_results)
        has_logs = bool(self._server_logs)

        parts = [Layout(name="header", size=3)]

        # Top row: metrics + results (+ optional comparison)
        top = Layout(name="top")
        top_right_parts = [Layout(self._render_results_table(), name="results")]
        if has_compare:
            top_right_parts.append(Layout(self._render_comparison(), name="compare", size=9))

        top.split_row(
            Layout(self._render_metrics_panel(), name="metrics", size=34),
            Layout(name="right"),
        )
        top["right"].split_column(*top_right_parts)
        parts.append(top)

        # Logs panel — full width, takes remaining space
        if has_logs:
            parts.append(Layout(self._render_log_panel(), name="logs", size=12))

        parts.append(Layout(name="footer", size=5))

        layout.split_column(*parts)
        layout["header"].update(self._render_header())
        layout["footer"].update(self._render_footer())

        return layout

    # ── Render sections ────────────────────────────────────

    def _render_header(self) -> Panel:
        """Top status bar."""
        elapsed = time.monotonic() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        time_str = f"{hrs}:{mins:02d}:{secs:02d}" if hrs else f"{mins:02d}:{secs:02d}"

        phase_styles = {
            "initializing": ("dim", "INIT"),
            "baseline": ("yellow", "BASELINE"),
            "optimizing": ("blue", "OPTIMIZING"),
            "complete": ("green", "COMPLETE"),
        }
        color, label = phase_styles.get(self._phase, ("white", self._phase.upper()))

        # Completed / total trials
        n_done = len(self._trial_results)
        n_total = self._config.optimization.n_trials

        left = Text()
        left.append(" vLLM Tuner ", style="bold white on magenta")
        left.append("  ")
        left.append(self._config.model, style="bold cyan")
        left.append("  ")
        left.append(self._config.study.name, style="dim")

        right = Text()
        right.append(f" {label} ", style=f"bold white on {color}")
        right.append(f"  {n_done}/{n_total} trials  ", style="dim")
        right.append(f"⏱ {time_str} ", style="bold")

        # Combine left and right
        txt = Text()
        txt.append_text(left)
        # Pad to push right side
        padding = max(0, (shutil.get_terminal_size((80, 24)).columns - 4) - len(left.plain) - len(right.plain))
        txt.append(" " * padding)
        txt.append_text(right)

        return Panel(txt, style="bright_blue", padding=(0, 0), height=3)

    def _render_metrics_panel(self) -> Panel:
        """GPU + performance metrics panel."""
        lines: list[Text] = []

        # GPU utilization bar (if available)
        if self._gpu_util is not None:
            gpu_line = Text()
            gpu_line.append(" GPU ", style="bold cyan")
            gpu_line.append_text(_bar(self._gpu_util / 100.0, width=16))
            gpu_line.append(f" {self._gpu_util:.0f}%", style="bold")
            if self._gpu_temp is not None:
                gpu_line.append(f"  {self._gpu_temp}°C", style="yellow" if self._gpu_temp > 75 else "dim")
            lines.append(gpu_line)

            if self._gpu_mem_used is not None and self._gpu_mem_total is not None:
                mem_frac = self._gpu_mem_used / self._gpu_mem_total if self._gpu_mem_total > 0 else 0
                mem_line = Text()
                mem_line.append(" MEM ", style="bold cyan")
                mem_line.append_text(_bar(mem_frac, width=16))
                mem_line.append(f" {self._gpu_mem_used:.1f}/{self._gpu_mem_total:.0f} GB", style="bold")
                lines.append(mem_line)
        else:
            no_gpu = Text()
            no_gpu.append(" GPU ", style="bold cyan")
            no_gpu.append("  [no metrics]", style="dim")
            lines.append(no_gpu)

        lines.append(Text())  # blank separator

        # Best metrics so far
        best = self._find_best_trial()
        best_line = Text()
        best_line.append(" Best ", style="bold green")
        if best:
            best_line.append(f"  {best.output_tokens_per_sec:.0f} tok/s", style="bold green")
            best_line.append(f"  {best.throughput_req_per_sec:.1f} req/s", style="green")
            best_line.append(f"  P95 {best.p95_latency_ms:.0f}ms", style="green")
        else:
            best_line.append("  —", style="dim")
        lines.append(best_line)

        # Sparkline for tokens/s
        spark_line = Text()
        spark_line.append(" tok/s ", style="dim")
        spark_line.append_text(_spark(self._tokens_history, width=22))
        lines.append(spark_line)

        # Sparkline for latency
        lat_line = Text()
        lat_line.append(" p95   ", style="dim")
        lat_line.append_text(_spark(self._latency_history, width=22))
        lines.append(lat_line)

        content = Group(*lines)
        return Panel(content, title="[bold]Metrics[/bold]", border_style="cyan", padding=(0, 0))

    def _render_log_panel(self) -> Panel:
        """Scrolling server output panel (full-width)."""
        # Fixed height panel — show last N lines that fit
        max_visible = 10

        visible = self._server_logs[-max_visible:]
        log_text = Text(no_wrap=True, overflow="ellipsis")

        if not visible:
            log_text.append("  Waiting for server output...", style="dim italic")
        else:
            for i, line in enumerate(visible):
                if i > 0:
                    log_text.append("\n")
                stripped = line.rstrip()
                low = stripped.lower()
                if "error" in low:
                    log_text.append(stripped, style="red")
                elif "warning" in low:
                    log_text.append(stripped, style="yellow")
                elif "ready" in low or "started" in low:
                    log_text.append(stripped, style="green")
                elif "benchmark" in low:
                    log_text.append(stripped, style="bright_blue")
                else:
                    log_text.append(stripped, style="dim")

        return Panel(
            log_text,
            title="[bold]Server Output[/bold]",
            border_style="dim",
        )

    def _render_results_table(self) -> Panel:
        """Trial results table (right side)."""
        table = Table(
            show_lines=False,
            show_edge=False,
            border_style="dim",
            pad_edge=False,
            expand=True,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Status", width=8)
        table.add_column("Throughput", justify="right", width=11)
        table.add_column("Tok/s", justify="right", width=9)
        table.add_column("P95", justify="right", width=8)
        table.add_column("TTFT", justify="right", width=8)
        table.add_column("Time", justify="right", width=7)
        table.add_column("Parameters", style="dim", ratio=1)

        # Calculate how many rows fit
        term_h = shutil.get_terminal_size((80, 24)).lines
        # Approximate: header=3, footer=5, compare≈9, table_header=3, panel_border=2
        has_compare = bool(self._baseline_result and self._trial_results)
        max_rows = max(3, term_h - (3 + 5 + (9 if has_compare else 0) + 5))

        visible = self._trial_results[-max_rows:]
        best_tokens = self._best_tokens

        for row in visible:
            status_style = {"completed": "green", "failed": "red", "pending": "yellow"}.get(row.status, "white")
            # Highlight best row
            is_best = (
                row.benchmark is not None and best_tokens > 0 and row.benchmark.output_tokens_per_sec >= best_tokens
            )
            num_style = "bold green" if is_best else "dim"
            info = row.error_message if row.status == "failed" and row.error_message else row.params_summary

            table.add_row(
                Text(str(row.trial_number), style=num_style),
                Text("✓" if row.status == "completed" else "✗" if row.status == "failed" else "…", style=status_style),
                row.throughput,
                row.tokens_per_sec,
                row.p95,
                row.ttft,
                row.duration,
                info,
            )

        caption = ""
        if len(self._trial_results) > max_rows:
            caption = f" showing last {max_rows} of {len(self._trial_results)} "

        return Panel(
            table,
            title="[bold]Trial Results[/bold]",
            subtitle=caption if caption else None,
            border_style="blue",
        )

    def _render_comparison(self) -> Panel:
        """Baseline vs best comparison (bottom-right)."""
        best = self._find_best_trial()
        if best is None or self._baseline_result is None:
            return Panel(Text("  Waiting for data...", style="dim"), title="Comparison", border_style="dim")

        bl = self._baseline_result

        def _delta(base: float, opt: float, lower_better: bool = False) -> Text:
            if base == 0:
                return Text("—", style="dim")
            pct = ((opt - base) / base) * 100
            color = "red" if pct < 0 else "green"
            sign = "+" if pct > 0 else ""
            return Text(f"{sign}{pct:.1f}%", style=f"bold {color}")

        table = Table(show_header=True, show_edge=False, border_style="dim", pad_edge=True, expand=True)
        table.add_column("", style="cyan", width=12)
        table.add_column("Baseline", justify="right", width=10)
        table.add_column("Best", justify="right", style="bold", width=10)
        table.add_column("Δ", justify="right", width=9)

        metrics = [
            ("Throughput", bl.throughput_req_per_sec, best.throughput_req_per_sec, False),
            ("Tokens/s", bl.output_tokens_per_sec, best.output_tokens_per_sec, False),
            ("P95 (ms)", bl.p95_latency_ms, best.p95_latency_ms, True),
            ("TTFT (ms)", bl.ttft_ms, best.ttft_ms, True),
        ]

        def _fmt(val: float) -> str:
            """Auto-precision: more decimals for small values."""
            if abs(val) < 0.01:
                return f"{val:.4f}"
            if abs(val) < 1:
                return f"{val:.3f}"
            if abs(val) < 100:
                return f"{val:.2f}"
            return f"{val:.1f}"

        for label, base_v, best_v, lower in metrics:
            table.add_row(label, _fmt(base_v), _fmt(best_v), _delta(base_v, best_v, lower))

        return Panel(table, title="[bold]Baseline vs Best[/bold]", border_style="green")

    def _render_footer(self) -> Panel:
        """Bottom progress bar and status."""
        n_done = len(self._trial_results)
        n_total = self._config.optimization.n_trials
        n_failed = sum(1 for r in self._trial_results if r.status == "failed")
        fraction = n_done / n_total if n_total > 0 else 0

        progress_line = Text()
        progress_line.append(" Progress ", style="bold")
        progress_line.append_text(_bar(fraction, width=40, fill_style="bold green"))
        progress_line.append(f"  {n_done}/{n_total}", style="bold")
        if n_failed:
            progress_line.append(f"  ({n_failed} failed)", style="red")

        detail = Text()
        detail.append(" ")
        if self._phase == "baseline":
            detail.append("◉ ", style="yellow")
            detail.append(self._phase_detail or "Running baseline...", style="yellow")
        elif self._phase == "optimizing":
            detail.append("◉ ", style="blue")
            detail.append(self._phase_detail or "Optimizing...", style="blue")
        elif self._phase == "complete":
            detail.append("✓ ", style="bold green")
            detail.append("Optimization complete — press Ctrl+C or wait", style="green")
        else:
            detail.append("◉ ", style="dim")
            detail.append("Initializing...", style="dim")

        content = Group(progress_line, detail)
        return Panel(content, border_style="bright_blue", padding=(0, 0))

    def _find_best_trial(self) -> BenchmarkResult | None:
        """Find best trial by first objective (throughput)."""
        best_val = -1.0
        best_bench = None
        for row in self._trial_results:
            if row.benchmark is not None and row.benchmark.output_tokens_per_sec > best_val:
                best_val = row.benchmark.output_tokens_per_sec
                best_bench = row.benchmark
        return best_bench

    # ── Static output methods (before/after live display) ──

    def print_banner(self) -> None:
        """Print a static startup banner (before entering alternate screen)."""
        self._console.print()
        self._console.rule("[bold magenta]vLLM Tuner[/bold magenta]", style="bright_blue")
        self._console.print(f"  Model:   [bold cyan]{self._config.model}[/bold cyan]")
        self._console.print(f"  Study:   [bold]{self._config.study.name}[/bold]")
        self._console.print(f"  Trials:  {self._config.optimization.n_trials}")
        self._console.print(f"  Sampler: {self._config.optimization.sampler.value}")
        if self._config.parameters:
            names = [p.name for p in self._config.parameters]
            self._console.print(f"  Params:  {', '.join(names)}")
        self._console.rule(style="bright_blue")
        self._console.print()

    def print_final_summary(self) -> None:
        """Print a final static summary after the live display ends."""
        self._console.print()

        if not self._trial_results:
            self._console.print("[yellow]No trials completed.[/yellow]")
            return

        completed = [r for r in self._trial_results if r.status == "completed"]
        failed = [r for r in self._trial_results if r.status == "failed"]

        elapsed = time.monotonic() - self._start_time
        mins, secs = divmod(int(elapsed), 60)

        self._console.rule("[bold green]Optimization Complete[/bold green]", style="green")
        self._console.print(f"  Total trials:     {len(self._trial_results)}")
        self._console.print(f"  Completed:        [green]{len(completed)}[/green]")
        if failed:
            self._console.print(f"  Failed:           [red]{len(failed)}[/red]")
        self._console.print(f"  Total time:       {mins:02d}:{secs:02d}")

        if completed:
            best = self._find_best_trial()
            if best:
                self._console.print()
                self._console.print("[bold]Best trial:[/bold]")
                self._console.print(f"  Throughput:  {best.throughput_req_per_sec:.2f} req/s")
                self._console.print(f"  Tokens/s:    {best.output_tokens_per_sec:.1f}")
                self._console.print(f"  P95 Latency: {best.p95_latency_ms:.1f} ms")
                self._console.print(f"  TTFT:        {best.ttft_ms:.1f} ms")

        if failed and not completed:
            self._console.print()
            self._console.print("[bold red]All trials failed. Errors:[/bold red]")
            seen_errors: set[str] = set()
            for r in failed:
                if r.error_message and r.error_message not in seen_errors:
                    seen_errors.add(r.error_message)
                    self._console.print(f"  [red]•[/red] {r.error_message}")

        self._console.rule(style="green")
        self._console.print()


class _TrialRow:
    """Internal data holder for a rendered trial row."""

    __slots__ = (
        "trial_number",
        "status",
        "throughput",
        "tokens_per_sec",
        "p95",
        "ttft",
        "duration",
        "params_summary",
        "benchmark",
        "error_message",
    )

    def __init__(
        self,
        trial_number: int,
        status: str,
        throughput: str,
        tokens_per_sec: str,
        p95: str,
        ttft: str,
        duration: str,
        params_summary: str,
        benchmark: BenchmarkResult | None,
        error_message: str = "",
    ):
        self.trial_number = trial_number
        self.status = status
        self.throughput = throughput
        self.tokens_per_sec = tokens_per_sec
        self.p95 = p95
        self.ttft = ttft
        self.duration = duration
        self.params_summary = params_summary
        self.benchmark = benchmark
        self.error_message = error_message

    @classmethod
    def from_result(cls, result: TrialResult) -> _TrialRow:
        bm = result.benchmark
        if bm and result.status == TrialStatus.COMPLETED:
            throughput = f"{bm.throughput_req_per_sec:.2f}"
            tokens = f"{bm.output_tokens_per_sec:.1f}"
            p95 = f"{bm.p95_latency_ms:.1f}"
            ttft = f"{bm.ttft_ms:.1f}"
        else:
            throughput = tokens = p95 = ttft = "—"

        params = result.config.parameters if result.config else {}
        # Compact params summary
        parts = []
        for k, v in list(params.items())[:4]:
            short_key = k.replace("_", "").replace("enable", "").replace("max", "mx")[:8]
            parts.append(f"{short_key}={v}")
        summary = ", ".join(parts)
        if len(params) > 4:
            summary += f" +{len(params) - 4}"

        return cls(
            trial_number=result.trial_number,
            status=result.status.value,
            throughput=throughput,
            tokens_per_sec=tokens,
            p95=p95,
            ttft=ttft,
            duration=f"{result.duration_seconds:.1f}s",
            params_summary=summary,
            benchmark=bm,
            error_message=result.error_message or "",
        )
