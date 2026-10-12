from __future__ import annotations

from vllm_tuner.core.models import BenchmarkResult, TrialResult
from vllm_tuner.helper.display import DisplayConsole
from vllm_tuner.helper.logging import get_logger


logger = get_logger()


class TerminalDashboard:
    """Rich terminal dashboard for displaying tuning progress and results."""

    def __init__(self):
        self._console = DisplayConsole()

    def show_trial_result(self, result: TrialResult) -> None:
        """Display a single trial result as a formatted table."""
        data = {
            "Trial": str(result.trial_number),
            "Status": result.status.value,
            "Duration": f"{result.duration_seconds:.2f}s",
        }
        if result.benchmark:
            data.update(
                {
                    "Throughput (req/s)": f"{result.benchmark.throughput_req_per_sec:.2f}",
                    "Tokens/s": f"{result.benchmark.output_tokens_per_sec:.1f}",
                    "P95 Latency (ms)": f"{result.benchmark.p95_latency_ms:.1f}",
                    "P99 Latency (ms)": f"{result.benchmark.p99_latency_ms:.1f}",
                    "TTFT (ms)": f"{result.benchmark.ttft_ms:.1f}",
                }
            )
        if result.error_message:
            data["Error"] = result.error_message

        self._console.display_dict_as_table(data, title=f"Trial #{result.trial_number}")

    def show_study_summary(self, summary: dict) -> None:
        """Display study summary as a formatted table."""
        display_data = {k: str(v) for k, v in summary.items()}
        self._console.display_dict_as_table(display_data, title="Study Summary")

    def show_progress(self, current: int, total: int, best_value: float | None = None) -> None:
        """Display study progress."""
        progress_str = f"[{current}/{total}]"
        best_str = f" | Best: {best_value:.4f}" if best_value is not None else ""
        self._console.print(f"[bold blue]Progress: {progress_str}{best_str}[/bold blue]")

    def show_comparison(self, baseline: BenchmarkResult, optimized: BenchmarkResult) -> None:
        """Display baseline vs optimized comparison."""

        def _pct(base: float, opt: float) -> str:
            if base == 0:
                return "N/A"
            pct = ((opt - base) / base) * 100
            sign = "+" if pct > 0 else ""
            return f"{sign}{pct:.1f}%"

        data = {
            "Throughput (req/s)": f"{baseline.throughput_req_per_sec:.2f} → {optimized.throughput_req_per_sec:.2f} ({_pct(baseline.throughput_req_per_sec, optimized.throughput_req_per_sec)})",  # noqa: E501
            "P95 Latency (ms)": f"{baseline.p95_latency_ms:.1f} → {optimized.p95_latency_ms:.1f} ({_pct(baseline.p95_latency_ms, optimized.p95_latency_ms)})",  # noqa: E501
            "P99 Latency (ms)": f"{baseline.p99_latency_ms:.1f} → {optimized.p99_latency_ms:.1f} ({_pct(baseline.p99_latency_ms, optimized.p99_latency_ms)})",  # noqa: E501
            "TTFT (ms)": f"{baseline.ttft_ms:.1f} → {optimized.ttft_ms:.1f} ({_pct(baseline.ttft_ms, optimized.ttft_ms)})",  # noqa: E501
        }
        self._console.display_dict_as_table(data, title="Baseline vs Optimized")
