from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from vllm_tuner.core.models import BenchmarkResult, StudyConfig, TrialResult, TrialStatus
from vllm_tuner.helper.logging import get_logger


logger = get_logger()


def _esc(value: object) -> str:
    """HTML-escape a value for safe embedding."""
    return html.escape(str(value))


def _fmt(value: float, decimals: int = 2) -> str:
    """Format a float to *decimals* places."""
    return f"{value:.{decimals}f}"


def _pct_change(baseline: float, best: float, *, lower_is_better: bool = False) -> tuple[float, str]:
    """Return (change_pct, css_class) for an improvement cell."""
    if baseline == 0:
        return 0.0, "neutral"
    if lower_is_better:
        delta = (baseline - best) / baseline * 100
    else:
        delta = (best - baseline) / baseline * 100
    css = "positive" if delta > 0 else ("negative" if delta < 0 else "neutral")
    return delta, css


class HTMLReportGenerator:
    """Generates a self-contained interactive HTML report.

    Uses Plotly.js (loaded from CDN) for charts — no Python dependencies
    beyond the standard library.
    """

    # ────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────

    def generate(
        self,
        results: list[TrialResult],
        output_dir: Path,
        study_name: str = "study",
        baseline: BenchmarkResult | None = None,
        objectives: list | None = None,
        study_config: StudyConfig | None = None,
        optuna_study: object | None = None,
    ) -> Path:
        """Generate an HTML report and return the file path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{study_name}_report.html"

        logger.info("HTMLReportGenerator: generating report with {} trial results", len(results))

        completed = [r for r in results if r.status == TrialStatus.COMPLETED and r.benchmark]

        best = None
        if completed:
            best = self._select_best(completed, objectives)

        html_content = self._render(
            study_name=study_name,
            results=results,
            completed=completed,
            best=best,
            baseline=baseline,
            study_config=study_config,
            optuna_study=optuna_study,
        )

        report_path.write_text(html_content, encoding="utf-8")
        logger.info("HTMLReportGenerator: report written to {}", report_path)
        return report_path

    # ────────────────────────────────────────────
    # Chart data builders (JSON for Plotly.js)
    # ────────────────────────────────────────────

    @staticmethod
    def _chart_data(completed: list[TrialResult]) -> dict:
        """Build JSON-serializable dict consumed by the Plotly.js template."""
        trials = []
        for r in completed:
            bm = r.benchmark
            hw = r.hardware
            trials.append(
                {
                    "num": r.trial_number,
                    "throughput": round(bm.throughput_req_per_sec, 3),
                    "tokens_per_sec": round(bm.output_tokens_per_sec, 3),
                    "p50": round(bm.p50_latency_ms, 2),
                    "p95": round(bm.p95_latency_ms, 2),
                    "p99": round(bm.p99_latency_ms, 2),
                    "ttft": round(bm.ttft_ms, 2),
                    "avg_tok_req": round(bm.avg_output_tokens_per_request, 1),
                    "mem_util": round(hw.avg_utilization, 3) if hw else 0,
                    "peak_vram_gb": round(hw.peak_memory_used_mb / 1024, 2) if hw else 0,
                    "params": r.config.parameters if r.config else {},
                }
            )
        return {"trials": trials}

    # ────────────────────────────────────────────
    # Full-page renderer
    # ────────────────────────────────────────────

    def _render(
        self,
        *,
        study_name: str,
        results: list[TrialResult],
        completed: list[TrialResult],
        best: TrialResult | None,
        baseline: BenchmarkResult | None,
        study_config: StudyConfig | None = None,
        optuna_study: object | None = None,
    ) -> str:
        gen_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        chart_json = json.dumps(self._chart_data(completed))

        expected_output = 0
        if study_config:
            expected_output = study_config.benchmark.output_tokens

        # ── Benchmark config section ──
        bench_config_html = self._benchmark_config_section(study_config) if study_config else ""

        # ── Summary metric cards ──
        bm = best.benchmark if best else None
        cards_html = self._metric_cards(bm, len(results), best)

        # ── Baseline comparison ──
        baseline_html = self._baseline_section(baseline, bm) if baseline and bm else ""

        # ── Best parameters ──
        best_params_html = self._best_params_section(best)

        # ── Trial results table ──
        trials_table_html = self._trials_table(results, expected_output)

        # ── Optuna analysis charts ──
        optuna_charts = self._optuna_charts(optuna_study) if optuna_study else []
        optuna_divs = ""
        optuna_scripts = ""
        for i, (title, fig_json) in enumerate(optuna_charts):
            div_id = f"optuna-chart-{i}"
            optuna_divs += f'<div class="chart-container"><h3>{_esc(title)}</h3><div id="{div_id}"></div></div>\n'
            optuna_scripts += (
                f"(function() {{\n"
                f"  var spec = {fig_json};\n"
                f"  Plotly.newPlot('{div_id}', spec.data, spec.layout, {{responsive: true}});\n"
                f"}})();\n"
            )
        optuna_section = ""
        if optuna_divs:
            optuna_section = f"<h2>Optuna Analysis</h2>\n{optuna_divs}"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>vLLM Tuning Report — {_esc(study_name)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{_CSS}
</head>
<body>

<div class="header">
  <h1>vLLM Tuner Report</h1>
  <p class="timestamp">Generated: {_esc(gen_time)}</p>
  <h2>Study: {_esc(study_name)}</h2>

  {bench_config_html}

  <h3>Best Configuration Metrics</h3>
  {cards_html}

  {baseline_html}

  {best_params_html}
</div>

<h2>Performance Charts</h2>

<div class="chart-container"><h3>Throughput Over Trials</h3><div id="chart-throughput"></div></div>
<div class="chart-container"><h3>Latency Over Trials (P95)</h3><div id="chart-latency"></div></div>
<div class="chart-container"><h3>Pareto Front: Throughput vs Latency</h3><div id="chart-pareto"></div></div>
<div class="chart-container"><h3>Tokens per Second</h3><div id="chart-tokens"></div></div>
<div class="chart-container"><h3>Avg Tokens per Request</h3><div id="chart-tok-req"></div></div>
<div class="chart-container"><h3>GPU Utilization &amp; VRAM</h3><div id="chart-gpu"></div></div>
<div class="chart-container">
  <h3>Combined View</h3>
  <div id="chart-combined" style="height:700px"></div>
</div>

{optuna_section}

<h2>Trial Results</h2>
{trials_table_html}

<footer><p>Generated by <strong>vllm_tuner</strong></p></footer>

<script>
(function() {{
var D = {chart_json};
var expectedOutput = {expected_output};
var T = D.trials;
if (!T.length) return;

var nums = T.map(t => t.num);
var tp   = T.map(t => t.throughput);
var p95  = T.map(t => t.p95);
var tps  = T.map(t => t.tokens_per_sec);
var mem  = T.map(t => t.mem_util);
var avgTokReq = T.map(t => t.avg_tok_req);
var peakVram  = T.map(t => t.peak_vram_gb);

// Throughput
Plotly.newPlot('chart-throughput', [{{
  x: nums, y: tp, mode: 'lines+markers', name: 'Throughput',
  line: {{color: '#2ecc71'}},
  hovertemplate: 'Trial %{{x}}<br>%{{y:.2f}} req/s<extra></extra>'
}}], {{xaxis: {{title: 'Trial'}}, yaxis: {{title: 'req/s'}}, hovermode: 'x unified', margin: {{t:10}} }});

// Latency
Plotly.newPlot('chart-latency', [{{
  x: nums, y: p95, mode: 'lines+markers', name: 'P95 Latency',
  line: {{color: '#f39c12'}},
  hovertemplate: 'Trial %{{x}}<br>%{{y:.1f}} ms<extra></extra>'
}}], {{xaxis: {{title: 'Trial'}}, yaxis: {{title: 'ms'}}, hovermode: 'x unified', margin: {{t:10}} }});

// Pareto
Plotly.newPlot('chart-pareto', [{{
  x: tp, y: p95, mode: 'markers', name: 'Trials',
  marker: {{size: 10, color: '#3498db', line: {{width: 1, color: '#2980b9'}}}},
  text: nums.map(n => 'Trial ' + n),
  hovertemplate: '%{{text}}<br>Throughput: %{{x:.2f}} req/s<br>P95: %{{y:.1f}} ms<extra></extra>'
}}], {{xaxis: {{title: 'Throughput (req/s)'}}, yaxis: {{title: 'P95 Latency (ms)'}}, hovermode: 'closest', margin: {{t:10}} }});

// Tokens/s
Plotly.newPlot('chart-tokens', [{{
  x: nums, y: tps, mode: 'lines+markers', name: 'Tokens/s',
  line: {{color: '#9b59b6'}},
  hovertemplate: 'Trial %{{x}}<br>%{{y:.1f}} tok/s<extra></extra>'
}}], {{xaxis: {{title: 'Trial'}}, yaxis: {{title: 'tokens/s'}}, hovermode: 'x unified', margin: {{t:10}} }});

// Avg Tokens per Request (with expected reference line)
var tokReqTraces = [{{
  x: nums, y: avgTokReq, type: 'bar', name: 'Avg Tok/Req',
  marker: {{color: avgTokReq.map(v => v < expectedOutput * 0.5 ? '#e74c3c' : '#27ae60')}},
  hovertemplate: 'Trial %{{x}}<br>%{{y:.0f}} tokens/req<extra></extra>'
}}];
var tokReqLayout = {{
  xaxis: {{title: 'Trial'}}, yaxis: {{title: 'tokens/request'}},
  hovermode: 'x unified', margin: {{t:10}},
  shapes: expectedOutput > 0 ? [{{
    type: 'line', x0: -0.5, x1: nums.length - 0.5,
    y0: expectedOutput, y1: expectedOutput,
    line: {{color: '#e67e22', width: 2, dash: 'dash'}}
  }}] : [],
  annotations: expectedOutput > 0 ? [{{
    x: nums[nums.length-1], y: expectedOutput,
    text: 'expected: ' + expectedOutput,
    showarrow: false, yshift: 12,
    font: {{color: '#e67e22', size: 11}}
  }}] : []
}};
Plotly.newPlot('chart-tok-req', tokReqTraces, tokReqLayout);

// GPU Utilization & VRAM
Plotly.newPlot('chart-gpu', [
  {{x: nums, y: mem.map(v => v * 100), type: 'bar', name: 'GPU Util %',
    marker: {{color: '#3498db', opacity: 0.7}},
    hovertemplate: 'Trial %{{x}}<br>%{{y:.1f}}%<extra></extra>'}},
  {{x: nums, y: peakVram, mode: 'lines+markers', name: 'Peak VRAM (GB)',
    yaxis: 'y2', line: {{color: '#e74c3c', width: 2}},
    hovertemplate: 'Trial %{{x}}<br>%{{y:.1f}} GB<extra></extra>'}}
], {{
  xaxis: {{title: 'Trial'}},
  yaxis: {{title: 'GPU Utilization %', side: 'left'}},
  yaxis2: {{title: 'Peak VRAM (GB)', side: 'right', overlaying: 'y'}},
  hovermode: 'x unified', margin: {{t:10}}, legend: {{x: 0, y: 1.15, orientation: 'h'}}
}});

// Combined 2x2
var combined = document.getElementById('chart-combined');
Plotly.newPlot(combined, [
  {{x: nums, y: tp,  mode: 'lines+markers', name: 'Throughput', line: {{color:'#2ecc71'}}, xaxis:'x1', yaxis:'y1'}},
  {{x: nums, y: p95, mode: 'lines+markers', name: 'P95',        line: {{color:'#f39c12'}}, xaxis:'x2', yaxis:'y2'}},
  {{x: nums, y: tps, mode: 'lines+markers', name: 'Tokens/s',   line: {{color:'#9b59b6'}}, xaxis:'x3', yaxis:'y3'}},
  {{x: tp,   y: p95, mode: 'markers',       name: 'Pareto',     marker:{{size:8, color:'#3498db', opacity:0.7}}, xaxis:'x4', yaxis:'y4'}}
], {{
  grid: {{rows:2, columns:2, pattern:'independent'}},
  xaxis:  {{title:'Trial'}}, yaxis:  {{title:'req/s'}},
  xaxis2: {{title:'Trial'}}, yaxis2: {{title:'ms'}},
  xaxis3: {{title:'Trial'}}, yaxis3: {{title:'tok/s'}},
  xaxis4: {{title:'req/s'}}, yaxis4: {{title:'ms'}},
  showlegend: false,
  margin: {{t:30}}
}});
}})();
{optuna_scripts}
</script>
</body>
</html>"""  # noqa: E501

    # ────────────────────────────────────────────
    # Best trial selection
    # ────────────────────────────────────────────

    @staticmethod
    def _select_best(completed: list[TrialResult], objectives: list | None = None) -> TrialResult:
        """Select the best trial based on the primary objective metric."""
        if not objectives:
            return max(completed, key=lambda r: r.benchmark.output_tokens_per_sec)

        primary = objectives[0]
        metric = primary.metric
        direction = getattr(primary, "direction", "maximize")
        minimize = str(direction).lower() in ("minimize", "min")

        def _score(r: TrialResult) -> float:
            val = getattr(r.benchmark, metric, None)
            if val is None:
                return float("inf") if minimize else float("-inf")
            return val

        if minimize:
            return min(completed, key=_score)
        return max(completed, key=_score)

    # ────────────────────────────────────────────
    # Optuna visualization charts
    # ────────────────────────────────────────────

    @staticmethod
    def _optuna_charts(study: object) -> list[tuple[str, str]]:
        """Generate Optuna visualization charts as serialized Plotly JSON.

        Returns a list of (title, json_string) tuples.
        Gracefully degrades when optional dependencies are missing.
        """
        try:
            import optuna.visualization as vis
        except ImportError:
            logger.debug("optuna.visualization not available, skipping Optuna charts")
            return []

        charts: list[tuple[str, str]] = []
        completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if len(completed_trials) < 2:
            logger.debug("Not enough completed trials ({}) for Optuna charts", len(completed_trials))
            return []

        # Parameter Importances (requires scikit-learn)
        try:
            fig = vis.plot_param_importances(study)
            charts.append(("Parameter Importances", fig.to_json()))
        except Exception as exc:
            logger.debug("Skipping param importances chart: {}", exc)

        # Parallel Coordinate
        try:
            fig = vis.plot_parallel_coordinate(study)
            charts.append(("Parallel Coordinate", fig.to_json()))
        except Exception as exc:
            logger.debug("Skipping parallel coordinate chart: {}", exc)

        # Slice Plot
        try:
            fig = vis.plot_slice(study)
            charts.append(("Parameter Slice", fig.to_json()))
        except Exception as exc:
            logger.debug("Skipping slice chart: {}", exc)

        return charts

    # ────────────────────────────────────────────
    # HTML fragment builders
    # ────────────────────────────────────────────

    @staticmethod
    def _benchmark_config_section(config: StudyConfig) -> str:
        """Render the benchmark configuration summary."""
        bc = config.benchmark
        sp = config.static_parameters
        rows = [
            ("Model", config.model),
            ("Prompt Tokens (configured)", bc.prompt_tokens),
            ("Output Tokens (configured)", bc.output_tokens),
            ("Concurrent Requests", bc.concurrent_requests),
            ("Max Seconds", bc.max_seconds),
            ("Benchmark Provider", bc.provider.value if hasattr(bc.provider, "value") else bc.provider),
        ]
        for k, v in sp.items():
            rows.append((f"static: {k}", v))
        row_html = "".join(f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>" for label, value in rows)
        return f"""<h3>Benchmark Configuration</h3>
<table class="params-table">
<thead><tr><th>Setting</th><th>Value</th></tr></thead>
<tbody>{row_html}</tbody>
</table>"""

    @staticmethod
    def _metric_cards(bm: BenchmarkResult | None, total_trials: int, best: TrialResult | None) -> str:
        cards = [
            (_fmt(bm.throughput_req_per_sec) if bm else "-", "Throughput (req/s)"),
            (_fmt(bm.output_tokens_per_sec, 1) if bm else "-", "Tokens/s"),
            (_fmt(bm.p95_latency_ms, 1) if bm else "-", "P95 Latency (ms)"),
            (_fmt(bm.avg_output_tokens_per_request, 0) if bm else "-", "Avg Tok/Req"),
            (str(total_trials), "Total Trials"),
            (f"#{best.trial_number}" if best else "-", "Best Trial"),
        ]
        items = "".join(
            f'<div class="metric-card"><div class="metric-value">{_esc(v)}</div>'
            f'<div class="metric-label">{_esc(l)}</div></div>'
            for v, l in cards
        )
        return f'<div class="metrics-grid">{items}</div>'

    @staticmethod
    def _baseline_section(baseline: BenchmarkResult, best: BenchmarkResult) -> str:
        rows_data = [
            ("Throughput (req/s)", baseline.throughput_req_per_sec, best.throughput_req_per_sec, False),
            ("Tokens/s", baseline.output_tokens_per_sec, best.output_tokens_per_sec, False),
            ("Avg Tokens/Request", baseline.avg_output_tokens_per_request, best.avg_output_tokens_per_request, False),
            ("P50 Latency (ms)", baseline.p50_latency_ms, best.p50_latency_ms, True),
            ("P95 Latency (ms)", baseline.p95_latency_ms, best.p95_latency_ms, True),
            ("P99 Latency (ms)", baseline.p99_latency_ms, best.p99_latency_ms, True),
            ("TTFT (ms)", baseline.ttft_ms, best.ttft_ms, True),
        ]
        rows = ""
        for label, base_val, best_val, lower_better in rows_data:
            delta, css = _pct_change(base_val, best_val, lower_is_better=lower_better)
            if delta > 0:
                sign = "+"
            elif delta < 0:
                sign = "-"
            else:
                sign = ""
            rows += (
                f"<tr><td>{_esc(label)}</td>"
                f"<td>{_esc(_fmt(base_val))}</td>"
                f"<td>{_esc(_fmt(best_val))}</td>"
                f'<td class="{css}">{sign}{_fmt(abs(delta), 1)}%</td></tr>'
            )
        return f"""<h3>Baseline vs Best Trial</h3>
<table class="params-table">
<thead><tr><th>Metric</th><th>Baseline</th><th>Best Trial</th><th>Change</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""

    @staticmethod
    def _best_params_section(best: TrialResult | None) -> str:
        if not best or not best.config or not best.config.parameters:
            return ""
        rows = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in best.config.parameters.items())
        return f"""<h3>Best Parameters</h3>
<table class="params-table">
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""

    @staticmethod
    def _trials_table(results: list[TrialResult], expected_output: int = 0) -> str:
        rows = []
        for r in results:
            bm = r.benchmark
            hw = r.hardware

            if bm and bm.successful_requests > 0:
                req_str = f"{bm.successful_requests}/{bm.total_requests}"
            elif bm:
                req_str = f"0/{bm.total_requests}"
            else:
                req_str = "-"

            avg_tok = bm.avg_output_tokens_per_request if bm else 0
            is_low = expected_output > 0 and 0 < avg_tok < expected_output * 0.5
            tok_css = ' class="negative"' if is_low else ""
            tok_str = f"{avg_tok:.0f}" if bm and avg_tok > 0 else "-"

            peak_vram_str = f"{hw.peak_memory_used_mb / 1024:.1f}" if hw and hw.peak_memory_used_mb > 0 else "-"
            gpu_util_str = f"{hw.avg_utilization * 100:.1f}" if hw and hw.avg_utilization > 0 else "-"

            rows.append(
                "<tr>"
                f"<td>{_esc(r.trial_number)}</td>"
                f"<td>{_esc(r.status.value)}</td>"
                f"<td><code>{_esc(r.config.parameters if r.config else {})}</code></td>"
                f"<td>{_fmt(bm.throughput_req_per_sec) if bm else '-'}</td>"
                f"<td>{_fmt(bm.output_tokens_per_sec, 1) if bm else '-'}</td>"
                f"<td>{req_str}</td>"
                f"<td{tok_css}>{tok_str}</td>"
                f"<td>{_fmt(bm.p50_latency_ms, 1) if bm else '-'}</td>"
                f"<td>{_fmt(bm.p95_latency_ms, 1) if bm else '-'}</td>"
                f"<td>{_fmt(bm.p99_latency_ms, 1) if bm else '-'}</td>"
                f"<td>{_fmt(bm.ttft_ms, 1) if bm and bm.ttft_ms else '-'}</td>"
                f"<td>{peak_vram_str}</td>"
                f"<td>{gpu_util_str}</td>"
                f"<td>{_fmt(r.duration_seconds, 1) if r.duration_seconds else '-'}</td>"
                f"<td>{_esc(r.error_message or '')}</td>"
                "</tr>"
            )
        return f"""<table>
<thead><tr>
  <th>#</th><th>Status</th><th>Parameters</th>
  <th>Throughput</th><th>Tokens/s</th><th>Requests</th><th>Avg Tok/Req</th>
  <th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th>
  <th>TTFT (ms)</th><th>Peak VRAM (GB)</th><th>GPU Util%</th>
  <th>Duration (s)</th><th>Error</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>"""


# ────────────────────────────────────────────────
# Embedded CSS (kept as module constant)
# ────────────────────────────────────────────────

_CSS = """<style>
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 1400px; margin: 0 auto; padding: 20px;
  background: #f5f5f5; color: #333;
}
h1 { color: #2c3e50; }
h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
.header {
  background: #fff; padding: 30px; border-radius: 8px;
  margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,.1);
}
.timestamp { color: #95a5a6; font-size: 14px; }
.metrics-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 15px; margin: 20px 0;
}
.metric-card {
  background: #fff; padding: 20px; border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
}
.metric-value { font-size: 28px; font-weight: bold; color: #2ecc71; }
.metric-label { color: #7f8c8d; font-size: 14px; }
.chart-container {
  background: #fff; padding: 20px; border-radius: 8px;
  margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,.1);
}
.params-table, table {
  width: 100%; border-collapse: collapse; margin: 20px 0;
}
th, td { padding: 10px 8px; text-align: left; border-bottom: 1px solid #ddd; font-size: 13px; }
th { background: #1a1a2e; color: #fff; }
tr:nth-child(even) { background: #f9f9f9; }
tr:hover { background: #e8f0fe; }
code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: .8em; }
.positive { color: #2ecc71; font-weight: bold; }
.negative { color: #e74c3c; font-weight: bold; }
.neutral  { color: #7f8c8d; font-weight: bold; }
footer { text-align: center; color: #95a5a6; margin-top: 40px; }
</style>"""
