from __future__ import annotations

from vllm_tuner.core.models import BenchmarkResult, ConstraintSpec, ObjectiveSpec, TrialResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()


class Optimizer:
    """Computes objective values and evaluates constraints for a trial.

    Currently a mock — returns metric values from BenchmarkResult directly.
    """

    def __init__(self, objectives: list[ObjectiveSpec], constraints: list[ConstraintSpec] | None = None):
        self._objectives = objectives
        self._constraints = constraints or []

    def compute_objective_values(self, result: TrialResult) -> list[float]:
        """Extract objective values from trial results."""
        if result.benchmark is None:
            logger.warning("Trial #{} has no benchmark results, returning zeros", result.trial_number)
            return [0.0] * len(self._objectives)

        values = []
        for obj in self._objectives:
            value = self._extract_metric(result.benchmark, obj.metric)
            values.append(value)

        logger.debug("Trial #{} objective values: {}", result.trial_number, values)
        return values

    def evaluate_constraints(self, parameters: dict, result: TrialResult | None = None) -> list[float]:
        """Evaluate constraint expressions. Returns list of margin values (>=0 means satisfied).

        Tokens are resolved from the trial ``parameters`` first, then from the
        benchmark ``result`` metrics (e.g. ``ttft_ms``, ``output_tokens_per_sec``).
        A trial with no benchmark (failed/dry-run) is treated as infeasible.
        """
        if result is not None and result.benchmark is not None:
            b = result.benchmark
            resolver = {
                "throughput_req_per_sec": b.throughput_req_per_sec,
                "output_tokens_per_sec": b.output_tokens_per_sec,
                "p50_latency_ms": b.p50_latency_ms,
                "p95_latency_ms": b.p95_latency_ms,
                "p99_latency_ms": b.p99_latency_ms,
                "ttft_ms": b.ttft_ms,
                "itl_ms": b.itl_ms,
                "duration_seconds": b.duration_seconds,
                "prefix_cache_hit_rate": b.prefix_cache_hit_rate,
            }
        else:
            resolver = {}
        resolver.update(parameters)

        if not self._constraints:
            return []

        constraint_values = []
        no_metrics = result is None or result.benchmark is None
        for constraint in self._constraints:
            if no_metrics:
                # No metrics collected → infeasible so the sampler avoids this trial.
                constraint_values.append(-1.0)
                continue
            value = self._eval_constraint(constraint.expression, resolver)
            constraint_values.append(value)
            if value < 0:
                logger.info("Constraint violated: {} (margin={:.2f})", constraint.expression, value)
        return constraint_values

    @staticmethod
    def _eval_constraint(expression: str, resolver: dict) -> float:
        """Evaluate a constraint expression. Returns margin (>=0 if satisfied).

        Supports simple comparisons like ``ttft_ms <= 10000`` or
        ``max_model_len >= max_num_seqs``. The returned margin is the signed
        distance from the threshold: positive when satisfied, negative when
        violated.
        """
        for op_str, margin_fn in (
            (">=", lambda lhs, rhs: lhs - rhs),
            ("<=", lambda lhs, rhs: rhs - lhs),
            ("!=", lambda lhs, rhs: 0.0 if lhs != rhs else -1.0),
            ("==", lambda lhs, rhs: 0.0 if lhs == rhs else -1.0),
            (">", lambda lhs, rhs: lhs - rhs),
            ("<", lambda lhs, rhs: rhs - lhs),
        ):
            if op_str in expression:
                parts = expression.split(op_str, 1)
                if len(parts) != 2:
                    break

                def _resolve(token: str) -> float:
                    try:
                        return float(token)
                    except ValueError:
                        return float(resolver.get(token, 0.0))

                lhs = _resolve(parts[0].strip())
                rhs = _resolve(parts[1].strip())
                return margin_fn(lhs, rhs)

        logger.warning("Cannot parse constraint expression: '{}'", expression)
        return 1.0  # Unknown expression — assume satisfied

    @staticmethod
    def _extract_metric(benchmark: BenchmarkResult, metric: str) -> float:
        """Extract a named metric from benchmark results."""
        metric_map = {
            "throughput_req_per_sec": benchmark.throughput_req_per_sec,
            "output_tokens_per_sec": benchmark.output_tokens_per_sec,
            "p50_latency_ms": benchmark.p50_latency_ms,
            "p95_latency_ms": benchmark.p95_latency_ms,
            "p99_latency_ms": benchmark.p99_latency_ms,
            "ttft_ms": benchmark.ttft_ms,
            "itl_ms": benchmark.itl_ms,
        }
        value = metric_map.get(metric, 0.0)
        if metric not in metric_map:
            logger.warning("Unknown metric '{}', returning 0.0", metric)
        return value
