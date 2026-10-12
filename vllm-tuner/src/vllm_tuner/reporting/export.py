from __future__ import annotations

import json
from pathlib import Path

from vllm_tuner.core.models import TrialResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()


class ConfigExporter:
    """Exports optimal configuration in YAML, JSON, and Helm values format."""

    def export_yaml(self, result: TrialResult, output_path: Path) -> Path:
        """Export best trial config as YAML."""
        import yaml

        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = self._build_config_dict(result)
        output_path.write_text(yaml.dump(config_data, default_flow_style=False, sort_keys=False))

        logger.info("ConfigExporter: YAML config written to {}", output_path)
        return output_path

    def export_json(self, result: TrialResult, output_path: Path) -> Path:
        """Export best trial config as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = self._build_config_dict(result)
        output_path.write_text(json.dumps(config_data, indent=2))

        logger.info("ConfigExporter: JSON config written to {}", output_path)
        return output_path

    def export_helm_values(self, result: TrialResult, output_path: Path) -> Path:
        """Export best trial config as Helm values for vllm-production-stack."""
        import yaml

        output_path.parent.mkdir(parents=True, exist_ok=True)

        params = self._get_all_params(result)
        extra_args = [f"--{k.replace('_', '-')}={v}" for k, v in params.items()]
        helm_values = {
            "servingRuntime": {
                "extraArgs": extra_args,
            },
        }
        output_path.write_text(yaml.dump(helm_values, default_flow_style=False, sort_keys=False))

        logger.info("ConfigExporter: Helm values written to {}", output_path)
        return output_path

    @staticmethod
    def _build_config_dict(result: TrialResult) -> dict:
        """Build a config dictionary from trial result."""
        params = {}
        if result.config:
            params = {**result.config.static_parameters, **result.config.parameters}

        data = {
            "vllm_parameters": {k.replace("_", "-"): v for k, v in params.items()},
            "trial_number": result.trial_number,
        }

        if result.benchmark:
            data["performance"] = {
                "throughput_req_per_sec": result.benchmark.throughput_req_per_sec,
                "output_tokens_per_sec": result.benchmark.output_tokens_per_sec,
                "p95_latency_ms": result.benchmark.p95_latency_ms,
                "p99_latency_ms": result.benchmark.p99_latency_ms,
                "ttft_ms": result.benchmark.ttft_ms,
            }

        return data

    @staticmethod
    def _get_all_params(result: TrialResult) -> dict:
        if result.config:
            return {**result.config.static_parameters, **result.config.parameters}
        return {}
