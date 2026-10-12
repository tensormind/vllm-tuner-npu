from __future__ import annotations

import json
from pathlib import Path

import fire
import yaml
from rich.panel import Panel

from vllm_tuner.helper.logging import get_logger


logger = get_logger()

_PRESETS_DIR = Path(__file__).resolve().parents[3] / "configs" / "optimization"


def _load_preset(name: str) -> dict:
    """Load an optimization preset from configs/optimization/<name>.yaml."""
    preset_path = _PRESETS_DIR / f"{name}.yaml"
    if not preset_path.exists():
        available = [p.stem for p in _PRESETS_DIR.glob("*.yaml")]
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    data = yaml.safe_load(preset_path.read_text()) or {}
    return data.get("optimization", data)


def _load_study_config(
    model: str | None = None,
    config: str | None = None,
    preset: str | None = None,
    n_trials: int = 50,
    backend: str = "local",
    output_dir: str = "./results",
):
    """Build a StudyConfig from CLI arguments and/or YAML file."""
    from vllm_tuner.core.models import StudyConfig

    if config:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config}")
        data = yaml.safe_load(config_path.read_text()) or {}
        # CLI overrides
        if model:
            data["model"] = model
        if preset:
            data["optimization"] = {**_load_preset(preset), **data.get("optimization", {})}
        if n_trials != 50:
            data.setdefault("optimization", {})["n_trials"] = n_trials
        if output_dir != "./results":
            data.setdefault("output", {})["directory"] = output_dir
        return StudyConfig.model_validate(data)

    if not model:
        raise ValueError("Either --model or --config must be specified")

    optimization = _load_preset(preset) if preset else {}
    if n_trials != 50:
        optimization["n_trials"] = n_trials

    return StudyConfig(
        model=model,
        optimization=optimization,
        execution={"backend": backend},
        output={"directory": output_dir},
    )


def _create_benchmark_provider(config):
    """Create a benchmark provider from config."""
    from vllm_tuner.core.models import AISBenchConfig, BenchmarkProviderType

    provider_type = config.benchmark.provider
    if provider_type == BenchmarkProviderType.GUIDELLM:
        from vllm_tuner.benchmarks.guidellm import GuideLLMProvider

        return GuideLLMProvider()
    elif provider_type == BenchmarkProviderType.VLLM_BENCHMARK:
        from vllm_tuner.benchmarks.vllm_benchmark import VLLMBenchmarkProvider

        return VLLMBenchmarkProvider(model=config.model)
    elif provider_type == BenchmarkProviderType.AISBENCH:
        from pathlib import Path

        from vllm_tuner.benchmarks.aisbench import AISBenchProvider

        ac = config.aisbench or AISBenchConfig()
        return AISBenchProvider(
            model_path=config.model,
            model_name=ac.model_name or (Path(config.model).name if config.model else ""),
            aisbench_work_path=ac.work_path,
            tools_path=ac.tools_path,
            dataset_dir=ac.dataset_dir,
            prefix_cache=ac.prefix_cache,
            npu_num=ac.npu_num,
            summarizer=ac.summarizer,
        )
    else:
        from vllm_tuner.benchmarks.http_client import HTTPBenchmarkProvider

        return HTTPBenchmarkProvider()


class CLI:
    """vLLM Tuner CLI — hyperparameter optimization for vLLM serving.

    Commands:
        run        Start a new tuning study
        resume     Resume an interrupted study
        report     Generate reports from a completed study
        export     Export optimal config (YAML/JSON/Helm)
        list       List available presets and parameters
        validate   Validate a configuration file
        recommend  Recommend vLLM parameters for a model
    """

    def run(
        self,
        model: str | None = None,
        config: str | None = None,
        preset: str | None = None,
        backend: str = "local",
        n_trials: int = 50,
        ray_address: str = "auto",
        output_dir: str = "./results",
    ) -> None:
        """Start a new tuning study.

        Args:
            model: HuggingFace model name (e.g., meta-llama/Llama-3.1-8B-Instruct)
            config: Path to study config YAML file
            preset: Optimization preset (high_throughput, low_latency, balanced, cost_optimized)
            backend: Execution backend (local, ray)
            n_trials: Number of optimization trials
            ray_address: Ray cluster address (for ray backend)
            output_dir: Output directory for results and reports

        """
        from vllm_tuner.baseline.runner import BaselineRunner
        from vllm_tuner.core.study_controller import StudyController
        from vllm_tuner.execution.local import LocalExecutionBackend
        from vllm_tuner.hardware.null import NullMonitor
        from vllm_tuner.reporting.live_dashboard import LiveDashboard
        from vllm_tuner.storage.sqlite import SQLiteStorage

        study_config = _load_study_config(model, config, preset, n_trials, backend, output_dir)

        live_dashboard = LiveDashboard(study_config)
        live_dashboard.print_banner()

        out_dir = Path(study_config.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        storage = SQLiteStorage(db_path=str(out_dir / "study.db"))
        benchmark_provider = _create_benchmark_provider(study_config)

        # Hardware monitor (optional)
        # Ascend NPU takes priority (this env runs on 910B3); fall back to NVIDIA GPU.
        try:
            from vllm_tuner.hardware.npu import NPUMonitor

            monitor = NPUMonitor()
            logger.info("Using Ascend NPU hardware monitor")
        except Exception:
            try:
                from vllm_tuner.hardware.nvml import NVMLMonitor

                monitor = NVMLMonitor()
                logger.info("Using NVIDIA GPU hardware monitor")
            except Exception:
                monitor = NullMonitor()

        # Execution backend
        if backend == "ray":
            from vllm_tuner.execution.ray_backend import RayExecutionBackend

            exec_backend = RayExecutionBackend(ray_address=ray_address)
        else:
            exec_backend = LocalExecutionBackend(
                model=study_config.model,
                benchmark_provider=benchmark_provider,
                benchmark_config=study_config.benchmark,
                monitor=monitor,
                dashboard=live_dashboard,
                startup_timeout=study_config.execution.startup_timeout,
            )

        # Run everything inside the live dashboard context
        controller = None
        baseline_result = None
        try:
            with live_dashboard.live_context():
                live_dashboard.on_study_start()

                # Baseline
                baseline_runner = BaselineRunner(
                    benchmark_provider=benchmark_provider,
                    monitor=monitor,
                    dashboard=live_dashboard,
                )
                baseline_result = baseline_runner.run_baseline(study_config)
                live_dashboard.on_baseline_complete(baseline_result)

                # Optimization
                controller = StudyController(study_config)
                controller.create_study(storage_url=storage.get_storage_url())
                controller.enqueue_recommended_trial()
                controller.optimize(execution_backend=exec_backend)

                live_dashboard.on_study_complete()
        except KeyboardInterrupt:
            logger.info("Interrupted by user — cleaning up...")
        finally:
            exec_backend.cleanup()

        # Post-live: save and report
        results = controller.results if controller else []
        for result in results:
            storage.save_trial(study_config.study.name, result)

        live_dashboard.print_final_summary()

        if results:
            from vllm_tuner.reporting.html import HTMLReportGenerator

            HTMLReportGenerator().generate(
                results,
                out_dir,
                study_config.study.name,
                baseline=baseline_result,
                objectives=study_config.optimization.objectives,
                study_config=study_config,
                optuna_study=controller.study if controller else None,
            )
            logger.info("Results saved to {}", out_dir)

    def resume(
        self,
        study_name: str = "vllm-tuning-study",
        storage: str = "sqlite:///study.db",
        n_trials: int | None = None,
    ) -> None:
        """Resume an interrupted tuning study.

        Args:
            study_name: Name of the study to resume
            storage: Storage URL (SQLite or PostgreSQL)
            n_trials: Additional trials to run (None = use original target)

        """
        import optuna

        from vllm_tuner.cli.rich_ui import console, print_error, print_footer, print_header, print_info
        from vllm_tuner.core.models import StudyConfig
        from vllm_tuner.core.study_controller import StudyController

        print_header("vLLM Tuner — Resume Study")

        try:
            existing_study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception as exc:
            print_error(f"Could not load study '{study_name}': {exc}")
            return

        completed = len(existing_study.trials)
        print_info("Study", study_name)
        print_info("Storage", storage)
        print_info("Completed trials", completed)

        config_data = existing_study.user_attrs.get("config", {})
        study_config = StudyConfig.model_validate(config_data) if config_data else StudyConfig(model="unknown")

        if n_trials:
            study_config.optimization.n_trials = completed + n_trials
            print_info("Additional trials", n_trials)
            print_info("New total", study_config.optimization.n_trials)

        console.print()
        controller = StudyController(study_config)
        controller._study = existing_study
        controller.optimize()

        from vllm_tuner.cli.rich_ui import print_success

        print_success(f"Optimization resumed — {len(existing_study.trials)} trials total")
        print_footer()

    def report(
        self,
        study_name: str = "vllm-tuning-study",
        storage: str = "sqlite:///study.db",
        output_dir: str = "./results",
        format: str = "html",
    ) -> None:
        """Generate reports from a completed study.

        Args:
            study_name: Name of the study
            storage: Storage URL
            output_dir: Output directory for reports
            format: Report format (html, json)

        """
        from rich.text import Text

        from vllm_tuner.cli.rich_ui import (
            console,
            make_table,
            print_error,
            print_footer,
            print_header,
            print_info,
            print_path,
            print_success,
            print_warning,
        )
        from vllm_tuner.core.models import TrialStatus
        from vllm_tuner.storage.sqlite import SQLiteStorage

        print_header("vLLM Tuner — Generate Report")

        db_path = storage.replace("sqlite:///", "")
        store = SQLiteStorage(db_path=db_path)
        results = store.load_trials(study_name)

        if not results:
            print_warning(f"No trial results found for study '{study_name}'")
            return

        # Show summary
        completed = [r for r in results if r.status == TrialStatus.COMPLETED]
        failed = [r for r in results if r.status == TrialStatus.FAILED]
        print_info("Study", study_name)
        print_info("Total trials", len(results))
        print_info("Completed", f"[green]{len(completed)}[/green]")
        if failed:
            print_info("Failed", f"[red]{len(failed)}[/red]")
        console.print()

        # Results table
        rows = []
        for r in results:
            bm = r.benchmark
            status_color = "green" if r.status == TrialStatus.COMPLETED else "red"
            rows.append(
                [
                    str(r.trial_number),
                    Text(r.status.value, style=status_color),
                    f"{bm.throughput_req_per_sec:.2f}" if bm else "—",
                    f"{bm.output_tokens_per_sec:.1f}" if bm else "—",
                    f"{bm.p95_latency_ms:.1f}" if bm else "—",
                    f"{r.duration_seconds:.1f}s",
                ]
            )

        table = make_table(
            "Trial Results",
            columns=[
                ("#", {"style": "dim", "width": 4}),
                ("Status", {"width": 10}),
                ("Throughput", {"justify": "right", "width": 12}),
                ("Tokens/s", {"justify": "right", "width": 10}),
                ("P95 (ms)", {"justify": "right", "width": 10}),
                ("Duration", {"justify": "right", "width": 10}),
            ],
            rows=rows,
        )
        console.print(table)
        console.print()

        # Generate report
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if format == "html":
            from vllm_tuner.reporting.html import HTMLReportGenerator

            path = HTMLReportGenerator().generate(results, out_dir, study_name)
            print_success("HTML report generated")
            print_path("Path", path)
        elif format == "json":
            json_path = out_dir / f"{study_name}_results.json"
            data = [r.model_dump(mode="json") for r in results]
            json_path.write_text(json.dumps(data, indent=2, default=str))
            print_success("JSON report generated")
            print_path("Path", json_path)
        else:
            print_error(f"Unknown format: '{format}'. Use: html, json")
            return

        print_footer()

    def export(
        self,
        study_name: str = "vllm-tuning-study",
        storage: str = "sqlite:///study.db",
        output: str = "./best_config.yaml",
        format: str = "yaml",
        helm: bool = False,
    ) -> None:
        """Export optimal configuration.

        Args:
            study_name: Name of the study
            storage: Storage URL
            output: Output file path
            format: Export format (yaml, json)
            helm: Also export Helm values file

        """
        import optuna

        from vllm_tuner.cli.rich_ui import (
            console,
            print_error,
            print_footer,
            print_header,
            print_info,
            print_kv_block,
            print_path,
            print_success,
        )

        print_header("vLLM Tuner — Export Configuration")

        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception as exc:
            print_error(f"Could not load study '{study_name}': {exc}")
            return

        best = study.best_trial
        params = best.params

        print_info("Study", study_name)
        print_info("Best trial", f"#{best.number}")
        print_info("Value", f"{best.value:.4f}" if best.value is not None else "N/A")
        console.print()

        console.print("  [bold]Optimized parameters:[/bold]")
        print_kv_block({f"--{k.replace('_', '-')}": v for k, v in params.items()})
        console.print()

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            output_path.write_text(json.dumps(params, indent=2))
        else:
            output_path.write_text(yaml.dump(params, default_flow_style=False))

        print_success(f"Config exported ({format.upper()})")
        print_path("Path", output_path)

        if helm:
            helm_path = output_path.with_name("helm_values.yaml")
            helm_data = {"vllm": {"extraArgs": [f"--{k}={v}" for k, v in params.items()]}}
            helm_path.write_text(yaml.dump(helm_data, default_flow_style=False))
            print_success("Helm values exported")
            print_path("Path", helm_path)

        print_footer()

    def list(self, what: str = "presets") -> None:
        """List available presets, parameters, or backends.

        Args:
            what: What to list (presets, parameters, backends, benchmarks)

        """
        from rich.text import Text

        from vllm_tuner.cli.rich_ui import console, make_table, print_error, print_footer, print_header

        print_header("vLLM Tuner — Available Options")

        if what == "presets":
            rows = [
                ["high_throughput", "Maximize request throughput (req/s)", "output_tokens_per_sec ↑", "tpe"],
                ["low_latency", "Minimize P95/P99 latency", "p95_latency_ms ↓", "tpe"],
                ["balanced", "Multi-objective: throughput + latency", "tokens/s ↑ + P95 ↓", "nsga2"],
                ["cost_optimized", "Maximize performance per dollar", "tokens/s per $/hr ↑", "tpe"],
            ]
            table = make_table(
                "Optimization Presets",
                columns=[
                    ("Preset", {"style": "bold cyan", "width": 18}),
                    ("Description", {"width": 40}),
                    ("Objective", {"style": "yellow", "width": 22}),
                    ("Sampler", {"style": "dim", "width": 8}),
                ],
                rows=rows,
            )
            console.print(table)
            console.print()
            console.print("  [dim]Usage: vllm-tuner run --preset balanced --model <model>[/dim]")

        elif what == "parameters":
            rows = [
                ["gpu_memory_utilization", "0.70–0.95", "KV cache VRAM fraction"],
                ["max_num_seqs", "16–512", "Max concurrent sequences"],
                ["max_num_batched_tokens", "1024–8192", "Max tokens per batch"],
                ["enable_chunked_prefill", "true/false", "Chunk long prefill requests"],
                ["enable_prefix_caching", "true/false", "Cache common prompt prefixes"],
                ["block_size", "16/32", "KV cache block size"],
                ["swap_space", "1–8", "CPU swap space (GiB)"],
                ["max_model_len", "model-dependent", "Max sequence length"],
            ]
            table = make_table(
                "Tunable Parameters",
                columns=[
                    ("Parameter", {"style": "bold cyan", "width": 28}),
                    ("Range", {"style": "yellow", "width": 16}),
                    ("Description", {"width": 36}),
                ],
                rows=rows,
            )
            console.print(table)

        elif what == "backends":
            rows = [
                [Text("local", style="bold cyan"), "Sequential subprocess execution", "Single machine, no setup"],
                [Text("ray", style="bold cyan"), "Distributed via Ray cluster", "Multi-node, KubeRay, auto-scaling"],
            ]
            table = make_table(
                "Execution Backends",
                columns=[
                    ("Backend", {"width": 10}),
                    ("Description", {"width": 34}),
                    ("Best for", {"style": "dim", "width": 36}),
                ],
                rows=rows,
            )
            console.print(table)

        elif what == "benchmarks":
            rows = [
                [Text("http", style="bold green"), "Built-in async HTTP (httpx)", "No extra deps, works everywhere"],
                [
                    Text("guidellm", style="bold cyan"),
                    "GuideLLM benchmarking suite",
                    "Comprehensive metrics, real datasets",
                ],
                [
                    Text("vllm_benchmark", style="bold cyan"),
                    "vLLM benchmark_serving.py",
                    "Official vLLM benchmark script",
                ],
                [
                    Text("aisbench", style="bold cyan"),
                    "ais-bench CLI (chat/completions)",
                    "Ascend NPU, real GSM8K prompts, prefix cache",
                ],
            ]
            table = make_table(
                "Benchmark Providers",
                columns=[
                    ("Provider", {"width": 16}),
                    ("Description", {"width": 30}),
                    ("Notes", {"style": "dim", "width": 34}),
                ],
                rows=rows,
            )
            console.print(table)

        else:
            print_error(f"Unknown list target: '{what}'")
            console.print("  [dim]Available: presets, parameters, backends, benchmarks[/dim]")

        print_footer()

    def validate(self, config: str) -> None:
        """Validate a study configuration file.

        Args:
            config: Path to the YAML config file to validate

        """
        from vllm_tuner.cli.rich_ui import (
            console,
            print_error,
            print_footer,
            print_header,
            print_info,
            print_success,
        )
        from vllm_tuner.core.models import StudyConfig

        print_header("vLLM Tuner — Validate Configuration")

        config_path = Path(config)
        if not config_path.exists():
            print_error(f"Config file not found: {config}")
            return

        try:
            data = yaml.safe_load(config_path.read_text()) or {}
            study_config = StudyConfig.model_validate(data)
        except Exception as e:
            print_error(f"Validation failed: {e}")
            print_footer()
            return

        print_success(f"Configuration is valid: {config}")
        console.print()

        # Show parsed config summary
        print_info("Model", study_config.model)
        print_info("Study", study_config.study.name)
        print_info("Trials", study_config.optimization.n_trials)
        print_info("Sampler", study_config.optimization.sampler.value)
        print_info("Benchmark", study_config.benchmark.provider.value)

        if study_config.parameters:
            console.print()
            console.print("  [bold]Parameters:[/bold]")
            for p in study_config.parameters:
                if p.options:
                    console.print(f"    [cyan]{p.name}[/cyan]: {p.options}")
                else:
                    console.print(f"    [cyan]{p.name}[/cyan]: {p.min}–{p.max} (step={p.step})")

        if study_config.static_parameters:
            console.print()
            console.print("  [bold]Static parameters:[/bold]")
            for k, v in study_config.static_parameters.items():
                console.print(f"    [cyan]{k}[/cyan]: {v}")

        if study_config.optimization.objectives:
            console.print()
            console.print("  [bold]Objectives:[/bold]")
            for obj in study_config.optimization.objectives:
                arrow = "↑" if obj.direction.value == "maximize" else "↓"
                console.print(f"    [cyan]{obj.metric}[/cyan] {arrow} (weight={obj.weight})")

        print_footer()

    def recommend(
        self,
        model: str,
        vram: float = 24.0,
        num_gpus: int = 1,
        device: str = "gpu",
        chip_type: str = "v6e",
        num_chips: int = 0,
    ) -> None:
        """Recommend vLLM parameters for a model based on available hardware.

        Downloads the model's HuggingFace config and estimates memory
        requirements to suggest optimal vLLM serving parameters.

        Args:
            model: HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)
            vram: Total GPU VRAM in GB per GPU (default: 24.0, GPU only)
            num_gpus: Number of GPUs available (default: 1, GPU only)
            device: Device type — "gpu" or "tpu" (default: "gpu")
            chip_type: TPU chip type — v4, v5p, v5e, v6e (default: "v6e", TPU only)
            num_chips: Number of TPU chips (default: chips_per_host for chip type, TPU only)

        """
        from rich.syntax import Syntax
        from rich.text import Text

        from vllm_tuner.cli.rich_ui import (
            console,
            make_table,
            print_error,
            print_footer,
            print_header,
            print_info,
            print_warning,
        )
        from vllm_tuner.utils.model_analyzer import TPU_CHIPS, DeviceType, analyze_model, analyze_model_tpu
        from vllm_tuner.utils.model_registry import get_model_config

        try:
            device_type = DeviceType(device.lower())
        except ValueError:
            print_error(f"Invalid device '{device}'. Use 'gpu' or 'tpu'.")
            return

        print_header("vLLM Tuner — Model Recommendation", f"[cyan]{model}[/cyan]")

        if device_type is DeviceType.TPU:
            chip_type = chip_type.lower()
            if chip_type not in TPU_CHIPS:
                print_error(f"Unknown TPU chip type '{chip_type}'. Valid types: {', '.join(sorted(TPU_CHIPS))}")
                return
            chip = TPU_CHIPS[chip_type]
            effective_chips = num_chips if num_chips > 0 else chip.chips_per_host
            print_info(
                "Hardware",
                f"{effective_chips}× TPU {chip_type} with {chip.hbm_gb:.0f} GB HBM each",
            )
        else:
            print_info("Hardware", f"{num_gpus}× GPU with {vram:.0f} GB VRAM each")

        try:
            config = get_model_config(model)
        except (RuntimeError, ValueError) as exc:
            print_error(str(exc))
            return

        if device_type is DeviceType.TPU:
            analysis = analyze_model_tpu(config, chip_type=chip_type, num_chips=num_chips, model_id=model)
        else:
            total_vram = vram * num_gpus
            analysis = analyze_model(config, total_vram_gb=total_vram, num_gpus=num_gpus, model_id=model)

        console.print()

        # Model analysis table
        analysis_rows: list[list[str | Text]] = [
            ["Parameters", f"{analysis.param_count / 1e9:.1f}B"],
            ["Weights memory", f"{analysis.weights_memory_gb:.1f} GB"],
            ["Quantized", f"{analysis.quant_method}" if analysis.is_quantized else "No (FP16)"],
        ]
        if analysis.is_moe:
            analysis_rows.append(["Architecture", f"MoE ({analysis.num_experts} experts)"])
        fit_label = "Can fit in HBM" if device_type is DeviceType.TPU else "Can fit in VRAM"
        fit_text = Text("Yes", style="green") if analysis.can_fit else Text("No", style="red")
        analysis_rows.append([fit_label, fit_text])

        table = make_table(
            "Model Analysis",
            columns=[
                ("Property", {"style": "cyan", "width": 18}),
                ("Value", {"width": 30}),
            ],
            rows=analysis_rows,
        )
        console.print(table)
        console.print()

        # Recommended parameters table
        if device_type is DeviceType.TPU:
            param_rows: list[list[str]] = [
                ["--device", "tpu"],
                ["--dtype", analysis.dtype],
                ["--max-model-len", str(analysis.max_model_len)],
                ["--max-num-seqs", str(analysis.max_num_seqs)],
                ["--max-num-batched-tokens", str(analysis.max_num_batched_tokens)],
                ["--tensor-parallel-size", str(analysis.tensor_parallel_size)],
            ]
            if analysis.data_parallel_size > 1:
                param_rows.append(["--data-parallel-size", str(analysis.data_parallel_size)])
            param_rows.append(["--block-size", str(analysis.block_size)])
        else:
            param_rows = [
                ["--dtype", analysis.dtype],
                ["--gpu-memory-utilization", str(analysis.gpu_memory_utilization)],
                ["--max-model-len", str(analysis.max_model_len)],
                ["--max-num-seqs", str(analysis.max_num_seqs)],
                ["--max-num-batched-tokens", str(analysis.max_num_batched_tokens)],
                ["--tensor-parallel-size", str(analysis.tensor_parallel_size)],
            ]
            if analysis.data_parallel_size > 1:
                param_rows.append(["--data-parallel-size", str(analysis.data_parallel_size)])
            if analysis.pipeline_parallel_size > 1:
                param_rows.append(["--pipeline-parallel-size", str(analysis.pipeline_parallel_size)])
            if analysis.enable_expert_parallel:
                param_rows.append(["--enable-expert-parallel", ""])
            param_rows.append(
                [
                    "--enable-chunked-prefill" if analysis.enable_chunked_prefill else "--no-enable-chunked-prefill",
                    "",
                ]
            )
            param_rows.append(
                [
                    "--enable-prefix-caching" if analysis.enable_prefix_caching else "--no-enable-prefix-caching",
                    "",
                ]
            )
            param_rows.append(["--kv-cache-dtype", analysis.kv_cache_dtype])
            param_rows.append(["--block-size", str(analysis.block_size)])
            param_rows.append(["--swap-space", str(analysis.swap_space)])
            if analysis.cpu_offload_gb > 0:
                param_rows.append(["--cpu-offload-gb", str(analysis.cpu_offload_gb)])
            if analysis.enforce_eager:
                param_rows.append(["--enforce-eager", ""])

        params_table = make_table(
            "Recommended vLLM Parameters",
            columns=[
                ("Parameter", {"style": "cyan", "width": 30}),
                ("Value", {"style": "yellow", "width": 20}),
            ],
            rows=param_rows,
        )
        console.print(params_table)
        console.print()

        # Build command
        if device_type is DeviceType.TPU:
            cmd_parts = [
                "vllm serve",
                model,
                "--device tpu",
                f"--dtype {analysis.dtype}",
                f"--max-model-len {analysis.max_model_len}",
                f"--max-num-seqs {analysis.max_num_seqs}",
                f"--max-num-batched-tokens {analysis.max_num_batched_tokens}",
                f"--tensor-parallel-size {analysis.tensor_parallel_size}",
            ]
            if analysis.data_parallel_size > 1:
                cmd_parts.append(f"--data-parallel-size {analysis.data_parallel_size}")
            cmd_parts.append(f"--block-size {analysis.block_size}")
        else:
            cmd_parts = [
                "vllm serve",
                model,
                f"--dtype {analysis.dtype}",
                f"--gpu-memory-utilization {analysis.gpu_memory_utilization}",
                f"--max-model-len {analysis.max_model_len}",
                f"--max-num-seqs {analysis.max_num_seqs}",
                f"--max-num-batched-tokens {analysis.max_num_batched_tokens}",
                f"--tensor-parallel-size {analysis.tensor_parallel_size}",
            ]
            if analysis.data_parallel_size > 1:
                cmd_parts.append(f"--data-parallel-size {analysis.data_parallel_size}")
            if analysis.pipeline_parallel_size > 1:
                cmd_parts.append(f"--pipeline-parallel-size {analysis.pipeline_parallel_size}")
            if analysis.enable_expert_parallel:
                cmd_parts.append("--enable-expert-parallel")
            if analysis.enable_chunked_prefill:
                cmd_parts.append("--enable-chunked-prefill")
            else:
                cmd_parts.append("--no-enable-chunked-prefill")
            if analysis.enable_prefix_caching:
                cmd_parts.append("--enable-prefix-caching")
            else:
                cmd_parts.append("--no-enable-prefix-caching")
            cmd_parts.append(f"--kv-cache-dtype {analysis.kv_cache_dtype}")
            cmd_parts.append(f"--block-size {analysis.block_size}")
            cmd_parts.append(f"--swap-space {analysis.swap_space}")
            if analysis.cpu_offload_gb > 0:
                cmd_parts.append(f"--cpu-offload-gb {analysis.cpu_offload_gb}")
            if analysis.enforce_eager:
                cmd_parts.append("--enforce-eager")

        cmd_str = " \\\n    ".join(cmd_parts)
        console.print(
            Panel(
                Syntax(cmd_str, "bash", theme="monokai", word_wrap=True),
                title="[bold]Suggested Command[/bold]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Warnings
        if analysis.warnings:
            console.print()
            for warning in analysis.warnings:
                print_warning(warning)

        # Tip
        console.print()
        console.print(
            "  [dim]Tip: vLLM 0.17+ supports --performance-mode {balanced,interactivity,throughput} "
            "which auto-tunes batching parameters.[/dim]"
        )

        print_footer()


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
