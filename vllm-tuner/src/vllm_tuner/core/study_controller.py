from __future__ import annotations

from typing import Any

import optuna

from vllm_tuner.core.models import Direction, SamplerType, StudyConfig, TrialResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StudyController:
    """Orchestrates an Optuna study for vLLM hyperparameter tuning.

    Currently a mock — creates real Optuna study but uses stub objective.
    """

    def __init__(self, config: StudyConfig):
        self._config = config
        self._study: optuna.Study | None = None
        self._results: list[TrialResult] = []

    def create_study(self, storage_url: str | None = None) -> optuna.Study:
        """Create or load an Optuna study based on the configuration."""
        directions = [
            "maximize" if obj.direction == Direction.MAXIMIZE else "minimize"
            for obj in self._config.optimization.objectives
        ]
        sampler = self._create_sampler(
            self._config.optimization.sampler,
            n_startup_trials=self._config.optimization.n_startup_trials,
        )

        effective_storage = storage_url or self._config.study.storage
        if effective_storage == "sqlite:///study.db":
            effective_storage = None

        self._study = optuna.create_study(
            study_name=self._config.study.name,
            storage=effective_storage,
            directions=directions,
            sampler=sampler,
            load_if_exists=True,
        )
        logger.info(
            "Study '{}' created/loaded (directions={}, sampler={})",
            self._config.study.name,
            directions,
            self._config.optimization.sampler.value,
        )
        return self._study

    def enqueue_recommended_trial(self) -> bool:
        """Enqueue a trial with recommended parameters as the first trial.

        Uses the model analyzer to compute intelligent defaults, then maps
        them to the study's configured parameter space (clamping to valid
        ranges).  Returns True if a trial was enqueued.
        """
        if self._study is None:
            return False
        if not self._config.model or not self._config.parameters:
            return False

        try:
            from vllm_tuner.utils.model_analyzer import analyze_model
            from vllm_tuner.utils.model_registry import get_model_config

            hf_config = get_model_config(self._config.model)
        except Exception as exc:
            logger.debug("Cannot get model config for recommended trial: {}", exc)
            return False

        hw = self._config.hardware
        num_gpus = len(hw.device_ids) if hw.device_ids else 1
        # Rough per-GPU VRAM default (will be overridden by real detection)
        per_gpu_vram = 24.0
        total_vram = per_gpu_vram * num_gpus

        analysis = analyze_model(hf_config, total_vram_gb=total_vram, num_gpus=num_gpus, model_id=self._config.model)

        # Map ModelAnalysis fields → study parameter names
        recommended = {
            "gpu_memory_utilization": analysis.gpu_memory_utilization,
            "max_num_seqs": analysis.max_num_seqs,
            "max_num_batched_tokens": analysis.max_num_batched_tokens,
            "enable_chunked_prefill": str(analysis.enable_chunked_prefill).lower(),
            "enable_prefix_caching": str(analysis.enable_prefix_caching).lower(),
            "swap_space": analysis.swap_space,
            "block_size": str(analysis.block_size),
            "enforce_eager": str(analysis.enforce_eager).lower(),
        }

        # Build enqueue dict: only include params that are in the study's search space,
        # and clamp numeric values to valid ranges / snap to valid options.
        enqueue_params: dict[str, float | str] = {}
        for param_spec in self._config.parameters:
            name = param_spec.name
            if name not in recommended:
                continue
            value = recommended[name]

            if param_spec.options is not None:
                # Categorical — pick the option if it matches, else skip
                str_val = str(value)
                if str_val in param_spec.options:
                    enqueue_params[name] = str_val
            elif param_spec.min is not None and param_spec.max is not None:
                # Numeric — clamp to [min, max] and snap to step grid
                num_val = float(value)
                num_val = max(param_spec.min, min(param_spec.max, num_val))
                if param_spec.step is not None and param_spec.step > 0:
                    steps = round((num_val - param_spec.min) / param_spec.step)
                    num_val = param_spec.min + steps * param_spec.step
                    num_val = min(num_val, param_spec.max)
                enqueue_params[name] = num_val

        if not enqueue_params:
            logger.debug("No recommended params overlap with search space")
            return False

        self._study.enqueue_trial(enqueue_params)
        logger.info("Enqueued recommended trial as first trial: {}", enqueue_params)
        return True

    def optimize(
        self,
        objective_fn: Any | None = None,
        execution_backend: Any | None = None,
        trial_runner: Any | None = None,
        optimizer: Any | None = None,
    ) -> None:
        """Run the optimization loop.

        If objective_fn is provided, uses it directly with Optuna.
        Otherwise, builds a default objective from the execution backend + trial runner.
        """
        if self._study is None:
            self.create_study()

        n_trials = self._config.optimization.n_trials
        logger.info("Starting optimization: n_trials={}", n_trials)

        if objective_fn is None:
            objective_fn = self._build_objective(execution_backend, trial_runner, optimizer)

        self._study.optimize(objective_fn, n_trials=n_trials)
        logger.info("Optimization complete: {} trials finished", len(self._study.trials))

    def _build_objective(
        self,
        execution_backend: Any | None = None,
        trial_runner: Any | None = None,
        optimizer_instance: Any | None = None,
    ) -> Any:
        """Build an Optuna objective function from configured components."""
        from vllm_tuner.core.models import TrialConfig
        from vllm_tuner.core.optimizer import Optimizer

        opt = optimizer_instance or Optimizer(
            objectives=self._config.optimization.objectives,
            constraints=self._config.constraints,
        )

        def objective(trial: optuna.Trial) -> list[float]:
            # Suggest values for each parameter
            params = {}
            for param_spec in self._config.parameters:
                if param_spec.options is not None:
                    params[param_spec.name] = trial.suggest_categorical(param_spec.name, param_spec.options)
                elif param_spec.min is not None and param_spec.max is not None:
                    if param_spec.step is not None:
                        params[param_spec.name] = trial.suggest_float(
                            param_spec.name,
                            param_spec.min,
                            param_spec.max,
                            step=param_spec.step,
                        )
                    elif param_spec.log_scale:
                        params[param_spec.name] = trial.suggest_float(
                            param_spec.name,
                            param_spec.min,
                            param_spec.max,
                            log=True,
                        )
                    else:
                        params[param_spec.name] = trial.suggest_float(
                            param_spec.name,
                            param_spec.min,
                            param_spec.max,
                        )

            trial_config = TrialConfig(
                trial_number=trial.number,
                parameters=params,
                static_parameters=self._config.static_parameters,
            )

            # Run trial via backend or runner
            if execution_backend is not None:
                handle = execution_backend.submit_trial(trial_config)
                results, _ = execution_backend.poll_trials([handle])
                result = results[0] if results else None
            elif trial_runner is not None:
                result = trial_runner.run_trial(trial_config)
            else:
                logger.info("No execution backend or trial runner — dry-run trial #{}", trial.number)
                result = None

            if result is None:
                return [0.0] * len(self._config.optimization.objectives)

            self.add_result(result)

            # Set constraint values (resolved from metrics + params) for both
            # reporting (user_attrs, margin>0 = satisfied) and the Optuna sampler
            # (set_constraint, value<=0 = feasible).
            constraint_margins = opt.evaluate_constraints(params, result)
            trial.set_user_attr("constraint_values", constraint_margins)
            for idx, c in enumerate(self._config.constraints):
                trial.set_constraint(c.expression, -constraint_margins[idx])

            # Return objective values
            values = opt.compute_objective_values(result)
            logger.info("Trial #{}: objective values = {}", trial.number, values)
            return values

        return objective

    def get_best_trials(self, n: int = 5) -> list[optuna.trial.FrozenTrial]:
        """Get the top N trials from the study."""
        if self._study is None:
            return []
        try:
            return self._study.best_trials[:n]
        except Exception:
            return sorted(self._study.trials, key=lambda t: t.values or [0], reverse=True)[:n]

    def get_study_summary(self) -> dict:
        """Get a summary of the study state."""
        if self._study is None:
            return {"status": "not_started"}

        trials = self._study.trials
        return {
            "study_name": self._config.study.name,
            "n_trials_completed": len(trials),
            "n_trials_target": self._config.optimization.n_trials,
            "best_values": [t.values for t in self.get_best_trials(1)],
        }

    def add_result(self, result: TrialResult) -> None:
        """Track a trial result for reporting."""
        self._results.append(result)

    @property
    def results(self) -> list[TrialResult]:
        return list(self._results)

    @property
    def study(self) -> optuna.Study | None:
        return self._study

    @staticmethod
    def _create_sampler(
        sampler_type: SamplerType,
        n_startup_trials: int = 10,
    ) -> optuna.samplers.BaseSampler:
        match sampler_type:
            case SamplerType.TPE:
                return optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
            case SamplerType.NSGA2:
                return optuna.samplers.NSGAIISampler()
            case SamplerType.RANDOM:
                return optuna.samplers.RandomSampler()
            case SamplerType.GRID:
                return optuna.samplers.GridSampler({})
            case _:
                logger.warning("Unknown sampler '{}', falling back to TPE", sampler_type)
                return optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
