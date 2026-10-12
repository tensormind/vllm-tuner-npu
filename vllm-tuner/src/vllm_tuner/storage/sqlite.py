from __future__ import annotations

import json
from pathlib import Path

from vllm_tuner.core.models import TrialResult
from vllm_tuner.helper.logging import get_logger
from vllm_tuner.storage.base import StorageBackend


logger = get_logger()


class SQLiteStorage(StorageBackend):
    """SQLite-based storage using Optuna's built-in SQLite support.

    Optuna handles core study/trial persistence via the storage URL.
    Extended trial data (hardware stats, telemetry) is stored as JSON
    sidecar files alongside the SQLite database.
    """

    def __init__(self, db_path: str = "study.db"):
        self._db_path = Path(db_path)
        self._sidecar_dir = self._db_path.parent / f"{self._db_path.stem}_data"
        logger.info("SQLiteStorage initialized: {}", self._db_path)

    def get_storage_url(self) -> str:
        return f"sqlite:///{self._db_path}"

    def save_trial(self, study_name: str, result: TrialResult) -> None:
        """Save extended trial data as a JSON sidecar file."""
        trial_dir = self._sidecar_dir / study_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_file = trial_dir / f"trial_{result.trial_number}.json"

        data = result.model_dump(mode="json")
        trial_file.write_text(json.dumps(data, indent=2, default=str))
        logger.debug("SQLiteStorage: saved trial #{} to {}", result.trial_number, trial_file)

    def load_trials(self, study_name: str) -> list[TrialResult]:
        """Load extended trial data from JSON sidecar files."""
        trial_dir = self._sidecar_dir / study_name
        if not trial_dir.exists():
            logger.debug("SQLiteStorage: no sidecar data for study '{}'", study_name)
            return []

        results = []
        for trial_file in sorted(trial_dir.glob("trial_*.json")):
            try:
                data = json.loads(trial_file.read_text())
                results.append(TrialResult.model_validate(data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("SQLiteStorage: failed to load {}: {}", trial_file, e)

        logger.info("SQLiteStorage: loaded {} trials for study '{}'", len(results), study_name)
        return results

    @property
    def name(self) -> str:
        return "sqlite"
