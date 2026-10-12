from __future__ import annotations

from vllm_tuner.core.models import TrialResult
from vllm_tuner.helper.logging import get_logger
from vllm_tuner.storage.base import StorageBackend


logger = get_logger()


class PostgreSQLStorage(StorageBackend):
    """PostgreSQL-based storage for production deployments.

    Currently a mock — returns connection URL but does not actually connect.
    Real implementation will use psycopg2 or asyncpg.
    """

    def __init__(self, connection_url: str = "postgresql://user:pass@localhost:5432/vllm_tuner"):
        self._connection_url = connection_url
        logger.info("PostgreSQLStorage initialized: {}", self._sanitize_url(connection_url))

    def get_storage_url(self) -> str:
        return self._connection_url

    def save_trial(self, study_name: str, result: TrialResult) -> None:
        logger.info("PostgreSQLStorage: would save trial #{} for study '{}'", result.trial_number, study_name)
        # TODO: save extended trial data to PostgreSQL

    def load_trials(self, study_name: str) -> list[TrialResult]:
        logger.info("PostgreSQLStorage: would load trials for study '{}'", study_name)
        # TODO: load extended trial data from PostgreSQL
        return []

    @property
    def name(self) -> str:
        return "postgresql"

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove password from URL for logging."""
        if "@" in url and ":" in url.split("@")[0]:
            prefix, rest = url.split("@", 1)
            user_part = prefix.rsplit(":", 1)[0]
            return f"{user_part}:***@{rest}"
        return url
