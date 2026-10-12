import pytest

from vllm_tuner.storage.base import StorageBackend
from vllm_tuner.storage.postgresql import PostgreSQLStorage
from vllm_tuner.storage.sqlite import SQLiteStorage


class TestStorageBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            StorageBackend()


class TestSQLiteStorage:
    def test_name(self):
        storage = SQLiteStorage()
        assert storage.name == "sqlite"

    def test_storage_url(self):
        storage = SQLiteStorage("my_study.db")
        assert storage.get_storage_url() == "sqlite:///my_study.db"

    def test_default_path(self):
        storage = SQLiteStorage()
        assert "study.db" in storage.get_storage_url()

    def test_save_and_load(self, sample_trial_result, tmp_path):
        db_path = str(tmp_path / "study.db")
        storage = SQLiteStorage(db_path)
        storage.save_trial("test-study", sample_trial_result)
        trials = storage.load_trials("test-study")
        assert len(trials) == 1
        assert trials[0].trial_number == sample_trial_result.trial_number
        assert trials[0].benchmark.throughput_req_per_sec == sample_trial_result.benchmark.throughput_req_per_sec

    def test_load_trials_empty(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        storage = SQLiteStorage(db_path)
        trials = storage.load_trials("nonexistent")
        assert trials == []


class TestPostgreSQLStorage:
    def test_name(self):
        storage = PostgreSQLStorage()
        assert storage.name == "postgresql"

    def test_storage_url(self):
        url = "postgresql://user:pass@host:5432/db"
        storage = PostgreSQLStorage(url)
        assert storage.get_storage_url() == url

    def test_sanitize_url(self):
        sanitized = PostgreSQLStorage._sanitize_url("postgresql://user:secret@host:5432/db")
        assert "secret" not in sanitized
        assert "***" in sanitized

    def test_sanitize_url_no_password(self):
        sanitized = PostgreSQLStorage._sanitize_url("postgresql://host:5432/db")
        assert sanitized == "postgresql://host:5432/db"

    def test_save_trial(self, sample_trial_result):
        storage = PostgreSQLStorage()
        storage.save_trial("test-study", sample_trial_result)

    def test_load_trials_empty(self):
        storage = PostgreSQLStorage()
        assert storage.load_trials("test-study") == []
