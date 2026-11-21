"""The store guard: an unsafe test-db path must abort the run, not warn."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest
from openmind import db


def test_env_override_is_used(monkeypatch, tmp_path):
    target = tmp_path / "openmind-test-alt.db"
    monkeypatch.setenv(conftest.TEST_DB_ENV, str(target))
    assert conftest.resolve_test_db() == target.resolve()


def test_default_is_beside_the_tests(monkeypatch):
    monkeypatch.delenv(conftest.TEST_DB_ENV, raising=False)
    assert conftest.resolve_test_db() == conftest.DEFAULT_TEST_DB


def test_refuses_a_name_that_does_not_say_test(monkeypatch, tmp_path):
    monkeypatch.setenv(conftest.TEST_DB_ENV, str(tmp_path / "openmind.db"))
    with pytest.raises(BaseException) as caught:
        conftest.resolve_test_db()
    assert "TEST STORE REFUSED" in str(caught.value)


def test_refuses_the_live_store(monkeypatch, tmp_path):
    # Naming is not the only gate: a path the server could be using is refused
    # even when its filename says "test".
    target = tmp_path / "openmind-test.db"
    monkeypatch.setenv(conftest.TEST_DB_ENV, str(target))
    monkeypatch.setattr(conftest, "_live_store_paths", lambda: {target.resolve()})
    with pytest.raises(BaseException) as caught:
        conftest.resolve_test_db()
    assert "TEST STORE REFUSED" in str(caught.value)
    assert "live store" in str(caught.value)


def test_refuses_the_default_store_filename(monkeypatch):
    # First line of defence: "openmind.db" fails the naming check before the
    # live-store check is even reached.
    monkeypatch.setenv(
        conftest.TEST_DB_ENV, str(conftest.REPO_ROOT / db.DEFAULT_PATH)
    )
    with pytest.raises(BaseException) as caught:
        conftest.resolve_test_db()
    assert "TEST STORE REFUSED" in str(caught.value)


def test_refuses_a_directory(monkeypatch, tmp_path):
    target = tmp_path / "test-store-dir"
    target.mkdir()
    monkeypatch.setenv(conftest.TEST_DB_ENV, str(target))
    with pytest.raises(BaseException) as caught:
        conftest.resolve_test_db()
    assert "TEST STORE REFUSED" in str(caught.value)


def test_destroy_removes_journals_before_the_database(tmp_path):
    path = tmp_path / "openmind-test-fixture.db"
    path.write_bytes(b"db")
    Path(f"{path}-wal").write_bytes(b"wal")
    Path(f"{path}-shm").write_bytes(b"shm")

    conftest._destroy(path)

    assert not path.exists()
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
