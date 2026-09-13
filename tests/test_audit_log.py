"""Tests for src.app.audit_log.AuditLog."""

from __future__ import annotations

import pytest

from src.app import AuditLog


@pytest.fixture
def log(tmp_path) -> AuditLog:
    return AuditLog(path=tmp_path / "access_log.jsonl")


def test_empty_log_has_no_recent_events(log):
    assert log.recent() == []


def test_record_persists_fields(log):
    event = log.record(name="Alice", similarity=0.91, granted=True)
    assert event.name == "Alice"
    assert event.similarity == 0.91
    assert event.granted is True
    assert event.timestamp  # non-empty ISO timestamp


def test_record_with_no_match_stores_name_none(log):
    event = log.record(name=None, similarity=0.12, granted=False)
    assert event.name is None
    assert event.granted is False


def test_recent_returns_newest_first(log):
    log.record(name="Alice", similarity=0.9, granted=True)
    log.record(name=None, similarity=0.1, granted=False)
    log.record(name="Bob", similarity=0.6, granted=True)

    names = [event.name for event in log.recent()]

    assert names == ["Bob", None, "Alice"]


def test_recent_respects_limit(log):
    for i in range(5):
        log.record(name=f"Person{i}", similarity=0.5, granted=True)

    events = log.recent(limit=2)

    assert [e.name for e in events] == ["Person4", "Person3"]


def test_log_file_created_on_first_write(log, tmp_path):
    path = tmp_path / "access_log.jsonl"
    assert not path.exists()
    log.record(name="Alice", similarity=0.9, granted=True)
    assert path.is_file()
