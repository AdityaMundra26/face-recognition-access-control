"""Tests for src.app.lockout.check_lockout."""

from __future__ import annotations

import datetime as dt

import pytest

from src.app import audit_log as audit_log_module
from src.app.audit_log import AuditLog
from src.app.lockout import COOLDOWN, MAX_FAILURES, WINDOW, check_lockout


@pytest.fixture
def audit(tmp_path) -> AuditLog:
    return AuditLog(path=tmp_path / "access_log.jsonl")


def _record_at(audit, monkeypatch, when, *, granted, reason=None):
    monkeypatch.setattr(audit_log_module, "_utcnow", lambda: when)
    audit.record(name=None, similarity=0.0, granted=granted, reason=reason)


def test_no_lockout_with_no_history(audit):
    status = check_lockout(audit)
    assert status.locked is False
    assert status.recent_failures == 0
    assert status.retry_after is None


def test_no_lockout_below_failure_threshold(audit, monkeypatch):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for _ in range(MAX_FAILURES - 1):
        _record_at(audit, monkeypatch, now, granted=False)

    status = check_lockout(audit, now=now)

    assert status.locked is False
    assert status.recent_failures == MAX_FAILURES - 1


def test_locked_out_at_failure_threshold(audit, monkeypatch):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for _ in range(MAX_FAILURES):
        _record_at(audit, monkeypatch, now, granted=False)

    status = check_lockout(audit, now=now)

    assert status.locked is True
    assert status.recent_failures == MAX_FAILURES
    assert status.retry_after == now + COOLDOWN


def test_successful_attempts_dont_count_as_failures(audit, monkeypatch):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for _ in range(MAX_FAILURES):
        _record_at(audit, monkeypatch, now, granted=True)

    status = check_lockout(audit, now=now)

    assert status.locked is False
    assert status.recent_failures == 0


def test_failures_outside_the_window_dont_count(audit, monkeypatch):
    old = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for _ in range(MAX_FAILURES):
        _record_at(audit, monkeypatch, old, granted=False)

    later = old + WINDOW + dt.timedelta(seconds=1)
    status = check_lockout(audit, now=later)

    assert status.locked is False
    assert status.recent_failures == 0


def test_unlocks_once_cooldown_passes(audit, monkeypatch):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for _ in range(MAX_FAILURES):
        _record_at(audit, monkeypatch, now, granted=False)

    after_cooldown = now + COOLDOWN + dt.timedelta(seconds=1)
    status = check_lockout(audit, now=after_cooldown)

    assert status.locked is False
