"""Lockout after repeated failed access-check attempts.

Mirrors a physical door controller's brute-force deterrent: once too many
failed checks land in the audit log within a short window, further access
checks are blocked until a cooldown passes. This locks out the whole access
check flow rather than tracking individual identities, because most
failures are unrecognized faces, which have no stable identity to rate
limit by.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from src.app.audit_log import AuditLog

MAX_FAILURES = 5
WINDOW = dt.timedelta(minutes=5)
COOLDOWN = dt.timedelta(minutes=2)


@dataclass(frozen=True)
class LockoutStatus:
    """Whether access checks are currently locked out, and why."""

    locked: bool
    recent_failures: int
    retry_after: dt.datetime | None  # None unless locked


def check_lockout(audit: AuditLog, *, now: dt.datetime | None = None) -> LockoutStatus:
    """Look at recent audit history and decide whether checks are locked out."""
    now = now or dt.datetime.now(dt.timezone.utc)
    window_start = now - WINDOW

    recent_failure_times: list[dt.datetime] = []
    for event in audit.recent(limit=200):  # newest-first
        timestamp = dt.datetime.fromisoformat(event.timestamp)
        if timestamp < window_start:
            break
        if not event.granted:
            recent_failure_times.append(timestamp)

    failure_count = len(recent_failure_times)
    if failure_count < MAX_FAILURES:
        return LockoutStatus(locked=False, recent_failures=failure_count, retry_after=None)

    retry_after = recent_failure_times[0] + COOLDOWN  # newest failure + cooldown
    locked = now < retry_after
    return LockoutStatus(
        locked=locked,
        recent_failures=failure_count,
        retry_after=retry_after if locked else None,
    )
