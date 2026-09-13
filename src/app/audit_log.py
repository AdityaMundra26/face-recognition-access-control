"""Append-only audit log of access-check attempts.

Each entry records who (if anyone) matched, at what similarity, and whether
access was granted, so there's a record of every access attempt made through
the app.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_LOG_PATH = Path("data/enrolled_faces/access_log.jsonl")


@dataclass(frozen=True)
class AccessEvent:
    """One recorded access-check attempt."""

    timestamp: str  # UTC ISO-8601
    name: str | None
    similarity: float
    granted: bool
    # None for a normal recognition outcome; e.g. "liveness_failed" when
    # access was denied before recognition even ran. Optional (and last)
    # so older log lines without it still parse.
    reason: str | None = None


class AuditLog:
    """Append-only JSON-Lines log of access-check attempts."""

    def __init__(self, path: str | Path = DEFAULT_LOG_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        name: str | None,
        similarity: float,
        granted: bool,
        reason: str | None = None,
    ) -> AccessEvent:
        """Append one access event and return it."""
        event = AccessEvent(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            name=name,
            similarity=similarity,
            granted=granted,
            reason=reason,
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        return event

    def recent(self, limit: int = 50) -> list[AccessEvent]:
        """Return up to ``limit`` most recent events, newest first."""
        if not self._path.is_file():
            return []
        lines = [line for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]
        events = [AccessEvent(**json.loads(line)) for line in lines[-limit:]]
        return list(reversed(events))
