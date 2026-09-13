"""A tiny local account store: usernames + bcrypt-hashed passwords.

Lets the app move beyond one shared APP_PASSWORD to named accounts, so a
departing staff member's access to the enroll/remove UI can be revoked
without changing everyone else's password. Deliberately minimal: one flat
JSON file, no roles, no password reset flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import bcrypt

DEFAULT_USERS_PATH = Path("data/enrolled_faces/users.json")


class UserStore:
    """JSON-file-backed store of local app accounts (username -> bcrypt hash)."""

    def __init__(self, path: str | Path = DEFAULT_USERS_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, str]:
        if not self._path.is_file():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, str]) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_user(self, username: str, password: str) -> None:
        """Create or overwrite an account. Raises ValueError on empty input."""
        username = username.strip()
        if not username:
            raise ValueError("username must be a non-empty string")
        if not password:
            raise ValueError("password must be a non-empty string")

        data = self._load()
        data[username] = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        self._save(data)

    def remove_user(self, username: str) -> bool:
        """Delete an account. Returns True if it existed."""
        data = self._load()
        if username not in data:
            return False
        del data[username]
        self._save(data)
        return True

    def verify(self, username: str, password: str) -> bool:
        """True if ``username`` exists and ``password`` matches its hash."""
        hashed = self._load().get(username)
        if hashed is None:
            return False
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    def list_usernames(self) -> list[str]:
        return sorted(self._load())

    def is_empty(self) -> bool:
        return not self._load()
