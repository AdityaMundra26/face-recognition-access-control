"""Tests for src.app.actuator.grant_access."""

from __future__ import annotations

import logging
from unittest.mock import patch

import requests

from src.app.actuator import WEBHOOK_ENV_VAR, grant_access


def test_grant_access_without_webhook_only_logs(monkeypatch, caplog):
    monkeypatch.delenv(WEBHOOK_ENV_VAR, raising=False)

    with patch("src.app.actuator.requests.post") as mock_post, caplog.at_level(logging.INFO):
        grant_access("Alice")

    mock_post.assert_not_called()
    assert any("ACCESS GRANTED for Alice" in record.message for record in caplog.records)


def test_grant_access_with_webhook_posts_expected_payload(monkeypatch):
    monkeypatch.setenv(WEBHOOK_ENV_VAR, "http://example.invalid/hook")

    with patch("src.app.actuator.requests.post") as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        grant_access("Alice")

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://example.invalid/hook"
    assert kwargs["json"] == {"event": "access_granted", "name": "Alice"}


def test_grant_access_swallows_webhook_failure(monkeypatch, caplog):
    monkeypatch.setenv(WEBHOOK_ENV_VAR, "http://example.invalid/hook")

    with patch("src.app.actuator.requests.post", side_effect=requests.RequestException("boom")):
        with caplog.at_level(logging.ERROR):
            grant_access("Alice")  # must not raise

    assert any("Failed to notify access webhook" in record.message for record in caplog.records)
