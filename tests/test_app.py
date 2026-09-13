"""Smoke test for the Streamlit app, via Streamlit's AppTest harness.

Runs the real app script (real InsightFace/chromadb, no mocking) but without
a browser or server, and against the default persist paths under
data/enrolled_faces/ since AppTest.from_file executes main.py as-is.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "app" / "main.py"


def test_app_loads_without_exceptions(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert not at.exception


def test_app_has_enroll_and_access_tabs(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert [tab.label for tab in at.tabs] == ["Enroll", "Check access"]


def test_app_title(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert at.title[0].value == "Face Recognition Access Control"


def test_app_without_password_configured_warns_but_stays_open(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert any("without login protection" in w.value for w in at.warning)
    assert at.tabs != []


def test_app_with_password_configured_blocks_until_login(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert at.tabs == []
    assert any("Login required" in h.value for h in at.header)


def test_app_wrong_password_shows_error_and_stays_locked(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()
    at.text_input[0].input("wrong-password").run()

    assert at.tabs == []
    assert any("Incorrect password" in e.value for e in at.error)


def test_app_correct_password_unlocks_tabs(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()
    at.text_input[0].input("s3cret").run()

    assert [tab.label for tab in at.tabs] == ["Enroll", "Check access"]
