"""Smoke test for the Streamlit app, via Streamlit's AppTest harness.

Runs the real app script (real InsightFace/chromadb, no mocking) but without
a browser or server, and against the default persist paths under
data/enrolled_faces/ since AppTest.from_file executes main.py as-is.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "app" / "main.py"


def test_app_loads_without_exceptions():
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert not at.exception


def test_app_has_enroll_and_access_tabs():
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert [tab.label for tab in at.tabs] == ["Enroll", "Check access"]


def test_app_title():
    at = AppTest.from_file(str(APP_SCRIPT))
    at.run()

    assert at.title[0].value == "Face Recognition Access Control"
