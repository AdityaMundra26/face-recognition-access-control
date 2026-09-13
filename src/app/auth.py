"""Simple password gate for the Streamlit app.

There's no user database — just one shared password, checked with a
constant-time comparison, gating the whole app (enroll + access-check) so a
random visitor can't add or remove enrolled faces. Set it via the
APP_PASSWORD environment variable, or a `app_password` entry in
.streamlit/secrets.toml.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

PASSWORD_ENV_VAR = "APP_PASSWORD"


def _configured_password() -> str | None:
    if PASSWORD_ENV_VAR in os.environ:
        return os.environ[PASSWORD_ENV_VAR]
    try:
        return st.secrets["app_password"]
    except (KeyError, FileNotFoundError):
        return None


def require_login() -> bool:
    """Render a password gate; return True once the correct password is entered.

    If no password is configured, the app is left open (with a warning)
    rather than locking everyone out of a fresh checkout.
    """
    expected = _configured_password()
    if expected is None:
        st.warning(
            f"No app password configured (set the {PASSWORD_ENV_VAR} environment "
            "variable) — running without login protection."
        )
        return True

    if st.session_state.get("authenticated"):
        return True

    st.header("🔐 Login required")
    password = st.text_input("Password", type="password")
    if password:
        if hmac.compare_digest(password, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False
