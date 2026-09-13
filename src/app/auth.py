"""Login for the Streamlit app: named accounts, or a shared password.

Starts simple (one shared password via APP_PASSWORD or
.streamlit/secrets.toml) so a fresh checkout still runs. Once any named
account exists in the UserStore, login switches to username+password for
everyone -- named accounts let a departing staff member's access be revoked
without changing everyone else's password, which a single shared password
can't do.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

from src.app.users import UserStore

PASSWORD_ENV_VAR = "APP_PASSWORD"


def _configured_password() -> str | None:
    if PASSWORD_ENV_VAR in os.environ:
        return os.environ[PASSWORD_ENV_VAR]
    try:
        return st.secrets["app_password"]
    except (KeyError, FileNotFoundError):
        return None


def require_login(users: UserStore) -> bool:
    """Render a login gate; return True once the user is authenticated.

    Delegates to per-account login if any accounts exist, otherwise to the
    single shared-password gate. If neither is configured, the app is left
    open (with a warning) rather than locking everyone out of a fresh
    checkout.
    """
    if st.session_state.get("authenticated"):
        return True

    if not users.is_empty():
        return _require_account_login(users)
    return _require_shared_password()


def _require_account_login(users: UserStore) -> bool:
    st.header("🔐 Login required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if username and password:
        if users.verify(username, password):
            st.session_state["authenticated"] = True
            st.session_state["authenticated_user"] = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    return False


def _require_shared_password() -> bool:
    expected = _configured_password()
    if expected is None:
        st.warning(
            f"No login configured (set the {PASSWORD_ENV_VAR} environment variable, "
            "or add a named account once logged in) — running without login protection."
        )
        return True

    st.header("🔐 Login required")
    password = st.text_input("Password", type="password")
    if password:
        if hmac.compare_digest(password, expected):
            st.session_state["authenticated"] = True
            st.session_state["authenticated_user"] = None
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def log_out() -> None:
    st.session_state["authenticated"] = False
    st.session_state["authenticated_user"] = None


def render_account_management(users: UserStore) -> None:
    """Sidebar UI (for an already-authenticated user) to add/remove accounts."""
    current_user = st.session_state.get("authenticated_user")
    with st.sidebar.expander("Manage accounts"):
        existing = users.list_usernames()
        st.caption(f"{len(existing)} account(s): {', '.join(existing) or 'none yet'}")
        if not existing:
            st.caption(
                f"No named accounts yet — login currently uses the shared "
                f"{PASSWORD_ENV_VAR}. Adding one here switches everyone to "
                "per-account login."
            )

        new_username = st.text_input("New username", key="new_account_username")
        new_password = st.text_input("New password", type="password", key="new_account_password")
        if st.button("Add account", disabled=not (new_username and new_password)):
            users.add_user(new_username, new_password)
            st.success(f"Added account '{new_username.strip()}'.")
            st.rerun()

        removable = [name for name in existing if name != current_user]
        if removable:
            to_remove = st.selectbox("Remove an account", removable, key="account_to_remove")
            if st.button("Remove account"):
                users.remove_user(to_remove)
                st.success(f"Removed account '{to_remove}'.")
                st.rerun()
