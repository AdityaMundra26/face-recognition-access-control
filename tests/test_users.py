"""Tests for src.app.users.UserStore."""

from __future__ import annotations

import pytest

from src.app.users import UserStore


@pytest.fixture
def users(tmp_path) -> UserStore:
    return UserStore(path=tmp_path / "users.json")


def test_new_store_is_empty(users):
    assert users.is_empty() is True
    assert users.list_usernames() == []


def test_add_user_then_verify_succeeds(users):
    users.add_user("alice", "hunter2")
    assert users.is_empty() is False
    assert users.verify("alice", "hunter2") is True


def test_verify_wrong_password_fails(users):
    users.add_user("alice", "hunter2")
    assert users.verify("alice", "wrong") is False


def test_verify_unknown_user_fails(users):
    assert users.verify("nobody", "anything") is False


def test_add_user_rejects_empty_username(users):
    with pytest.raises(ValueError, match="username"):
        users.add_user("   ", "hunter2")


def test_add_user_rejects_empty_password(users):
    with pytest.raises(ValueError, match="password"):
        users.add_user("alice", "")


def test_add_user_strips_username(users):
    users.add_user("  alice  ", "hunter2")
    assert users.list_usernames() == ["alice"]


def test_add_user_overwrites_existing_password(users):
    users.add_user("alice", "old-password")
    users.add_user("alice", "new-password")
    assert users.verify("alice", "old-password") is False
    assert users.verify("alice", "new-password") is True


def test_remove_user_returns_true_and_removes(users):
    users.add_user("alice", "hunter2")
    assert users.remove_user("alice") is True
    assert users.verify("alice", "hunter2") is False
    assert users.is_empty() is True


def test_remove_missing_user_returns_false(users):
    assert users.remove_user("nobody") is False


def test_list_usernames_sorted(users):
    users.add_user("bob", "pw")
    users.add_user("alice", "pw")
    assert users.list_usernames() == ["alice", "bob"]
