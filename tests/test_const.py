"""Tests for const module."""

from bleak_connection_manager.const import (
    DEFAULT_MAX_ATTEMPTS,
    DISCONNECT_TIMEOUT,
    IS_LINUX,
    THREAD_SAFETY_TIMEOUT,
    LockConfig,
)


def test_lock_config_defaults():
    config = LockConfig()
    assert config.enabled is False
    assert config.lock_dir == "/run"
    assert config.lock_timeout == 15.0
    assert config.max_slots == 2


def test_lock_config_path_for_slot():
    config = LockConfig()
    path = config.path_for_slot("hci0", 0)
    assert path == "/run/bleak-cm-hci0-slot-0.lock"


def test_lock_config_path_for_adapter():
    """path_for_adapter is backwards-compatible alias for slot 0."""
    config = LockConfig()
    path = config.path_for_adapter("hci0")
    assert path == "/run/bleak-cm-hci0-slot-0.lock"


def test_lock_config_path_for_none_adapter():
    config = LockConfig()
    path = config.path_for_slot(None, 0)
    assert path == "/run/bleak-cm-default-slot-0.lock"


def test_lock_config_custom():
    config = LockConfig(
        enabled=True,
        lock_dir="/tmp",
        lock_template="test-{adapter}-{slot}.lock",
        lock_timeout=5.0,
        max_slots=4,
    )
    assert config.enabled is True
    assert config.max_slots == 4
    assert config.path_for_slot("hci1", 2) == "/tmp/test-hci1-2.lock"


def test_constants():
    assert DEFAULT_MAX_ATTEMPTS == 4
    assert DISCONNECT_TIMEOUT == 5.0
    assert THREAD_SAFETY_TIMEOUT == 45.0
    assert isinstance(IS_LINUX, bool)
