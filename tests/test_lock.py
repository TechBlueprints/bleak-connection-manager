"""Tests for lock module — slot-based cross-process locking."""

import asyncio
import os
import tempfile

import pytest

from bleak_connection_manager.const import LockConfig
from bleak_connection_manager.lock import (
    acquire_lock,
    acquire_slot,
    release_lock,
    release_slot,
)


# ── Basic acquire / release ────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_release_slot():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=2.0,
        )
        fd = await acquire_slot(config, "hci0")
        assert fd is not None
        release_slot(fd)


@pytest.mark.asyncio
async def test_acquire_slot_disabled():
    config = LockConfig(enabled=False)
    fd = await acquire_slot(config, "hci0")
    assert fd is None


@pytest.mark.asyncio
async def test_release_slot_none():
    release_slot(None)


# ── Backwards-compatible aliases ───────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_release_lock_aliases():
    """acquire_lock / release_lock are aliases for acquire_slot / release_slot."""
    assert acquire_lock is acquire_slot
    assert release_lock is release_slot

    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=2.0,
        )
        fd = await acquire_lock(config, "hci0")
        assert fd is not None
        release_lock(fd)


# ── Slot file creation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slot_file_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=2.0,
            max_slots=3,
        )
        fd = await acquire_slot(config, "hci0")
        assert fd is not None
        # The slot file for slot 0 should exist
        slot_path = config.path_for_slot("hci0", 0)
        assert os.path.exists(slot_path)
        release_slot(fd)


# ── Multiple slots ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_slots_concurrent():
    """Multiple slots can be held simultaneously."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=2.0,
            max_slots=3,
        )
        fd1 = await acquire_slot(config, "hci0")
        fd2 = await acquire_slot(config, "hci0")
        fd3 = await acquire_slot(config, "hci0")

        assert fd1 is not None
        assert fd2 is not None
        assert fd3 is not None

        # All three should be different fds
        assert len({fd1, fd2, fd3}) == 3

        release_slot(fd1)
        release_slot(fd2)
        release_slot(fd3)


@pytest.mark.asyncio
async def test_all_slots_busy_times_out():
    """When all slots are held, acquire_slot times out gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=0.5,  # Short timeout for testing
            max_slots=1,
        )
        # Hold the only slot
        fd1 = await acquire_slot(config, "hci0")
        assert fd1 is not None

        # Try to acquire another — should time out and return None
        fd2 = await acquire_slot(config, "hci0")
        assert fd2 is None

        release_slot(fd1)


@pytest.mark.asyncio
async def test_slot_released_then_reacquired():
    """After releasing a slot, it can be acquired again."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=2.0,
            max_slots=1,
        )
        fd1 = await acquire_slot(config, "hci0")
        assert fd1 is not None
        release_slot(fd1)

        fd2 = await acquire_slot(config, "hci0")
        assert fd2 is not None
        release_slot(fd2)


@pytest.mark.asyncio
async def test_max_slots_one_is_strict_serialization():
    """max_slots=1 behaves like the old binary lock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=0.4,
            max_slots=1,
        )
        fd1 = await acquire_slot(config, "hci0")
        assert fd1 is not None

        # Second acquire should fail (timeout)
        fd2 = await acquire_slot(config, "hci0")
        assert fd2 is None

        release_slot(fd1)

        # Now it should succeed
        fd3 = await acquire_slot(config, "hci0")
        assert fd3 is not None
        release_slot(fd3)


# ── Different adapters are independent ─────────────────────────────


@pytest.mark.asyncio
async def test_different_adapters_independent():
    """Slots for different adapters don't interfere."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            lock_timeout=0.4,
            max_slots=1,
        )
        fd_hci0 = await acquire_slot(config, "hci0")
        fd_hci1 = await acquire_slot(config, "hci1")

        assert fd_hci0 is not None
        assert fd_hci1 is not None

        release_slot(fd_hci0)
        release_slot(fd_hci1)


# ── LockConfig path generation ─────────────────────────────────────


def test_lock_config_path_for_slot():
    config = LockConfig(lock_dir="/run")
    path = config.path_for_slot("hci0", 2)
    assert path == "/run/bleak-cm-hci0-slot-2.lock"


def test_lock_config_path_for_adapter_backwards_compat():
    config = LockConfig(lock_dir="/run")
    path = config.path_for_adapter("hci0")
    assert path == "/run/bleak-cm-hci0-slot-0.lock"


def test_lock_config_path_for_none_adapter():
    config = LockConfig(lock_dir="/run")
    path = config.path_for_slot(None, 0)
    assert path == "/run/bleak-cm-default-slot-0.lock"


def test_lock_config_defaults():
    config = LockConfig()
    assert config.enabled is False
    assert config.max_slots == 2
    assert config.lock_timeout == 15.0


def test_lock_config_custom():
    config = LockConfig(
        enabled=True,
        lock_dir="/tmp/test",
        lock_timeout=5.0,
        max_slots=4,
    )
    assert config.max_slots == 4
    assert config.lock_dir == "/tmp/test"
