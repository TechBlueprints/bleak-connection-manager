"""Tests for scan_lock module — cross-process BLE scan serialization."""

import asyncio
import os
import tempfile
from unittest.mock import patch

import pytest

from bleak_connection_manager.const import ScanLockConfig
from bleak_connection_manager.scan_lock import (
    ScanLock,
    acquire_scan_lock,
    release_scan_lock,
)


def _make_config(tmpdir: str, **overrides) -> ScanLockConfig:
    """Create a ScanLockConfig pointing at a temporary directory."""
    defaults = {
        "enabled": True,
        "lock_dir": tmpdir,
        "lock_timeout": 2.0,
    }
    defaults.update(overrides)
    return ScanLockConfig(**defaults)


# ── ScanLockConfig tests ──────────────────────────────────────────


def test_config_path_for_adapter():
    cfg = ScanLockConfig(lock_dir="/run")
    assert cfg.path_for_adapter("hci0") == "/run/bleak-cm-hci0-scan.lock"
    assert cfg.path_for_adapter("hci1") == "/run/bleak-cm-hci1-scan.lock"


def test_config_path_for_adapter_default():
    cfg = ScanLockConfig(lock_dir="/run")
    assert cfg.path_for_adapter(None) == "/run/bleak-cm-default-scan.lock"


def test_config_custom_template():
    cfg = ScanLockConfig(
        lock_dir="/tmp",
        lock_template="my-app-{adapter}-scan.lock",
    )
    assert cfg.path_for_adapter("hci0") == "/tmp/my-app-hci0-scan.lock"


# ── acquire / release tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_and_release():
    """Basic acquire + release cycle succeeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir)
        fd = await acquire_scan_lock(cfg, "hci0")
        assert fd is not None
        assert isinstance(fd, int)
        release_scan_lock(fd)


@pytest.mark.asyncio
async def test_acquire_creates_lock_file():
    """Lock file is created on acquire."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir)
        fd = await acquire_scan_lock(cfg, "hci0")
        try:
            expected_path = os.path.join(tmpdir, "bleak-cm-hci0-scan.lock")
            assert os.path.exists(expected_path)
        finally:
            release_scan_lock(fd)


@pytest.mark.asyncio
async def test_acquire_disabled_returns_none():
    """Disabled config returns None immediately."""
    cfg = ScanLockConfig(enabled=False)
    fd = await acquire_scan_lock(cfg, "hci0")
    assert fd is None


@pytest.mark.asyncio
async def test_release_none_is_noop():
    """Releasing None is safe (no-op)."""
    release_scan_lock(None)


@pytest.mark.asyncio
async def test_release_twice_is_safe():
    """Releasing an already-released fd doesn't crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir)
        fd = await acquire_scan_lock(cfg, "hci0")
        release_scan_lock(fd)
        # Second release should be safe (fd is already closed)
        release_scan_lock(fd)


@pytest.mark.asyncio
async def test_acquire_exclusive():
    """Second acquire on same adapter blocks until first is released."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, lock_timeout=0.5)

        fd1 = await acquire_scan_lock(cfg, "hci0")
        assert fd1 is not None

        # Second acquire should time out (same adapter, exclusive)
        fd2 = await acquire_scan_lock(cfg, "hci0")
        assert fd2 is None

        # Release first, now second should succeed
        release_scan_lock(fd1)
        fd3 = await acquire_scan_lock(cfg, "hci0")
        assert fd3 is not None
        release_scan_lock(fd3)


@pytest.mark.asyncio
async def test_acquire_different_adapters_independent():
    """Locks on different adapters are independent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir)

        fd0 = await acquire_scan_lock(cfg, "hci0")
        fd1 = await acquire_scan_lock(cfg, "hci1")

        assert fd0 is not None
        assert fd1 is not None

        release_scan_lock(fd0)
        release_scan_lock(fd1)


@pytest.mark.asyncio
@patch("bleak_connection_manager.scan_lock._HAS_FCNTL", False)
async def test_acquire_no_fcntl_returns_none():
    """Without fcntl (non-Linux), returns None."""
    cfg = ScanLockConfig(enabled=True)
    fd = await acquire_scan_lock(cfg, "hci0")
    assert fd is None


# ── ScanLock context manager tests ────────────────────────────────


@pytest.mark.asyncio
async def test_context_manager_basic():
    """ScanLock context manager acquires and releases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir)

        async with ScanLock(cfg, "hci0") as lock:
            assert lock.acquired is True

        # After exit, lock should be released — we can re-acquire
        fd = await acquire_scan_lock(cfg, "hci0")
        assert fd is not None
        release_scan_lock(fd)


@pytest.mark.asyncio
async def test_context_manager_releases_on_exception():
    """ScanLock releases even when the body raises."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir)

        with pytest.raises(ValueError, match="boom"):
            async with ScanLock(cfg, "hci0"):
                raise ValueError("boom")

        # Lock should be released despite the exception
        fd = await acquire_scan_lock(cfg, "hci0")
        assert fd is not None
        release_scan_lock(fd)


@pytest.mark.asyncio
async def test_context_manager_graceful_degradation():
    """ScanLock enters even when lock can't be acquired (disabled)."""
    cfg = ScanLockConfig(enabled=False)

    async with ScanLock(cfg, "hci0") as lock:
        assert lock.acquired is False


@pytest.mark.asyncio
async def test_context_manager_exclusive():
    """Two ScanLock context managers on same adapter — second blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, lock_timeout=0.3)
        order: list[str] = []

        async def holder():
            async with ScanLock(cfg, "hci0"):
                order.append("holder_start")
                await asyncio.sleep(0.5)
                order.append("holder_end")

        async def waiter():
            # Small delay so holder acquires first
            await asyncio.sleep(0.05)
            async with ScanLock(cfg, "hci0") as lock:
                if lock.acquired:
                    order.append("waiter_acquired")
                else:
                    order.append("waiter_degraded")

        await asyncio.gather(holder(), waiter())

        # The waiter should have timed out and degraded because the
        # holder held the lock for 0.5s but waiter timeout is 0.3s
        assert "holder_start" in order
        assert "waiter_degraded" in order


@pytest.mark.asyncio
async def test_scan_lock_separate_from_connection_lock():
    """Scan lock and connection lock are independent resources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from bleak_connection_manager.const import LockConfig
        from bleak_connection_manager.lock import acquire_slot, release_slot

        conn_cfg = LockConfig(enabled=True, lock_dir=tmpdir, max_slots=1)
        scan_cfg = _make_config(tmpdir)

        # Acquire connection lock
        conn_fd = await acquire_slot(conn_cfg, "hci0")
        assert conn_fd is not None

        # Scan lock should still be acquirable (different file)
        scan_fd = await acquire_scan_lock(scan_cfg, "hci0")
        assert scan_fd is not None

        release_scan_lock(scan_fd)
        release_slot(conn_fd)
