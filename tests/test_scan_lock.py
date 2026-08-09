"""Tests for scan_lock module — in-process per-adapter scan serialization."""

import asyncio

import pytest

from bleak_connection_manager.const import ScanLockConfig
from bleak_connection_manager.scan_lock import (
    ScanLock,
    acquire_scan_lock,
    release_scan_lock,
)


@pytest.mark.asyncio
async def test_acquire_disabled_returns_none():
    cfg = ScanLockConfig(enabled=False)
    assert await acquire_scan_lock(cfg, "hci0") is None


@pytest.mark.asyncio
async def test_acquire_and_release():
    cfg = ScanLockConfig(enabled=True)
    lock = await acquire_scan_lock(cfg, "hci0")
    assert lock is not None
    assert lock.locked()
    release_scan_lock(lock)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_release_none_is_noop():
    release_scan_lock(None)


@pytest.mark.asyncio
async def test_double_release_is_safe():
    cfg = ScanLockConfig(enabled=True)
    lock = await acquire_scan_lock(cfg, "hci0")
    release_scan_lock(lock)
    release_scan_lock(lock)  # Must not raise


@pytest.mark.asyncio
async def test_nonblocking_attempt_fails_while_held():
    """lock_timeout=0 is a non-blocking attempt."""
    cfg = ScanLockConfig(enabled=True)
    held = await acquire_scan_lock(cfg, "hci0")
    assert held is not None

    instant = ScanLockConfig(enabled=True, lock_timeout=0.0)
    assert await acquire_scan_lock(instant, "hci0") is None

    release_scan_lock(held)
    second = await acquire_scan_lock(instant, "hci0")
    assert second is not None
    release_scan_lock(second)


@pytest.mark.asyncio
async def test_timeout_expiry_degrades_to_none():
    cfg = ScanLockConfig(enabled=True, lock_timeout=0.1)
    held = await acquire_scan_lock(cfg, "hci0")
    assert held is not None

    assert await acquire_scan_lock(cfg, "hci0") is None

    release_scan_lock(held)


@pytest.mark.asyncio
async def test_waiter_acquires_after_release():
    cfg = ScanLockConfig(enabled=True, lock_timeout=5.0)
    held = await acquire_scan_lock(cfg, "hci0")

    async def waiter():
        lock = await acquire_scan_lock(cfg, "hci0")
        assert lock is not None
        release_scan_lock(lock)
        return True

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert not task.done()  # Blocked on the held lock

    release_scan_lock(held)
    assert await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_adapters_are_independent():
    cfg = ScanLockConfig(enabled=True, lock_timeout=0.0)
    lock0 = await acquire_scan_lock(cfg, "hci0")
    lock1 = await acquire_scan_lock(cfg, "hci1")
    assert lock0 is not None
    assert lock1 is not None
    assert lock0 is not lock1
    release_scan_lock(lock0)
    release_scan_lock(lock1)


@pytest.mark.asyncio
async def test_none_adapter_defaults_to_hci0():
    cfg = ScanLockConfig(enabled=True, lock_timeout=0.0)
    lock = await acquire_scan_lock(cfg, None)
    assert lock is not None

    assert await acquire_scan_lock(cfg, "hci0") is None  # Same lock
    release_scan_lock(lock)


@pytest.mark.asyncio
async def test_scan_lock_context_manager():
    cfg = ScanLockConfig(enabled=True)
    async with ScanLock(cfg, "hci0") as sl:
        assert sl.acquired
        instant = ScanLockConfig(enabled=True, lock_timeout=0.0)
        assert await acquire_scan_lock(instant, "hci0") is None

    # Released on exit
    lock = await acquire_scan_lock(ScanLockConfig(enabled=True), "hci0")
    assert lock is not None
    release_scan_lock(lock)


@pytest.mark.asyncio
async def test_scan_lock_context_manager_disabled():
    cfg = ScanLockConfig(enabled=False)
    async with ScanLock(cfg, "hci0") as sl:
        assert not sl.acquired


@pytest.mark.asyncio
async def test_serializes_concurrent_scans():
    """Concurrent lock holders on one adapter never overlap."""
    cfg = ScanLockConfig(enabled=True, lock_timeout=5.0)
    active = 0
    max_active = 0

    async def scan():
        nonlocal active, max_active
        lock = await acquire_scan_lock(cfg, "hci0")
        assert lock is not None
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        release_scan_lock(lock)

    await asyncio.gather(scan(), scan(), scan())
    assert max_active == 1
