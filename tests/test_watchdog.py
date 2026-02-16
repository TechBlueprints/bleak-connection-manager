"""Tests for watchdog module."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bleak_connection_manager.watchdog import ConnectionWatchdog


@pytest.mark.asyncio
async def test_watchdog_start_stop():
    watchdog = ConnectionWatchdog(timeout=10.0)
    assert not watchdog.is_running
    watchdog.start()
    assert watchdog.is_running
    watchdog.stop()
    await asyncio.sleep(0.1)
    assert not watchdog.is_running


@pytest.mark.asyncio
async def test_watchdog_start_idempotent():
    watchdog = ConnectionWatchdog(timeout=10.0)
    watchdog.start()
    watchdog.start()  # Should be no-op
    assert watchdog.is_running
    watchdog.stop()


def test_watchdog_notify_activity():
    watchdog = ConnectionWatchdog(timeout=10.0)
    before = time.monotonic()
    watchdog.notify_activity()
    after = time.monotonic()
    assert before <= watchdog.last_activity <= after


@pytest.mark.asyncio
async def test_watchdog_fires_on_timeout():
    callback = AsyncMock()
    watchdog = ConnectionWatchdog(timeout=0.2, on_timeout=callback)
    watchdog.start()
    await asyncio.sleep(0.5)
    callback.assert_called_once()
    assert not watchdog.is_running


@pytest.mark.asyncio
async def test_watchdog_activity_prevents_timeout():
    callback = AsyncMock()
    watchdog = ConnectionWatchdog(timeout=0.4, on_timeout=callback)
    watchdog.start()
    await asyncio.sleep(0.2)
    watchdog.notify_activity()
    await asyncio.sleep(0.2)
    watchdog.notify_activity()
    await asyncio.sleep(0.2)
    watchdog.stop()
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_stop_before_start():
    watchdog = ConnectionWatchdog(timeout=10.0)
    watchdog.stop()  # Should be safe


@pytest.mark.asyncio
async def test_watchdog_callback_exception_handled():
    async def bad_callback():
        raise RuntimeError("oops")

    watchdog = ConnectionWatchdog(timeout=0.1, on_timeout=bad_callback)
    watchdog.start()
    await asyncio.sleep(0.5)
    # Should not propagate
    assert not watchdog.is_running
