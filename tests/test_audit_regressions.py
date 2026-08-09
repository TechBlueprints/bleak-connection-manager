"""Regression tests for the August 2026 audit findings.

Each test pins one confirmed bug:

1. External cancellation must propagate out of ``establish_connection``.
   Previously ``asyncio.CancelledError`` was swallowed by the retry
   loop, which also defeated ``overall_timeout`` (``asyncio.wait_for``
   cancels the inner task and then waits for it to actually stop).
2. ``BleakNotFoundError`` must raise immediately per the documented
   contract.  Previously the ``except BleakError`` clause shadowed it
   (``BleakNotFoundError`` is a ``BleakError`` subclass) and the
   not-found was retried through the full escalation chain.
3. ``diagnose_stuck_state`` must not block the event loop while
   running hcitool subprocess queries.
4. ``ConnectionWatchdog`` cleanup must target the device's own
   adapter, not an hci0 default.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak import BleakClient
from bleak.backends.device import BLEDevice
from bleak_retry_connector import BleakNotFoundError

from bleak_connection_manager.connection import establish_connection
from bleak_connection_manager.diagnostics import diagnose_stuck_state
from bleak_connection_manager.watchdog import ConnectionWatchdog


def _make_device(address="AA:BB:CC:DD:EE:FF", name="TestDevice", adapter="hci0"):
    return BLEDevice(
        address,
        name,
        {"path": f"/org/bluez/{adapter}/dev_{address.replace(':', '_')}"},
    )


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection.IS_LINUX", False)
@patch("bleak_connection_manager.connection._brc_establish_connection")
async def test_external_cancel_propagates(mock_brc):
    """task.cancel() from the caller must stop the retry loop immediately."""
    started = asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        await asyncio.sleep(30)

    mock_brc.side_effect = hang

    task = asyncio.create_task(
        establish_connection(BleakClient, _make_device(), "T", max_attempts=3)
    )
    await started.wait()
    task.cancel()

    done, _ = await asyncio.wait({task}, timeout=1.0)
    try:
        assert task in done, "cancellation was swallowed; task kept retrying"
        assert task.cancelled()
    finally:
        # Unstick the broken-code case so a failing test doesn't leak the task
        while not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=1.0)
        if not task.cancelled():
            task.exception()


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection.IS_LINUX", False)
@patch("bleak_connection_manager.connection._brc_establish_connection")
async def test_not_found_raises_immediately(mock_brc):
    """BleakNotFoundError must not be retried — documented contract."""
    mock_brc.side_effect = BleakNotFoundError("Device not found")

    with pytest.raises(BleakNotFoundError):
        await establish_connection(
            BleakClient, _make_device(), "T", max_attempts=3
        )

    assert mock_brc.call_count == 1


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch(
    "bleak_connection_manager.diagnostics._get_device_properties",
    new_callable=AsyncMock,
)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address")
@patch("bleak_connection_manager.diagnostics.hci_available")
async def test_diagnose_stuck_state_does_not_block_loop(
    mock_avail, mock_find, mock_props
):
    """Slow hcitool queries must run off the event loop thread."""
    mock_props.return_value = None
    mock_find.return_value = None

    def slow_hci_query(adapter):
        time.sleep(0.4)
        return True

    mock_avail.side_effect = slow_hci_query

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    ticker_task.cancel()

    assert ticks >= 5, "event loop was blocked during the hcitool query"


@pytest.mark.asyncio
@patch(
    "bleak_connection_manager.watchdog.remove_device", new_callable=AsyncMock
)
@patch(
    "bleak_connection_manager.watchdog.verified_disconnect",
    new_callable=AsyncMock,
)
async def test_watchdog_cleanup_uses_device_adapter(mock_vd, mock_rd):
    """Cleanup for an hci1 device must target hci1, not the hci0 default."""
    client = MagicMock()
    client.disconnect = AsyncMock()
    device = _make_device(adapter="hci1")

    watchdog = ConnectionWatchdog(timeout=30.0, client=client, device=device)
    await watchdog._cleanup_connection()

    def _adapter_arg(call):
        if "adapter" in call.kwargs:
            return call.kwargs["adapter"]
        return call.args[1] if len(call.args) > 1 else None

    assert _adapter_arg(mock_vd.call_args) == "hci1"
    assert _adapter_arg(mock_rd.call_args) == "hci1"
