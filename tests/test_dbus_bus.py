"""Tests for dbus_bus module — wait_for_bluez and helpers."""

from unittest.mock import AsyncMock, patch

import pytest

import bleak_connection_manager.dbus_bus as dbus_bus_mod


@pytest.fixture(autouse=True)
def _reset_bluez_ready():
    """Reset the module-level _bluez_ready flag between tests."""
    dbus_bus_mod._bluez_ready = False
    yield
    dbus_bus_mod._bluez_ready = False


# ── wait_for_bluez: non-Linux ─────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.dbus_bus.IS_LINUX", False)
async def test_wait_for_bluez_non_linux():
    assert await dbus_bus_mod.wait_for_bluez() is True


# ── wait_for_bluez: BlueZ immediately available ───────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.dbus_bus.IS_LINUX", True)
@patch(
    "bleak_connection_manager.dbus_bus._ping_bluez",
    new_callable=AsyncMock,
    return_value=True,
)
@patch(
    "bleak_connection_manager.dbus_bus._poll_bluez",
    new_callable=AsyncMock,
    return_value=True,
)
async def test_wait_for_bluez_immediate(mock_poll, mock_ping):
    """First call — no cache, poll succeeds immediately."""
    assert await dbus_bus_mod.wait_for_bluez() is True
    mock_poll.assert_awaited_once()
    assert dbus_bus_mod._bluez_ready is True


# ── wait_for_bluez: cached + still alive ──────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.dbus_bus.IS_LINUX", True)
@patch(
    "bleak_connection_manager.dbus_bus._ping_bluez",
    new_callable=AsyncMock,
    return_value=True,
)
async def test_wait_for_bluez_cached_still_alive(mock_ping):
    """Cache says ready, ping confirms — returns immediately."""
    dbus_bus_mod._bluez_ready = True
    assert await dbus_bus_mod.wait_for_bluez() is True
    mock_ping.assert_awaited_once()


# ── wait_for_bluez: cached but bluetoothd crashed ────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.dbus_bus.IS_LINUX", True)
@patch(
    "bleak_connection_manager.dbus_bus._ping_bluez",
    new_callable=AsyncMock,
    return_value=False,
)
@patch(
    "bleak_connection_manager.dbus_bus._poll_bluez",
    new_callable=AsyncMock,
    return_value=True,
)
async def test_wait_for_bluez_cache_invalidated(mock_poll, mock_ping):
    """Cache says ready, but ping fails — invalidates cache and re-polls."""
    dbus_bus_mod._bluez_ready = True
    assert await dbus_bus_mod.wait_for_bluez() is True
    mock_ping.assert_awaited_once()
    mock_poll.assert_awaited_once()
    assert dbus_bus_mod._bluez_ready is True


# ── wait_for_bluez: poll times out, restart succeeds ─────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.dbus_bus.IS_LINUX", True)
@patch(
    "bleak_connection_manager.dbus_bus._ping_bluez",
    new_callable=AsyncMock,
    return_value=False,
)
async def test_wait_for_bluez_restart_on_timeout(mock_ping):
    """Poll exhausts, restart_bluetoothd succeeds, second poll succeeds."""
    poll_results = iter([False, True])

    async def fake_poll(timeout, interval):
        return next(poll_results)

    with (
        patch(
            "bleak_connection_manager.dbus_bus._poll_bluez",
            side_effect=fake_poll,
        ),
        patch(
            "bleak_connection_manager.recovery.restart_bluetoothd",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_restart,
    ):
        assert await dbus_bus_mod.wait_for_bluez(timeout=0.1) is True
        mock_restart.assert_awaited_once()
    assert dbus_bus_mod._bluez_ready is True


# ── wait_for_bluez: poll times out, restart fails ────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.dbus_bus.IS_LINUX", True)
@patch(
    "bleak_connection_manager.dbus_bus._ping_bluez",
    new_callable=AsyncMock,
    return_value=False,
)
@patch(
    "bleak_connection_manager.dbus_bus._poll_bluez",
    new_callable=AsyncMock,
    return_value=False,
)
async def test_wait_for_bluez_restart_fails(mock_poll, mock_ping):
    """Poll exhausts, restart fails — returns False."""
    with patch(
        "bleak_connection_manager.recovery.restart_bluetoothd",
        new_callable=AsyncMock,
        return_value=False,
    ):
        assert await dbus_bus_mod.wait_for_bluez(timeout=0.1) is False
    assert dbus_bus_mod._bluez_ready is False


# ── wait_for_bluez: restart succeeds but BlueZ still missing ─────


@pytest.mark.asyncio
@patch("bleak_connection_manager.dbus_bus.IS_LINUX", True)
@patch(
    "bleak_connection_manager.dbus_bus._ping_bluez",
    new_callable=AsyncMock,
    return_value=False,
)
@patch(
    "bleak_connection_manager.dbus_bus._poll_bluez",
    new_callable=AsyncMock,
    return_value=False,
)
async def test_wait_for_bluez_restart_ok_but_bluez_missing(mock_poll, mock_ping):
    """Restart succeeds but BlueZ never appears on D-Bus — returns False."""
    with patch(
        "bleak_connection_manager.recovery.restart_bluetoothd",
        new_callable=AsyncMock,
        return_value=True,
    ):
        assert await dbus_bus_mod.wait_for_bluez(timeout=0.1) is False
    assert dbus_bus_mod._bluez_ready is False
