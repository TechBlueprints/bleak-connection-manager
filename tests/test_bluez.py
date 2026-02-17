"""Tests for bluez module."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bleak_connection_manager.bluez import (
    _power_cycle_adapter_with_cooldown,
    address_to_bluez_path,
    ensure_adapter_scan_ready,
    get_adapter_discovering,
    is_inactive_connection,
    power_cycle_adapter,
    verified_disconnect,
)


def test_address_to_bluez_path_default_adapter():
    path = address_to_bluez_path("AA:BB:CC:DD:EE:FF")
    assert path == "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"


def test_address_to_bluez_path_specific_adapter():
    path = address_to_bluez_path("AA:BB:CC:DD:EE:FF", "hci1")
    assert path == "/org/bluez/hci1/dev_AA_BB_CC_DD_EE_FF"


def test_address_to_bluez_path_lowercase_normalized():
    path = address_to_bluez_path("aa:bb:cc:dd:ee:ff")
    assert path == "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_is_inactive_connection_non_linux():
    result = await is_inactive_connection("AA:BB:CC:DD:EE:FF")
    assert result is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_is_inactive_connection_connected_not_resolved(mock_props):
    mock_props.return_value = {
        "Connected": True,
        "ServicesResolved": False,
    }
    result = await is_inactive_connection("AA:BB:CC:DD:EE:FF")
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_is_inactive_connection_connected_and_resolved(mock_props):
    mock_props.return_value = {
        "Connected": True,
        "ServicesResolved": True,
    }
    result = await is_inactive_connection("AA:BB:CC:DD:EE:FF")
    assert result is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_is_inactive_connection_not_connected(mock_props):
    mock_props.return_value = {
        "Connected": False,
        "ServicesResolved": False,
    }
    result = await is_inactive_connection("AA:BB:CC:DD:EE:FF")
    assert result is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_is_inactive_connection_no_props(mock_props):
    mock_props.return_value = None
    result = await is_inactive_connection("AA:BB:CC:DD:EE:FF")
    assert result is False


# ── verified_disconnect tests ──────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_verified_disconnect_non_linux():
    """Non-Linux always returns True (assume OK)."""
    result = await verified_disconnect("AA:BB:CC:DD:EE:FF")
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.disconnect_device", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_verified_disconnect_immediate_success(mock_props, mock_disconnect):
    """Device becomes disconnected on first poll."""
    mock_props.return_value = {"Connected": False}

    result = await verified_disconnect(
        "AA:BB:CC:DD:EE:FF", timeout=2.0, poll_interval=0.1
    )
    assert result is True
    mock_disconnect.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.disconnect_device", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_verified_disconnect_device_gone(mock_props, mock_disconnect):
    """Device disappears from D-Bus (props returns None)."""
    mock_props.return_value = None

    result = await verified_disconnect(
        "AA:BB:CC:DD:EE:FF", timeout=2.0, poll_interval=0.1
    )
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.remove_device", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.disconnect_device", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_verified_disconnect_escalates_to_remove(
    mock_props, mock_disconnect, mock_remove
):
    """Device stays Connected=True past timeout, escalates to remove_device."""
    mock_props.return_value = {"Connected": True}

    result = await verified_disconnect(
        "AA:BB:CC:DD:EE:FF", timeout=0.3, poll_interval=0.1
    )
    assert result is False
    mock_remove.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.disconnect_device", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._get_device_properties")
async def test_verified_disconnect_eventually_disconnects(
    mock_props, mock_disconnect
):
    """Device is still connected on first poll, then disconnects."""
    mock_props.side_effect = [
        {"Connected": True},
        {"Connected": True},
        {"Connected": False},
    ]

    result = await verified_disconnect(
        "AA:BB:CC:DD:EE:FF", timeout=2.0, poll_interval=0.1
    )
    assert result is True


# ── get_adapter_discovering tests ──────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_get_adapter_discovering_non_linux():
    result = await get_adapter_discovering("hci0")
    assert result is None


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
async def test_get_adapter_discovering_true():
    from dbus_fast import MessageType

    mock_reply = MagicMock()
    mock_reply.message_type = MessageType.METHOD_RETURN
    val = MagicMock()
    val.value = True
    mock_reply.body = [val]

    mock_bus = AsyncMock()
    mock_bus.call.return_value = mock_reply

    with patch("bleak_connection_manager.dbus_bus.get_bus", return_value=mock_bus):
        result = await get_adapter_discovering("hci0")
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
async def test_get_adapter_discovering_false():
    from dbus_fast import MessageType

    mock_reply = MagicMock()
    mock_reply.message_type = MessageType.METHOD_RETURN
    val = MagicMock()
    val.value = False
    mock_reply.body = [val]

    mock_bus = AsyncMock()
    mock_bus.call.return_value = mock_reply

    with patch("bleak_connection_manager.dbus_bus.get_bus", return_value=mock_bus):
        result = await get_adapter_discovering("hci0")
    assert result is False


# ── power_cycle_adapter tests ─────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_power_cycle_adapter_non_linux():
    result = await power_cycle_adapter("hci0")
    assert result is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
async def test_power_cycle_adapter_success():
    from dbus_fast import MessageType

    mock_reply = MagicMock()
    mock_reply.message_type = MessageType.METHOD_RETURN

    mock_bus = AsyncMock()
    mock_bus.call.return_value = mock_reply

    with (
        patch("bleak_connection_manager.dbus_bus.get_bus", return_value=mock_bus),
        patch("bleak_connection_manager.bluez._POWER_CYCLE_SETTLE", 0.0),
    ):
        result = await power_cycle_adapter("hci0")

    assert result is True
    # Called twice: Powered=False and Powered=True
    assert mock_bus.call.call_count == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
async def test_power_cycle_adapter_failure_on_power_off():
    from dbus_fast import MessageType

    mock_reply = MagicMock()
    mock_reply.message_type = MessageType.ERROR
    mock_reply.body = ["Failed"]

    mock_bus = AsyncMock()
    mock_bus.call.return_value = mock_reply

    with patch("bleak_connection_manager.dbus_bus.get_bus", return_value=mock_bus):
        result = await power_cycle_adapter("hci0")

    assert result is False


# ── power_cycle_adapter_with_cooldown tests ────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.power_cycle_adapter", new_callable=AsyncMock)
async def test_power_cycle_cooldown_allows_first(mock_cycle):
    import bleak_connection_manager.bluez as bluez_mod

    # Clear any previous cooldown state
    bluez_mod._last_power_cycle.pop("hci_test_cd", None)
    mock_cycle.return_value = True

    result = await _power_cycle_adapter_with_cooldown("hci_test_cd")
    assert result is True
    mock_cycle.assert_called_once_with("hci_test_cd")

    # Clean up
    bluez_mod._last_power_cycle.pop("hci_test_cd", None)


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.power_cycle_adapter", new_callable=AsyncMock)
async def test_power_cycle_cooldown_blocks_rapid(mock_cycle):
    import bleak_connection_manager.bluez as bluez_mod

    # Simulate a recent power-cycle
    bluez_mod._last_power_cycle["hci_test_cd2"] = time.monotonic()
    mock_cycle.return_value = True

    result = await _power_cycle_adapter_with_cooldown("hci_test_cd2")
    assert result is False
    mock_cycle.assert_not_called()

    # Clean up
    bluez_mod._last_power_cycle.pop("hci_test_cd2", None)


# ── ensure_adapter_scan_ready tests ────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_ensure_adapter_scan_ready_non_linux():
    result = await ensure_adapter_scan_ready("hci0")
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._probe_start_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_clean(mock_disc, mock_probe):
    """Clean adapter — no power-cycle needed."""
    mock_disc.return_value = False
    mock_probe.return_value = True

    result = await ensure_adapter_scan_ready("hci0")
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._power_cycle_adapter_with_cooldown", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_discovering_true(mock_disc, mock_cycle):
    """Discovering=True — should power-cycle and re-check."""
    # First call returns True (stale), second call (after cycle) returns False
    mock_disc.side_effect = [True, False]
    mock_cycle.return_value = True

    result = await ensure_adapter_scan_ready("hci0")
    assert result is True
    mock_cycle.assert_called_once_with("hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._power_cycle_adapter_with_cooldown", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_discovering_true_cycle_fails(
    mock_disc, mock_cycle
):
    """Discovering=True but power-cycle cooldown active — returns False."""
    mock_disc.return_value = True
    mock_cycle.return_value = False

    result = await ensure_adapter_scan_ready("hci0")
    assert result is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._power_cycle_adapter_with_cooldown", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._probe_start_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_stale_inprogress(
    mock_disc, mock_probe, mock_cycle
):
    """Discovering=False but probe returns InProgress — power-cycle."""
    mock_disc.return_value = False
    mock_probe.return_value = False  # InProgress
    mock_cycle.return_value = True

    result = await ensure_adapter_scan_ready("hci0")
    assert result is True
    mock_cycle.assert_called_once_with("hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._power_cycle_adapter_with_cooldown", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._probe_start_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_stale_inprogress_cycle_fails(
    mock_disc, mock_probe, mock_cycle
):
    """Discovering=False + InProgress but cooldown active — returns False."""
    mock_disc.return_value = False
    mock_probe.return_value = False
    mock_cycle.return_value = False

    result = await ensure_adapter_scan_ready("hci0")
    assert result is False
