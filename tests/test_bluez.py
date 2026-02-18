"""Tests for bluez module."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bleak_connection_manager.bluez import (
    _get_adapter_powered,
    _hciconfig_up,
    _power_cycle_adapter_with_cooldown,
    _set_adapter_powered,
    address_to_bluez_path,
    ensure_adapter_scan_ready,
    ensure_adapters_up,
    get_adapter_discovering,
    get_connected_devices,
    is_inactive_connection,
    power_cycle_adapter,
    try_stop_discovery,
    verified_disconnect,
)
from bleak_connection_manager.const import AdapterScanState


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
@patch("bleak_connection_manager.bluez._get_adapter_powered", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._set_adapter_powered", new_callable=AsyncMock)
async def test_power_cycle_adapter_success(mock_set, mock_get):
    """Happy path: power off succeeds, power on succeeds, verification passes."""
    mock_set.return_value = True
    mock_get.return_value = True

    with patch("bleak_connection_manager.bluez._POWER_CYCLE_SETTLE", 0.0):
        result = await power_cycle_adapter("hci0")

    assert result is True
    assert mock_set.call_count == 2
    mock_set.assert_any_call("hci0", False)
    mock_set.assert_any_call("hci0", True)


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._set_adapter_powered", new_callable=AsyncMock)
async def test_power_cycle_adapter_failure_on_power_off(mock_set):
    mock_set.return_value = False

    result = await power_cycle_adapter("hci0")

    assert result is False
    mock_set.assert_called_once_with("hci0", False)


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._hciconfig_up")
@patch("bleak_connection_manager.bluez._get_adapter_powered", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._set_adapter_powered", new_callable=AsyncMock)
async def test_power_cycle_adapter_dbus_retry_succeeds(mock_set, mock_get, mock_hci):
    """Powered=True fails first, but D-Bus retry succeeds."""
    mock_set.side_effect = [True, True, True]
    mock_get.side_effect = [False, True]

    with patch("bleak_connection_manager.bluez._POWER_CYCLE_SETTLE", 0.0):
        result = await power_cycle_adapter("hci0")

    assert result is True
    mock_hci.assert_not_called()


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._hciconfig_up")
@patch("bleak_connection_manager.bluez._get_adapter_powered", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._set_adapter_powered", new_callable=AsyncMock)
async def test_power_cycle_adapter_hciconfig_fallback(mock_set, mock_get, mock_hci):
    """D-Bus Powered=True fails twice, hciconfig fallback recovers."""
    mock_set.side_effect = [True, True, True]
    mock_get.side_effect = [False, False, True]
    mock_hci.return_value = True

    with patch("bleak_connection_manager.bluez._POWER_CYCLE_SETTLE", 0.0):
        result = await power_cycle_adapter("hci0")

    assert result is True
    mock_hci.assert_called_once_with("hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._hciconfig_up")
@patch("bleak_connection_manager.bluez._get_adapter_powered", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._set_adapter_powered", new_callable=AsyncMock)
async def test_power_cycle_adapter_all_recovery_fails(mock_set, mock_get, mock_hci):
    """All recovery attempts fail — adapter left DOWN."""
    mock_set.side_effect = [True, True, True]
    mock_get.return_value = False
    mock_hci.return_value = False

    with patch("bleak_connection_manager.bluez._POWER_CYCLE_SETTLE", 0.0):
        result = await power_cycle_adapter("hci0")

    assert result is False
    mock_hci.assert_called_once_with("hci0")


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


# ── try_stop_discovery tests ───────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_try_stop_discovery_non_linux():
    result = await try_stop_discovery("hci0")
    assert result is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
async def test_try_stop_discovery_success():
    from dbus_fast import MessageType

    mock_reply = MagicMock()
    mock_reply.message_type = MessageType.METHOD_RETURN

    mock_bus = AsyncMock()
    mock_bus.call.return_value = mock_reply

    with patch("bleak_connection_manager.dbus_bus.get_bus", return_value=mock_bus):
        result = await try_stop_discovery("hci0")
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
async def test_try_stop_discovery_fails():
    from dbus_fast import MessageType

    mock_reply = MagicMock()
    mock_reply.message_type = MessageType.ERROR
    mock_reply.body = ["org.bluez.Error.Failed"]

    mock_bus = AsyncMock()
    mock_bus.call.return_value = mock_reply

    with patch("bleak_connection_manager.dbus_bus.get_bus", return_value=mock_bus):
        result = await try_stop_discovery("hci0")
    assert result is False


# ── get_connected_devices tests ────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_get_connected_devices_non_linux():
    result = await get_connected_devices("hci0")
    assert result == []


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
async def test_get_connected_devices_returns_connected():
    from dbus_fast import MessageType

    addr1 = MagicMock()
    addr1.value = "AA:BB:CC:DD:EE:FF"
    conn1 = MagicMock()
    conn1.value = True
    addr2 = MagicMock()
    addr2.value = "11:22:33:44:55:66"
    conn2 = MagicMock()
    conn2.value = False

    objects = {
        "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF": {
            "org.bluez.Device1": {"Address": addr1, "Connected": conn1},
        },
        "/org/bluez/hci0/dev_11_22_33_44_55_66": {
            "org.bluez.Device1": {"Address": addr2, "Connected": conn2},
        },
        "/org/bluez/hci1/dev_99_88_77_66_55_44": {
            "org.bluez.Device1": {"Address": MagicMock(value="99:88:77:66:55:44"),
                                   "Connected": MagicMock(value=True)},
        },
    }

    mock_reply = MagicMock()
    mock_reply.message_type = MessageType.METHOD_RETURN
    mock_reply.body = [objects]

    mock_bus = AsyncMock()
    mock_bus.call.return_value = mock_reply

    with patch("bleak_connection_manager.dbus_bus.get_bus", return_value=mock_bus):
        result = await get_connected_devices("hci0")

    assert result == ["AA:BB:CC:DD:EE:FF"]


# ── ensure_adapter_scan_ready tests (tiered — no direct power-cycle) ─


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_ensure_adapter_scan_ready_non_linux():
    result = await ensure_adapter_scan_ready("hci0")
    assert result == AdapterScanState.READY


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._probe_start_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_clean(mock_disc, mock_probe):
    """Clean adapter — no recovery needed."""
    mock_disc.return_value = False
    mock_probe.return_value = True

    result = await ensure_adapter_scan_ready("hci0")
    assert result == AdapterScanState.READY


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.try_stop_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_discovering_true_stop_clears(
    mock_disc, mock_stop
):
    """Discovering=True — StopDiscovery clears it (Tier 1 success)."""
    mock_disc.side_effect = [True, False]
    mock_stop.return_value = True

    result = await ensure_adapter_scan_ready("hci0")
    assert result == AdapterScanState.READY
    mock_stop.assert_called_once_with("hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.try_stop_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_discovering_true_stop_fails(
    mock_disc, mock_stop
):
    """Discovering=True — StopDiscovery doesn't help, returns STUCK."""
    mock_disc.side_effect = [True, True]
    mock_stop.return_value = False

    result = await ensure_adapter_scan_ready("hci0")
    assert result == AdapterScanState.STUCK
    mock_stop.assert_called_once_with("hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.try_stop_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._probe_start_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_inprogress_stop_clears(
    mock_disc, mock_probe, mock_stop
):
    """Hidden InProgress — StopDiscovery clears it (Tier 1 success)."""
    mock_disc.return_value = False
    mock_probe.side_effect = [False, True]
    mock_stop.return_value = True

    result = await ensure_adapter_scan_ready("hci0")
    assert result == AdapterScanState.READY
    mock_stop.assert_called_once_with("hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.try_stop_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._probe_start_discovery", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_adapter_discovering", new_callable=AsyncMock)
async def test_ensure_adapter_scan_ready_inprogress_stop_fails(
    mock_disc, mock_probe, mock_stop
):
    """Hidden InProgress — StopDiscovery doesn't help, returns EXTERNAL_SCAN."""
    mock_disc.return_value = False
    mock_probe.side_effect = [False, False]
    mock_stop.return_value = False

    result = await ensure_adapter_scan_ready("hci0")
    assert result == AdapterScanState.EXTERNAL_SCAN


# ── _power_cycle_adapter_with_cooldown connection awareness ────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez.power_cycle_adapter", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez.get_connected_devices", new_callable=AsyncMock)
async def test_power_cycle_warns_about_active_connections(
    mock_connected, mock_cycle
):
    """Power-cycle logs warning when adapter has active connections."""
    import bleak_connection_manager.bluez as bluez_mod

    bluez_mod._last_power_cycle.pop("hci_test_warn", None)
    mock_connected.return_value = ["AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66"]
    mock_cycle.return_value = True

    result = await _power_cycle_adapter_with_cooldown("hci_test_warn")
    assert result is True
    mock_connected.assert_called_once_with("hci_test_warn")
    mock_cycle.assert_called_once_with("hci_test_warn")

    bluez_mod._last_power_cycle.pop("hci_test_warn", None)


# ── ensure_adapters_up tests ──────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", False)
async def test_ensure_adapters_up_non_linux():
    """Non-Linux is a no-op."""
    await ensure_adapters_up(["hci0", "hci1"])


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._get_adapter_powered", new_callable=AsyncMock)
async def test_ensure_adapters_up_all_up(mock_get):
    """All adapters already UP — nothing to do."""
    mock_get.return_value = True
    await ensure_adapters_up(["hci0", "hci1"])
    assert mock_get.call_count == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._hciconfig_up")
@patch("bleak_connection_manager.bluez._set_adapter_powered", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._get_adapter_powered", new_callable=AsyncMock)
async def test_ensure_adapters_up_recovers_down_adapter(mock_get, mock_set, mock_hci):
    """One adapter DOWN — D-Bus brings it back."""
    mock_get.side_effect = [True, False, True]
    mock_set.return_value = True

    await ensure_adapters_up(["hci0", "hci1"])
    mock_set.assert_called_once_with("hci1", True)
    mock_hci.assert_not_called()


@pytest.mark.asyncio
@patch("bleak_connection_manager.bluez.IS_LINUX", True)
@patch("bleak_connection_manager.bluez._hciconfig_up")
@patch("bleak_connection_manager.bluez._set_adapter_powered", new_callable=AsyncMock)
@patch("bleak_connection_manager.bluez._get_adapter_powered", new_callable=AsyncMock)
async def test_ensure_adapters_up_hciconfig_fallback(mock_get, mock_set, mock_hci):
    """D-Bus power-on fails, hciconfig fallback succeeds."""
    mock_get.side_effect = [False, False, True]
    mock_set.return_value = True

    await ensure_adapters_up(["hci0"])
    mock_hci.assert_called_once_with("hci0")
