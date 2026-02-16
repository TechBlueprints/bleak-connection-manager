"""Tests for diagnostics module."""

from unittest.mock import AsyncMock, patch

import pytest

from bleak_connection_manager.diagnostics import (
    StuckState,
    clear_stuck_state,
    diagnose_stuck_state,
)


def test_stuck_state_enum_values():
    assert StuckState.NOT_STUCK.value == "not_stuck"
    assert StuckState.INACTIVE_CONNECTION.value == "inactive_connection"
    assert StuckState.STALE_CACHE.value == "stale_cache"
    assert StuckState.ORPHAN_HCI_HANDLE.value == "orphan_hci_handle"
    assert StuckState.PHANTOM_NO_HANDLE.value == "phantom_no_handle"


# ── diagnose_stuck_state tests ──────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", False)
async def test_diagnose_not_linux():
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.NOT_STUCK


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address", return_value=None)
@patch("bleak_connection_manager.diagnostics._get_device_properties", return_value=None)
async def test_diagnose_device_not_found(mock_props, mock_hci):
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.NOT_STUCK


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address", return_value=None)
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_phantom_no_handle(mock_props, mock_hci):
    """D-Bus Connected=True, ServicesResolved=False, but NO HCI handle."""
    mock_props.return_value = {"Connected": True, "ServicesResolved": False}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.PHANTOM_NO_HANDLE


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address", return_value=None)
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_phantom_services_key_absent(mock_props, mock_hci):
    """D-Bus Connected=True, ServicesResolved absent, no HCI handle."""
    mock_props.return_value = {"Connected": True}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.PHANTOM_NO_HANDLE


class _FakeHciConn:
    """Minimal stand-in for HciConnection."""
    def __init__(self, handle=64, address="AA:BB:CC:DD:EE:FF", adapter="hci0"):
        self.handle = handle
        self.address = address
        self.adapter = adapter


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address")
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_inactive_connection(mock_props, mock_hci):
    """D-Bus Connected=True, ServicesResolved=False, WITH HCI handle."""
    mock_hci.return_value = _FakeHciConn()
    mock_props.return_value = {"Connected": True, "ServicesResolved": False}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.INACTIVE_CONNECTION


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address")
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_inactive_services_missing(mock_props, mock_hci):
    """D-Bus Connected=True, ServicesResolved key absent, HCI handle exists."""
    mock_hci.return_value = _FakeHciConn()
    mock_props.return_value = {"Connected": True}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.INACTIVE_CONNECTION


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address")
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_healthy_connection(mock_props, mock_hci):
    """D-Bus Connected=True, ServicesResolved=True, HCI handle exists."""
    mock_hci.return_value = _FakeHciConn()
    mock_props.return_value = {"Connected": True, "ServicesResolved": True}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.NOT_STUCK


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address", return_value=None)
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_stale_cache(mock_props, mock_hci):
    """D-Bus exists, Connected=False, no HCI handle."""
    mock_props.return_value = {"Connected": False, "ServicesResolved": False}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.STALE_CACHE


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address", return_value=None)
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_stale_cache_not_connected_key_missing(mock_props, mock_hci):
    """D-Bus exists, Connected key absent, no HCI handle."""
    mock_props.return_value = {"Alias": "SomeDevice"}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.STALE_CACHE


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address")
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_orphan_hci_handle(mock_props, mock_hci):
    """D-Bus says not connected, but HCI handle exists."""
    mock_hci.return_value = _FakeHciConn()
    mock_props.return_value = {"Connected": False}
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.ORPHAN_HCI_HANDLE


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address")
@patch("bleak_connection_manager.diagnostics._get_device_properties", return_value=None)
async def test_diagnose_orphan_hci_no_dbus(mock_props, mock_hci):
    """Device not in D-Bus at all, but HCI handle exists."""
    mock_hci.return_value = _FakeHciConn()
    result = await diagnose_stuck_state("AA:BB:CC:DD:EE:FF", "hci0")
    assert result == StuckState.ORPHAN_HCI_HANDLE


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.find_connection_by_address", return_value=None)
@patch("bleak_connection_manager.diagnostics._get_device_properties")
async def test_diagnose_passes_adapters(mock_props, mock_hci):
    """Verify adapters list is forwarded to find_connection_by_address."""
    mock_props.return_value = None
    await diagnose_stuck_state(
        "AA:BB:CC:DD:EE:FF", "hci0", adapters=["hci0", "hci1"]
    )
    mock_hci.assert_called_once_with(
        "AA:BB:CC:DD:EE:FF", adapters=["hci0", "hci1"]
    )


# ── clear_stuck_state tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_clear_not_stuck():
    result = await clear_stuck_state("AA:BB:CC:DD:EE:FF", "hci0", StuckState.NOT_STUCK)
    assert result is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", False)
async def test_clear_non_linux():
    result = await clear_stuck_state(
        "AA:BB:CC:DD:EE:FF", "hci0", StuckState.INACTIVE_CONNECTION
    )
    assert result is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.disconnect_device")
@patch("bleak_connection_manager.diagnostics.remove_device")
async def test_clear_phantom_no_handle(mock_remove, mock_disconnect):
    result = await clear_stuck_state(
        "AA:BB:CC:DD:EE:FF", "hci0", StuckState.PHANTOM_NO_HANDLE
    )
    assert result is True
    mock_disconnect.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")
    mock_remove.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.disconnect_device")
@patch("bleak_connection_manager.diagnostics.remove_device")
async def test_clear_inactive_connection(mock_remove, mock_disconnect):
    result = await clear_stuck_state(
        "AA:BB:CC:DD:EE:FF", "hci0", StuckState.INACTIVE_CONNECTION
    )
    assert result is True
    mock_disconnect.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")
    mock_remove.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.disconnect_by_address")
@patch("bleak_connection_manager.diagnostics.remove_device")
async def test_clear_orphan_hci_handle(mock_remove, mock_hci_disconnect):
    result = await clear_stuck_state(
        "AA:BB:CC:DD:EE:FF", "hci0", StuckState.ORPHAN_HCI_HANDLE
    )
    assert result is True
    mock_hci_disconnect.assert_called_once_with(
        "AA:BB:CC:DD:EE:FF", adapters=["hci0"]
    )
    mock_remove.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.disconnect_by_address")
@patch("bleak_connection_manager.diagnostics.remove_device")
async def test_clear_orphan_hci_handle_multiple_adapters(mock_remove, mock_hci_disconnect):
    result = await clear_stuck_state(
        "AA:BB:CC:DD:EE:FF", "hci0", StuckState.ORPHAN_HCI_HANDLE,
        adapters=["hci0", "hci1"],
    )
    assert result is True
    mock_hci_disconnect.assert_called_once_with(
        "AA:BB:CC:DD:EE:FF", adapters=["hci0", "hci1"]
    )


@pytest.mark.asyncio
@patch("bleak_connection_manager.diagnostics.IS_LINUX", True)
@patch("bleak_connection_manager.diagnostics.remove_device")
async def test_clear_stale_cache(mock_remove):
    result = await clear_stuck_state(
        "AA:BB:CC:DD:EE:FF", "hci0", StuckState.STALE_CACHE
    )
    assert result is True
    mock_remove.assert_called_once_with("AA:BB:CC:DD:EE:FF", "hci0")
