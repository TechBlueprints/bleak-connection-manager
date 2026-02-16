"""Tests for bluez module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bleak_connection_manager.bluez import (
    address_to_bluez_path,
    is_inactive_connection,
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
