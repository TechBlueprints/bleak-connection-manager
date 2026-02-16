"""Tests for adapters module."""

from unittest.mock import MagicMock, patch

from bleak.backends.device import BLEDevice

from bleak_connection_manager.adapters import (
    discover_adapters,
    make_device_for_adapter,
    pick_adapter,
)


def test_pick_adapter_single():
    assert pick_adapter(["hci0"], 1) == "hci0"
    assert pick_adapter(["hci0"], 2) == "hci0"
    assert pick_adapter(["hci0"], 3) == "hci0"


def test_pick_adapter_round_robin():
    adapters = ["hci0", "hci1", "hci2"]
    assert pick_adapter(adapters, 1) == "hci0"
    assert pick_adapter(adapters, 2) == "hci1"
    assert pick_adapter(adapters, 3) == "hci2"
    assert pick_adapter(adapters, 4) == "hci0"  # wraps around


def test_pick_adapter_empty():
    assert pick_adapter([], 1) == "hci0"


def test_make_device_for_adapter():
    device = BLEDevice(
        "AA:BB:CC:DD:EE:FF",
        "TestDevice",
        {"path": "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"},
    )
    new_device = make_device_for_adapter(device, "hci1")

    assert new_device.address == "AA:BB:CC:DD:EE:FF"
    assert new_device.name == "TestDevice"
    assert "hci1" in new_device.details["path"]


def test_make_device_for_adapter_no_details():
    device = BLEDevice("AA:BB:CC:DD:EE:FF", "TestDevice", {})
    new_device = make_device_for_adapter(device, "hci0")
    assert new_device.details["path"] == "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"


@patch("bleak_connection_manager.adapters.IS_LINUX", False)
def test_discover_adapters_non_linux():
    assert discover_adapters() == ["hci0"]
