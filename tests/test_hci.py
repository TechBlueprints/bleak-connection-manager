"""Tests for the hci module (hcitool-based implementation).

All tests mock subprocess.run to simulate hcitool output, so they
can run on any platform without root or hcitool installed.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from bleak_connection_manager.hci import (
    HciConnection,
    _parse_hcitool_con,
    cancel_le_connect,
    disconnect_by_address,
    disconnect_handle,
    disconnect_reason_str,
    find_connection_by_address,
    get_connections,
    hci_available,
)


# ── _parse_hcitool_con ────────────────────────────────────────────


def test_parse_empty_output():
    assert _parse_hcitool_con("Connections:\n", "hci0") == []


def test_parse_one_le_connection():
    output = (
        "Connections:\n"
        "\t< LE AA:BB:CC:DD:EE:FF handle 64 state 1 lm MASTER\n"
    )
    result = _parse_hcitool_con(output, "hci0")
    assert len(result) == 1
    conn = result[0]
    assert conn.handle == 64
    assert conn.address == "AA:BB:CC:DD:EE:FF"
    assert conn.link_type == 0x80
    assert conn.link_type_name == "LE"
    assert conn.outgoing is True
    assert conn.state == 1
    assert conn.adapter == "hci0"


def test_parse_incoming_acl():
    output = (
        "Connections:\n"
        "\t> ACL 11:22:33:44:55:66 handle 42 state 1 lm SLAVE\n"
    )
    result = _parse_hcitool_con(output, "hci1")
    assert len(result) == 1
    conn = result[0]
    assert conn.handle == 42
    assert conn.address == "11:22:33:44:55:66"
    assert conn.link_type == 0x01
    assert conn.link_type_name == "ACL"
    assert conn.outgoing is False
    assert conn.adapter == "hci1"


def test_parse_multiple_connections():
    output = (
        "Connections:\n"
        "\t< LE AA:BB:CC:DD:EE:01 handle 64 state 1 lm MASTER\n"
        "\t> ACL AA:BB:CC:DD:EE:02 handle 65 state 1 lm SLAVE\n"
        "\t< LE AA:BB:CC:DD:EE:03 handle 66 state 1 lm MASTER\n"
    )
    result = _parse_hcitool_con(output, "hci0")
    assert len(result) == 3
    assert result[0].handle == 64
    assert result[1].handle == 65
    assert result[2].handle == 66


def test_parse_lowercase_address():
    output = (
        "Connections:\n"
        "\t< LE aa:bb:cc:dd:ee:ff handle 64 state 1 lm MASTER\n"
    )
    result = _parse_hcitool_con(output, "hci0")
    assert len(result) == 1
    assert result[0].address == "AA:BB:CC:DD:EE:FF"


def test_parse_no_lm_field():
    output = (
        "Connections:\n"
        "\t< LE AA:BB:CC:DD:EE:FF handle 64 state 1\n"
    )
    result = _parse_hcitool_con(output, "hci0")
    assert len(result) == 1
    assert result[0].handle == 64


# ── HciConnection dataclass ───────────────────────────────────────


def test_hci_connection_frozen():
    conn = HciConnection(
        handle=64,
        address="AA:BB:CC:DD:EE:FF",
        link_type=0x80,
        link_type_name="LE",
        outgoing=True,
        state=1,
        link_mode=0,
        adapter="hci0",
    )
    assert conn.handle == 64
    assert conn.address == "AA:BB:CC:DD:EE:FF"
    assert conn.link_type_name == "LE"
    with pytest.raises(AttributeError):
        conn.handle = 99  # type: ignore[misc]


# ── disconnect_reason_str ─────────────────────────────────────────


def test_disconnect_reason_known():
    assert "Connection Timeout" in disconnect_reason_str(0x08)
    assert "0x08" in disconnect_reason_str(0x08)


def test_disconnect_reason_unknown():
    assert "Unknown" in disconnect_reason_str(0xFF)
    assert "0xFF" in disconnect_reason_str(0xFF)


# ── hci_available ─────────────────────────────────────────────────


@patch("bleak_connection_manager.hci._HCITOOL", None)
def test_hci_available_no_hcitool():
    assert hci_available("hci0") is False


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_hci_available_success(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    assert hci_available("hci0") is True
    mock_subprocess.run.assert_called_once()


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_hci_available_failure(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(returncode=1)
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    assert hci_available("hci0") is False


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_hci_available_timeout(mock_subprocess):
    mock_subprocess.run.side_effect = subprocess.TimeoutExpired(
        cmd="hcitool", timeout=5.0,
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    assert hci_available("hci0") is False


# ── get_connections ────────────────────────────────────────────────


@patch("bleak_connection_manager.hci._HCITOOL", None)
def test_get_connections_no_hcitool():
    assert get_connections("hci0") == []


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_get_connections_empty(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(
        returncode=0, stdout="Connections:\n",
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = get_connections("hci0")
    assert result == []


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_get_connections_one_le(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(
        returncode=0,
        stdout=(
            "Connections:\n"
            "\t< LE AA:BB:CC:DD:EE:FF handle 64 state 1 lm MASTER\n"
        ),
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

    result = get_connections("hci0")
    assert len(result) == 1
    conn = result[0]
    assert conn.handle == 64
    assert conn.address == "AA:BB:CC:DD:EE:FF"
    assert conn.link_type_name == "LE"


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_get_connections_failure(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(
        returncode=1, stderr="Can't get connection list",
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = get_connections("hci0")
    assert result == []


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_get_connections_timeout(mock_subprocess):
    mock_subprocess.run.side_effect = subprocess.TimeoutExpired(
        cmd="hcitool", timeout=5.0,
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = get_connections("hci0")
    assert result == []


# ── find_connection_by_address ─────────────────────────────────────


@patch("bleak_connection_manager.hci.get_connections")
def test_find_connection_default_adapter(mock_get):
    mock_get.return_value = [
        HciConnection(
            handle=64,
            address="AA:BB:CC:DD:EE:FF",
            link_type=0x80,
            link_type_name="LE",
            outgoing=True,
            state=1,
            link_mode=0,
            adapter="hci0",
        )
    ]

    result = find_connection_by_address("AA:BB:CC:DD:EE:FF")
    assert result is not None
    assert result.handle == 64
    mock_get.assert_called_once_with("hci0")


@patch("bleak_connection_manager.hci.get_connections")
def test_find_connection_case_insensitive(mock_get):
    mock_get.return_value = [
        HciConnection(
            handle=64,
            address="AA:BB:CC:DD:EE:FF",
            link_type=0x80,
            link_type_name="LE",
            outgoing=True,
            state=1,
            link_mode=0,
            adapter="hci0",
        )
    ]

    result = find_connection_by_address("aa:bb:cc:dd:ee:ff")
    assert result is not None
    assert result.handle == 64


@patch("bleak_connection_manager.hci.get_connections")
def test_find_connection_not_found(mock_get):
    mock_get.return_value = [
        HciConnection(
            handle=64,
            address="11:22:33:44:55:66",
            link_type=0x80,
            link_type_name="LE",
            outgoing=True,
            state=1,
            link_mode=0,
            adapter="hci0",
        )
    ]

    result = find_connection_by_address("AA:BB:CC:DD:EE:FF")
    assert result is None


@patch("bleak_connection_manager.hci.get_connections")
def test_find_connection_multiple_adapters(mock_get):
    mock_get.side_effect = [
        [],
        [
            HciConnection(
                handle=99,
                address="AA:BB:CC:DD:EE:FF",
                link_type=0x80,
                link_type_name="LE",
                outgoing=True,
                state=1,
                link_mode=0,
                adapter="hci1",
            )
        ],
    ]

    result = find_connection_by_address(
        "AA:BB:CC:DD:EE:FF", adapters=["hci0", "hci1"]
    )
    assert result is not None
    assert result.handle == 99
    assert result.adapter == "hci1"
    assert mock_get.call_count == 2


@patch("bleak_connection_manager.hci.get_connections")
def test_find_connection_specific_adapter(mock_get):
    mock_get.return_value = []

    result = find_connection_by_address("AA:BB:CC:DD:EE:FF", adapter="hci2")
    assert result is None
    mock_get.assert_called_once_with("hci2")


# ── disconnect_handle ──────────────────────────────────────────────


@patch("bleak_connection_manager.hci._HCITOOL", None)
def test_disconnect_handle_no_hcitool():
    assert disconnect_handle("hci0", 64) is False


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_disconnect_handle_success(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = disconnect_handle("hci0", 64)
    assert result is True
    args = mock_subprocess.run.call_args[0][0]
    assert args == ["/usr/bin/hcitool", "-i", "hci0", "ledc", "64"]


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_disconnect_handle_failure(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(
        returncode=1, stderr="Disconnect failed",
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = disconnect_handle("hci0", 64)
    assert result is False


# ── disconnect_by_address ──────────────────────────────────────────


@patch("bleak_connection_manager.hci.disconnect_handle")
@patch("bleak_connection_manager.hci.find_connection_by_address")
def test_disconnect_by_address_found(mock_find, mock_disc):
    mock_find.return_value = HciConnection(
        handle=64,
        address="AA:BB:CC:DD:EE:FF",
        link_type=0x80,
        link_type_name="LE",
        outgoing=True,
        state=1,
        link_mode=0,
        adapter="hci0",
    )
    mock_disc.return_value = True

    result = disconnect_by_address("AA:BB:CC:DD:EE:FF", adapter="hci0")
    assert result is True
    mock_disc.assert_called_once_with("hci0", 64)


@patch("bleak_connection_manager.hci.find_connection_by_address")
def test_disconnect_by_address_not_found(mock_find):
    mock_find.return_value = None

    result = disconnect_by_address("AA:BB:CC:DD:EE:FF", adapter="hci0")
    assert result is False


# ── cancel_le_connect ──────────────────────────────────────────────


@patch("bleak_connection_manager.hci._HCITOOL", None)
def test_cancel_le_connect_no_hcitool():
    assert cancel_le_connect("hci0") is False


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_cancel_le_connect_success(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = cancel_le_connect("hci0")
    assert result is True
    args = mock_subprocess.run.call_args[0][0]
    assert args == ["/usr/bin/hcitool", "-i", "hci0", "cmd", "0x08", "0x000E"]


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_cancel_le_connect_failure(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(
        returncode=1, stderr="Command failed",
    )
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = cancel_le_connect("hci0")
    assert result is False


@patch("bleak_connection_manager.hci._HCITOOL", "/usr/bin/hcitool")
@patch("bleak_connection_manager.hci.subprocess")
def test_cancel_le_connect_specific_adapter(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(returncode=0)
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    result = cancel_le_connect("hci1")
    assert result is True
    args = mock_subprocess.run.call_args[0][0]
    assert args == ["/usr/bin/hcitool", "-i", "hci1", "cmd", "0x08", "0x000E"]
