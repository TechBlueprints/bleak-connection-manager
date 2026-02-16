"""Tests for the hci module.

Since HCI sockets require Linux + CAP_NET_ADMIN, all tests mock the
socket/ioctl layer.  This verifies the struct parsing, address
formatting, and error handling without needing root privileges.
"""

from __future__ import annotations

import ctypes
import struct
from unittest.mock import MagicMock, call, patch

import pytest

from bleak_connection_manager.hci import (
    LE_LINK,
    ACL_LINK,
    MAX_CONN,
    HCI_COMMAND_PKT,
    HCI_OE_USER_ENDED,
    OGF_LE_CTL,
    OCF_LE_CREATE_CONN_CANCEL,
    cancel_le_connect,
    HciConnection,
    _adapter_to_dev_id,
    _bdaddr_t,
    _hci_conn_info,
    _hci_conn_list_req,
    disconnect_by_address,
    disconnect_handle,
    find_connection_by_address,
    get_connections,
)


# ── Helper to build mock ioctl data ────────────────────────────────


def _make_conn_list_buf(dev_id: int, connections: list[dict]) -> bytes:
    """Build raw bytes matching _hci_conn_list_req with filled conn_info.

    Each connection dict should have:
      handle, address (str like "AA:BB:CC:DD:EE:FF"), type, out, state, link_mode
    """
    buf = _hci_conn_list_req()
    buf.dev_id = dev_id
    buf.conn_num = len(connections)
    for i, conn in enumerate(connections):
        info = buf.conn_info[i]
        info.handle = conn["handle"]
        addr_bytes = bytes(
            int(x, 16) for x in reversed(conn["address"].split(":"))
        )
        ctypes.memmove(info.bdaddr.b, addr_bytes, 6)
        info.type = conn["type"]
        info.out = conn["out"]
        info.state = conn["state"]
        info.link_mode = conn["link_mode"]
    return bytes(buf)


def _mock_ioctl_factory(dev_id: int, connections: list[dict]):
    """Return a side_effect for fcntl.ioctl that fills the buffer."""
    raw = _make_conn_list_buf(dev_id, connections)

    def _ioctl(fd, request, buf):
        ctypes.memmove(ctypes.addressof(buf), raw, len(raw))
        return 0

    return _ioctl


# ── _adapter_to_dev_id ─────────────────────────────────────────────


def test_adapter_to_dev_id():
    assert _adapter_to_dev_id("hci0") == 0
    assert _adapter_to_dev_id("hci1") == 1
    assert _adapter_to_dev_id("hci12") == 12


# ── _bdaddr_t.__str__ ─────────────────────────────────────────────


def test_bdaddr_str():
    addr = _bdaddr_t()
    # BlueZ stores addresses in little-endian byte order,
    # so AA:BB:CC:DD:EE:FF is stored as [FF, EE, DD, CC, BB, AA]
    addr.b[0] = 0xFF
    addr.b[1] = 0xEE
    addr.b[2] = 0xDD
    addr.b[3] = 0xCC
    addr.b[4] = 0xBB
    addr.b[5] = 0xAA
    assert str(addr) == "AA:BB:CC:DD:EE:FF"


# ── HciConnection dataclass ───────────────────────────────────────


def test_hci_connection_frozen():
    conn = HciConnection(
        handle=64,
        address="AA:BB:CC:DD:EE:FF",
        link_type=LE_LINK,
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


# ── get_connections ────────────────────────────────────────────────


@patch("bleak_connection_manager.hci.IS_LINUX", False)
def test_get_connections_not_linux():
    assert get_connections("hci0") == []


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", False)
def test_get_connections_no_fcntl():
    assert get_connections("hci0") == []


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
@patch("bleak_connection_manager.hci.fcntl")
def test_get_connections_empty(mock_fcntl, mock_socket_mod):
    """No connections on the adapter."""
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock

    mock_fcntl.ioctl.side_effect = _mock_ioctl_factory(0, [])

    result = get_connections("hci0")
    assert result == []
    mock_sock.bind.assert_called_once_with((0,))
    mock_sock.close.assert_called_once()


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
@patch("bleak_connection_manager.hci.fcntl")
def test_get_connections_one_le(mock_fcntl, mock_socket_mod):
    """One LE connection."""
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock

    mock_fcntl.ioctl.side_effect = _mock_ioctl_factory(0, [
        {
            "handle": 64,
            "address": "AA:BB:CC:DD:EE:FF",
            "type": LE_LINK,
            "out": 1,
            "state": 1,
            "link_mode": 0,
        }
    ])

    result = get_connections("hci0")
    assert len(result) == 1
    conn = result[0]
    assert conn.handle == 64
    assert conn.address == "AA:BB:CC:DD:EE:FF"
    assert conn.link_type == LE_LINK
    assert conn.link_type_name == "LE"
    assert conn.outgoing is True
    assert conn.adapter == "hci0"


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
@patch("bleak_connection_manager.hci.fcntl")
def test_get_connections_multiple(mock_fcntl, mock_socket_mod):
    """Multiple connections on different adapters."""
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock

    mock_fcntl.ioctl.side_effect = _mock_ioctl_factory(1, [
        {
            "handle": 64,
            "address": "AA:BB:CC:DD:EE:01",
            "type": LE_LINK,
            "out": 1,
            "state": 1,
            "link_mode": 0,
        },
        {
            "handle": 65,
            "address": "AA:BB:CC:DD:EE:02",
            "type": ACL_LINK,
            "out": 0,
            "state": 1,
            "link_mode": 0,
        },
    ])

    result = get_connections("hci1")
    assert len(result) == 2
    assert result[0].handle == 64
    assert result[0].link_type_name == "LE"
    assert result[1].handle == 65
    assert result[1].link_type_name == "ACL"
    assert result[1].outgoing is False


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_get_connections_permission_error(mock_socket_mod):
    """PermissionError returns empty list, not exception."""
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock
    mock_sock.bind.side_effect = PermissionError("Operation not permitted")

    result = get_connections("hci0")
    assert result == []
    mock_sock.close.assert_called_once()


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_get_connections_os_error(mock_socket_mod):
    """OSError (e.g. adapter not found) returns empty list."""
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock
    mock_sock.bind.side_effect = OSError("No such device")

    result = get_connections("hci0")
    assert result == []
    mock_sock.close.assert_called_once()


# ── find_connection_by_address ─────────────────────────────────────


@patch("bleak_connection_manager.hci.get_connections")
def test_find_connection_default_adapter(mock_get):
    mock_get.return_value = [
        HciConnection(
            handle=64,
            address="AA:BB:CC:DD:EE:FF",
            link_type=LE_LINK,
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
            link_type=LE_LINK,
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
            link_type=LE_LINK,
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
    """Search across multiple adapters, found on second."""
    mock_get.side_effect = [
        [],  # hci0 has no connections
        [
            HciConnection(
                handle=99,
                address="AA:BB:CC:DD:EE:FF",
                link_type=LE_LINK,
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


@patch("bleak_connection_manager.hci.IS_LINUX", False)
def test_disconnect_handle_not_linux():
    assert disconnect_handle("hci0", 64) is False


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_disconnect_handle_sends_command(mock_socket_mod):
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock

    result = disconnect_handle("hci0", 64, reason=HCI_OE_USER_ENDED)
    assert result is True

    mock_sock.bind.assert_called_once_with((0,))
    mock_sock.send.assert_called_once()

    # Verify the command packet structure
    sent_data = mock_sock.send.call_args[0][0]
    assert sent_data[0] == HCI_COMMAND_PKT
    # Opcode 0x0406 little-endian
    assert sent_data[1] == 0x06
    assert sent_data[2] == 0x04
    # Param length = 3
    assert sent_data[3] == 3
    # Handle = 64 (0x0040) little-endian, masked to 12 bits
    handle_bytes = struct.unpack("<H", sent_data[4:6])[0]
    assert handle_bytes == 64
    # Reason
    assert sent_data[6] == HCI_OE_USER_ENDED

    mock_sock.close.assert_called_once()


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_disconnect_handle_permission_error(mock_socket_mod):
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock
    mock_sock.bind.side_effect = PermissionError("Operation not permitted")

    result = disconnect_handle("hci0", 64)
    assert result is False
    mock_sock.close.assert_called_once()


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_disconnect_handle_os_error(mock_socket_mod):
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock
    mock_sock.send.side_effect = OSError("Connection reset")

    result = disconnect_handle("hci0", 64)
    assert result is False
    mock_sock.close.assert_called_once()


# ── disconnect_by_address ──────────────────────────────────────────


@patch("bleak_connection_manager.hci.disconnect_handle")
@patch("bleak_connection_manager.hci.find_connection_by_address")
def test_disconnect_by_address_found(mock_find, mock_disc):
    mock_find.return_value = HciConnection(
        handle=64,
        address="AA:BB:CC:DD:EE:FF",
        link_type=LE_LINK,
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


@patch("bleak_connection_manager.hci.IS_LINUX", False)
def test_cancel_le_connect_not_linux():
    assert cancel_le_connect("hci0") is False


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_cancel_le_connect_sends_command(mock_socket_mod):
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock

    result = cancel_le_connect("hci0")
    assert result is True

    mock_sock.bind.assert_called_once_with((0,))
    mock_sock.send.assert_called_once()

    # Verify the command packet structure
    sent_data = mock_sock.send.call_args[0][0]
    assert sent_data[0] == HCI_COMMAND_PKT
    # Opcode = (0x08 << 10) | 0x000E = 0x200E, little-endian
    opcode = (OGF_LE_CTL << 10) | OCF_LE_CREATE_CONN_CANCEL
    assert struct.unpack("<H", sent_data[1:3])[0] == opcode
    # Param length = 0 (no parameters)
    assert sent_data[3] == 0
    # Total packet length = 4 bytes (type + opcode + param_len)
    assert len(sent_data) == 4

    mock_sock.close.assert_called_once()


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_cancel_le_connect_permission_error(mock_socket_mod):
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock
    mock_sock.bind.side_effect = PermissionError("Operation not permitted")

    result = cancel_le_connect("hci0")
    assert result is False
    mock_sock.close.assert_called_once()


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_cancel_le_connect_os_error(mock_socket_mod):
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock
    mock_sock.send.side_effect = OSError("Network is down")

    result = cancel_le_connect("hci0")
    assert result is False
    mock_sock.close.assert_called_once()


@patch("bleak_connection_manager.hci.IS_LINUX", True)
@patch("bleak_connection_manager.hci._HAS_FCNTL", True)
@patch("bleak_connection_manager.hci.socket")
def test_cancel_le_connect_specific_adapter(mock_socket_mod):
    mock_sock = MagicMock()
    mock_socket_mod.socket.return_value = mock_sock

    result = cancel_le_connect("hci1")
    assert result is True
    mock_sock.bind.assert_called_once_with((1,))
