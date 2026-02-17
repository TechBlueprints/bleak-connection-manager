"""Raw HCI socket utilities for connection-level inspection and control.

Provides direct access to the Linux HCI layer — the ground truth for
BLE connection state.  This fills a critical gap: D-Bus (BlueZ layer 2)
can disagree with HCI (layer 1), and detecting that mismatch is the
only way to find phantom connections and stale handles.

The socket, ioctl, and ctypes patterns in this module are derived from
``bluetooth-adapters`` by J. Nick Koston, specifically
``bluetooth_adapters.systems.linux_hci``.  That code is licensed under
the Apache License 2.0 (see below).  The ``bdaddr_t`` struct and its
``__str__`` method, the ``AF_BLUETOOTH`` / ``BTPROTO_HCI`` socket
constants, and the ``socket → bind → ioctl`` flow follow the same
approach.

This module extends the pattern to ``HCIGETCONNLIST`` (connection
listing) and raw HCI command injection (disconnect), which are not
present in ``bluetooth-adapters``.

No external shell tools (``hcitool``, ``bluetoothctl``) are needed.

**Requires Linux and CAP_NET_ADMIN capability** (typically root).
All functions degrade gracefully on non-Linux or when permissions
are insufficient — they return empty results rather than raising.

Key functions:

- :func:`get_connections` — list active HCI connections on an adapter
  (equivalent to ``hcitool con``).
- :func:`disconnect_handle` — disconnect a specific HCI handle
  (equivalent to ``hcitool ledc <handle>``).
- :func:`cancel_le_connect` — cancel a pending LE Create Connection
  (equivalent to ``hcitool cmd 0x08 0x000E``).
- :func:`find_connection_by_address` — find a connection by BLE MAC.

Portions derived from bluetooth-adapters:

    Copyright 2022 J. Nick Koston

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    implied.  See the License for the specific language governing
    permissions and limitations under the License.

    Source: https://github.com/Bluetooth-Devices/bluetooth-adapters
"""

from __future__ import annotations

import ctypes
import logging
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .const import IS_LINUX

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

_LOGGER = logging.getLogger(__name__)

# ── Socket constants ────────────────────────────────────────────────

AF_BLUETOOTH = 31
BTPROTO_HCI = 1

# ── HCI ioctl constants ────────────────────────────────────────────
# From linux/include/net/bluetooth/hci.h
# Pattern: _IOR('H', nr, int) = 0x80000000 | (4 << 16) | (0x48 << 8) | nr

HCIGETCONNLIST = 0x800448D4  # _IOR('H', 212, int)

# ── HCI link types ─────────────────────────────────────────────────

SCO_LINK = 0x00
ACL_LINK = 0x01
ESCO_LINK = 0x02
LE_LINK = 0x80

_LINK_TYPE_NAMES = {
    SCO_LINK: "SCO",
    ACL_LINK: "ACL",
    ESCO_LINK: "eSCO",
    LE_LINK: "LE",
}

# ── HCI command constants ──────────────────────────────────────────

HCI_COMMAND_PKT = 0x01
OGF_LINK_CTL = 0x01
OCF_DISCONNECT = 0x0006
HCI_OE_USER_ENDED = 0x13  # "Remote User Terminated Connection"

OGF_LE_CTL = 0x08
OCF_LE_CREATE_CONN_CANCEL = 0x000E

# ── HCI event constants ────────────────────────────────────────────

HCI_EVENT_PKT = 0x04
EVT_DISCONN_COMPLETE = 0x05

# setsockopt level and option for HCI event filter
SOL_HCI = 0
HCI_FILTER = 2

# ── HCI disconnect reason codes ───────────────────────────────────
# From Bluetooth Core Spec Vol 1 Part F (Error Codes).
# Only the codes commonly seen in practice are listed; the rest
# fall through to "Unknown (0xNN)".

HCI_DISCONNECT_REASONS: dict[int, str] = {
    0x05: "Authentication Failure",
    0x06: "PIN or Key Missing",
    0x07: "Memory Capacity Exceeded",
    0x08: "Connection Timeout",
    0x09: "Connection Limit Exceeded",
    0x0A: "Synchronous Connection Limit Exceeded",
    0x0C: "Command Disallowed",
    0x0D: "Rejected Limited Resources",
    0x0E: "Rejected Security Reasons",
    0x0F: "Rejected Unacceptable BD_ADDR",
    0x10: "Connection Accept Timeout Exceeded",
    0x13: "Remote User Terminated Connection",
    0x14: "Remote Device Terminated Low Resources",
    0x15: "Remote Device Terminated Power Off",
    0x16: "Connection Terminated by Local Host",
    0x1A: "Unsupported Remote Feature",
    0x1F: "Unspecified Error",
    0x22: "LMP/LL Response Timeout",
    0x28: "Instant Passed",
    0x2A: "Different Transaction Collision",
    0x3B: "Unacceptable Connection Parameters",
    0x3E: "Connection Failed to be Established",
}


def disconnect_reason_str(reason: int) -> str:
    """Return a human-readable string for an HCI disconnect reason code."""
    name = HCI_DISCONNECT_REASONS.get(reason)
    if name:
        return f"{name} (0x{reason:02X})"
    return f"Unknown (0x{reason:02X})"


# Maximum connections to query per adapter
MAX_CONN = 20


# ── ctypes structs (mirror kernel hci.h) ───────────────────────────

class _bdaddr_t(ctypes.Structure):
    """Bluetooth device address (6 bytes, little-endian)."""

    _fields_ = [("b", ctypes.c_uint8 * 6)]

    def __str__(self) -> str:
        return ":".join(f"{x:02X}" for x in reversed(self.b))


class _hci_conn_info(ctypes.Structure):
    """Per-connection info returned by HCIGETCONNLIST."""

    _fields_ = [
        ("handle", ctypes.c_uint16),
        ("bdaddr", _bdaddr_t),
        ("type", ctypes.c_uint8),
        ("out", ctypes.c_uint8),
        ("state", ctypes.c_uint16),
        ("link_mode", ctypes.c_uint32),
    ]


class _hci_conn_list_req(ctypes.Structure):
    """Request buffer for HCIGETCONNLIST ioctl."""

    _fields_ = [
        ("dev_id", ctypes.c_uint16),
        ("conn_num", ctypes.c_uint16),
        ("conn_info", _hci_conn_info * MAX_CONN),
    ]


# ── Public data model ──────────────────────────────────────────────


@dataclass(frozen=True)
class HciConnection:
    """A single active HCI connection on an adapter.

    Attributes
    ----------
    handle:
        The HCI connection handle (1-4095).
    address:
        The remote device's Bluetooth address (e.g. ``"AA:BB:CC:DD:EE:FF"``).
    link_type:
        The link type (``LE_LINK=0x80``, ``ACL_LINK=0x01``, etc.).
    link_type_name:
        Human-readable link type (``"LE"``, ``"ACL"``, etc.).
    outgoing:
        ``True`` if this host initiated the connection (central role).
    state:
        HCI connection state code.
    link_mode:
        HCI link mode flags.
    adapter:
        The adapter name (e.g. ``"hci0"``).
    """

    handle: int
    address: str
    link_type: int
    link_type_name: str
    outgoing: bool
    state: int
    link_mode: int
    adapter: str


# ── Core functions ─────────────────────────────────────────────────


def get_connections(adapter: str = "hci0") -> list[HciConnection]:
    """List active HCI connections on an adapter.

    This is the Python equivalent of ``hcitool -i <adapter> con``.
    It queries the kernel's HCI layer directly via ioctl, bypassing
    BlueZ D-Bus entirely.

    Parameters
    ----------
    adapter:
        The adapter name (e.g. ``"hci0"``).

    Returns a list of :class:`HciConnection` objects, or an empty list
    if the query fails (non-Linux, insufficient permissions, adapter
    not found, etc.).
    """
    if not IS_LINUX or not _HAS_FCNTL:
        return []

    dev_id = _adapter_to_dev_id(adapter)
    sock: socket.socket | None = None

    try:
        sock = socket.socket(AF_BLUETOOTH, socket.SOCK_RAW, BTPROTO_HCI)
        sock.bind((dev_id,))

        buf = _hci_conn_list_req()
        buf.dev_id = dev_id
        buf.conn_num = MAX_CONN

        fcntl.ioctl(sock.fileno(), HCIGETCONNLIST, buf)

        connections: list[HciConnection] = []
        for i in range(buf.conn_num):
            info = buf.conn_info[i]
            address = str(info.bdaddr)
            link_type = info.type
            connections.append(
                HciConnection(
                    handle=info.handle,
                    address=address,
                    link_type=link_type,
                    link_type_name=_LINK_TYPE_NAMES.get(link_type, f"0x{link_type:02x}"),
                    outgoing=bool(info.out),
                    state=info.state,
                    link_mode=info.link_mode,
                    adapter=adapter,
                )
            )

        return connections

    except PermissionError:
        _LOGGER.debug(
            "Insufficient permissions to query HCI connections on %s "
            "(needs CAP_NET_ADMIN / root)",
            adapter,
        )
        return []
    except OSError as ex:
        _LOGGER.debug(
            "Failed to query HCI connections on %s: %s", adapter, ex
        )
        return []
    finally:
        if sock is not None:
            sock.close()


def find_connection_by_address(
    address: str,
    adapter: str | None = None,
    adapters: list[str] | None = None,
) -> HciConnection | None:
    """Find an active HCI connection by BLE MAC address.

    Searches the specified adapter(s) for a connection matching the
    given address.  If ``adapter`` is provided, only that adapter is
    checked.  If ``adapters`` is provided, all are checked.  If neither,
    only ``hci0`` is checked.

    Parameters
    ----------
    address:
        The BLE MAC address to search for (case-insensitive).
    adapter:
        A single adapter to check.
    adapters:
        Multiple adapters to check.

    Returns the matching :class:`HciConnection`, or ``None`` if not found.
    """
    target = address.upper()

    if adapter is not None:
        search_adapters = [adapter]
    elif adapters is not None:
        search_adapters = adapters
    else:
        search_adapters = ["hci0"]

    for adap in search_adapters:
        for conn in get_connections(adap):
            if conn.address.upper() == target:
                return conn

    return None


def disconnect_handle(
    adapter: str,
    handle: int,
    reason: int = HCI_OE_USER_ENDED,
) -> bool:
    """Disconnect a specific HCI connection handle.

    This is the Python equivalent of ``hcitool -i <adapter> ledc <handle>``.
    It sends an HCI Disconnect command directly to the controller,
    bypassing BlueZ D-Bus.

    This is needed for Stuck State 20 (Peripheral Holdoff) where the
    peripheral won't advertise because a stale HCI handle keeps it in
    connected mode.  D-Bus ``RemoveDevice`` does NOT clear HCI handles.

    Parameters
    ----------
    adapter:
        The adapter name (e.g. ``"hci0"``).
    handle:
        The HCI connection handle to disconnect.
    reason:
        The HCI disconnect reason code.  Default is ``0x13``
        (Remote User Terminated Connection).

    Returns ``True`` if the command was sent successfully.
    """
    if not IS_LINUX or not _HAS_FCNTL:
        return False

    dev_id = _adapter_to_dev_id(adapter)
    sock: socket.socket | None = None

    try:
        sock = socket.socket(AF_BLUETOOTH, socket.SOCK_RAW, BTPROTO_HCI)
        sock.bind((dev_id,))

        # Build HCI command packet:
        # [pkt_type=0x01][opcode_le=0x0406][param_len=3][handle_le][reason]
        opcode = (OGF_LINK_CTL << 10) | OCF_DISCONNECT
        param = struct.pack("<HB", handle & 0x0FFF, reason)
        cmd = struct.pack("<BHB", HCI_COMMAND_PKT, opcode, len(param)) + param

        sock.send(cmd)

        _LOGGER.info(
            "Sent HCI disconnect for handle %d on %s (reason=0x%02x)",
            handle,
            adapter,
            reason,
        )
        return True

    except PermissionError:
        _LOGGER.debug(
            "Insufficient permissions to disconnect HCI handle %d on %s",
            handle,
            adapter,
        )
        return False
    except OSError as ex:
        _LOGGER.debug(
            "Failed to disconnect HCI handle %d on %s: %s",
            handle,
            adapter,
            ex,
        )
        return False
    finally:
        if sock is not None:
            sock.close()


def cancel_le_connect(adapter: str = "hci0") -> bool:
    """Cancel a pending LE Create Connection on an adapter.

    This is the Python equivalent of ``hcitool -i <adapter> cmd 0x08 0x000E``.
    It sends the HCI ``LE_Create_Connection_Cancel`` command to abort a
    stuck connection attempt in the controller.

    This is needed for Stuck State 3 (Pending LE Create Connection)
    where ``org.bluez.Error.InProgress`` persists because the controller
    has a pending ``LE_Create_Connection`` that will never complete.
    D-Bus ``RemoveDevice`` clears BlueZ's ``dev->connect`` pointer but
    may not always cancel the HCI-level command.

    Safe to call even if no connection is pending — the controller
    returns ``Command Disallowed`` which we silently ignore.

    Parameters
    ----------
    adapter:
        The adapter name (e.g. ``"hci0"``).

    Returns ``True`` if the command was sent successfully.
    """
    if not IS_LINUX or not _HAS_FCNTL:
        return False

    dev_id = _adapter_to_dev_id(adapter)
    sock: socket.socket | None = None

    try:
        sock = socket.socket(AF_BLUETOOTH, socket.SOCK_RAW, BTPROTO_HCI)
        sock.bind((dev_id,))

        # HCI LE Create Connection Cancel has no parameters.
        # Opcode = (OGF_LE_CTL << 10) | OCF_LE_CREATE_CONN_CANCEL
        opcode = (OGF_LE_CTL << 10) | OCF_LE_CREATE_CONN_CANCEL
        cmd = struct.pack("<BHB", HCI_COMMAND_PKT, opcode, 0)

        sock.send(cmd)

        _LOGGER.info(
            "Sent LE Create Connection Cancel on %s",
            adapter,
        )
        return True

    except PermissionError:
        _LOGGER.debug(
            "Insufficient permissions to cancel LE connect on %s",
            adapter,
        )
        return False
    except OSError as ex:
        _LOGGER.debug(
            "Failed to cancel LE connect on %s: %s",
            adapter,
            ex,
        )
        return False
    finally:
        if sock is not None:
            sock.close()


def disconnect_by_address(
    address: str,
    adapter: str | None = None,
    adapters: list[str] | None = None,
) -> bool:
    """Find and disconnect an HCI connection by BLE MAC address.

    Combines :func:`find_connection_by_address` and
    :func:`disconnect_handle` into a single call.

    Returns ``True`` if a connection was found and the disconnect
    command was sent, ``False`` otherwise.
    """
    conn = find_connection_by_address(address, adapter=adapter, adapters=adapters)
    if conn is None:
        return False

    _LOGGER.info(
        "Found stale HCI connection for %s on %s (handle=%d, type=%s), "
        "sending disconnect",
        conn.address,
        conn.adapter,
        conn.handle,
        conn.link_type_name,
    )
    return disconnect_handle(conn.adapter, conn.handle)


# ── HCI Disconnect Monitor ─────────────────────────────────────────


@dataclass(frozen=True)
class HciDisconnectEvent:
    """Parsed HCI Disconnection Complete event.

    Attributes
    ----------
    handle:
        The HCI connection handle that was disconnected.
    reason:
        The HCI reason code (e.g. ``0x08`` for Connection Timeout).
    reason_str:
        Human-readable reason string.
    address:
        The BLE MAC address of the disconnected device, or ``None``
        if the handle could not be mapped to an address.
    adapter:
        The adapter the event was received on.
    timestamp:
        Monotonic timestamp when the event was received.
    """

    handle: int
    reason: int
    reason_str: str
    address: str | None
    adapter: str
    timestamp: float


class HciDisconnectMonitor:
    """Monitor HCI Disconnection Complete events on a raw socket.

    Listens for ``EVT_DISCONN_COMPLETE`` (event code ``0x05``) on the
    specified adapter and logs the disconnect reason code for each
    event.  Optionally invokes a callback with the parsed event.

    The monitor takes a snapshot of active HCI connections periodically
    to map handles back to device addresses.  This is best-effort —
    if the handle is not in the snapshot, the address will be ``None``.

    Usage::

        monitor = HciDisconnectMonitor("hci0")
        monitor.start()
        # ... later ...
        monitor.stop()

    Or as a context manager::

        with HciDisconnectMonitor("hci0") as monitor:
            # monitor is running
            ...

    Parameters
    ----------
    adapter:
        The adapter to monitor (e.g. ``"hci0"``).
    on_disconnect:
        Optional callback invoked for each disconnect event.
        Called from the monitor thread — must be thread-safe.
    snapshot_interval:
        How often (seconds) to refresh the handle→address mapping.
        Default is 5 seconds.
    """

    def __init__(
        self,
        adapter: str = "hci0",
        on_disconnect: Callable[[HciDisconnectEvent], None] | None = None,
        snapshot_interval: float = 5.0,
    ) -> None:
        self._adapter = adapter
        self._on_disconnect = on_disconnect
        self._snapshot_interval = snapshot_interval
        self._running = False
        self._thread: threading.Thread | None = None
        # handle → address mapping, refreshed periodically
        self._handle_map: dict[int, str] = {}
        self._last_snapshot: float = 0.0

    @property
    def adapter(self) -> str:
        return self._adapter

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start monitoring in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            name=f"HciDisconnMon_{self._adapter}",
            target=self._run,
            daemon=True,
        )
        self._thread.start()
        _LOGGER.info(
            "HciDisconnectMonitor: started on %s", self._adapter
        )

    def stop(self) -> None:
        """Stop monitoring.

        Safe to call multiple times or before ``start()``.
        """
        self._running = False
        # The thread blocks on socket recv with a timeout, so it will
        # notice _running=False within ~1 second.
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> HciDisconnectMonitor:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _refresh_handle_map(self) -> None:
        """Refresh the handle→address mapping from active connections."""
        now = time.monotonic()
        if now - self._last_snapshot < self._snapshot_interval:
            return
        self._last_snapshot = now
        try:
            connections = get_connections(self._adapter)
            self._handle_map = {
                conn.handle: conn.address for conn in connections
            }
        except Exception:
            pass

    def _run(self) -> None:
        """Background thread: open HCI socket, filter for events, read."""
        if not IS_LINUX or not _HAS_FCNTL:
            _LOGGER.debug(
                "HciDisconnectMonitor: not available on this platform"
            )
            return

        dev_id = _adapter_to_dev_id(self._adapter)
        sock: socket.socket | None = None

        try:
            sock = socket.socket(AF_BLUETOOTH, socket.SOCK_RAW, BTPROTO_HCI)
            sock.bind((dev_id,))

            # Set up HCI event filter to only receive event packets
            # and only the EVT_DISCONN_COMPLETE event.
            #
            # struct hci_filter {
            #   uint32_t type_mask;       // bit 4 = HCI_EVENT_PKT
            #   uint32_t event_mask[2];   // bit 5 = EVT_DISCONN_COMPLETE
            #   uint16_t opcode;          // 0 = don't filter by opcode
            # };
            type_mask = 1 << HCI_EVENT_PKT  # only event packets
            event_mask_lo = 1 << EVT_DISCONN_COMPLETE  # only disconnect events
            event_mask_hi = 0
            opcode = 0
            hci_filter = struct.pack(
                "<III H",
                type_mask,
                event_mask_lo,
                event_mask_hi,
                opcode,
            )
            sock.setsockopt(SOL_HCI, HCI_FILTER, hci_filter)

            # Set a read timeout so we can check _running periodically
            sock.settimeout(1.0)

            _LOGGER.debug(
                "HciDisconnectMonitor: listening on %s", self._adapter
            )

            while self._running:
                self._refresh_handle_map()

                try:
                    data = sock.recv(64)
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        _LOGGER.debug(
                            "HciDisconnectMonitor: socket error on %s",
                            self._adapter,
                            exc_info=True,
                        )
                    break

                if len(data) < 3:
                    continue

                # HCI event packet: [pkt_type=0x04][evt_code][plen][params...]
                pkt_type = data[0]
                evt_code = data[1]
                plen = data[2]

                if pkt_type != HCI_EVENT_PKT or evt_code != EVT_DISCONN_COMPLETE:
                    continue

                if plen < 4 or len(data) < 7:
                    continue

                # EVT_DISCONN_COMPLETE params: [status:1][handle:2LE][reason:1]
                status = data[3]
                handle = struct.unpack_from("<H", data, 4)[0] & 0x0FFF
                reason = data[6]

                # Map handle to address (best-effort)
                address = self._handle_map.pop(handle, None)

                event = HciDisconnectEvent(
                    handle=handle,
                    reason=reason,
                    reason_str=disconnect_reason_str(reason),
                    address=address,
                    adapter=self._adapter,
                    timestamp=time.monotonic(),
                )

                if status == 0x00:
                    addr_str = address or f"handle={handle}"
                    _LOGGER.info(
                        "HCI disconnect: %s reason=%s on %s",
                        addr_str,
                        event.reason_str,
                        self._adapter,
                    )
                else:
                    _LOGGER.debug(
                        "HCI disconnect event with non-zero status "
                        "0x%02X for handle %d on %s",
                        status,
                        handle,
                        self._adapter,
                    )

                if self._on_disconnect is not None:
                    try:
                        self._on_disconnect(event)
                    except Exception:
                        _LOGGER.debug(
                            "HciDisconnectMonitor: callback error",
                            exc_info=True,
                        )

        except PermissionError:
            _LOGGER.warning(
                "HciDisconnectMonitor: insufficient permissions on %s "
                "(needs CAP_NET_ADMIN / root)",
                self._adapter,
            )
        except OSError as ex:
            _LOGGER.warning(
                "HciDisconnectMonitor: failed to open socket on %s: %s",
                self._adapter,
                ex,
            )
        finally:
            if sock is not None:
                sock.close()
            self._running = False
            _LOGGER.debug(
                "HciDisconnectMonitor: stopped on %s", self._adapter
            )


# ── Helpers ────────────────────────────────────────────────────────


def _adapter_to_dev_id(adapter: str) -> int:
    """Convert adapter name (e.g. ``"hci0"``) to device id (e.g. ``0``)."""
    return int(adapter.removeprefix("hci"))
