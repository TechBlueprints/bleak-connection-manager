"""BlueZ D-Bus utilities for connection state inspection.

Provides functions to detect phantom/inactive BLE connections and
construct BlueZ D-Bus object paths.  Uses the shared ``MessageBus``
from :mod:`.dbus_bus` and raw ``bus.call()`` for all queries — no
proxy objects, no introspection, no fire-and-forget ``AddMatch``.

These functions fill a gap: ``bleak-retry-connector`` does not expose
connection state inspection, and ``bleak`` itself does not provide
phantom detection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .const import IS_LINUX

_LOGGER = logging.getLogger(__name__)

# D-Bus constants
_BLUEZ_SERVICE = "org.bluez"
_DEVICE_INTERFACE = "org.bluez.Device1"
_ADAPTER_INTERFACE = "org.bluez.Adapter1"
_PROPERTIES_INTERFACE = "org.freedesktop.DBus.Properties"
_OBJECT_MANAGER_INTERFACE = "org.freedesktop.DBus.ObjectManager"


def address_to_bluez_path(address: str, adapter: str = "hci0") -> str:
    """Convert a BLE address + adapter to a BlueZ D-Bus object path.

    Example::

        >>> address_to_bluez_path("AA:BB:CC:DD:EE:FF", "hci0")
        '/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF'
    """
    dev_part = f"dev_{address.upper().replace(':', '_')}"
    return f"/org/bluez/{adapter}/{dev_part}"


async def _get_device_properties(
    address: str, adapter: str = "hci0"
) -> dict[str, Any] | None:
    """Fetch D-Bus properties for a BLE device.

    Returns a dict of property name → value, or ``None`` if the device
    is not found on D-Bus.
    """
    if not IS_LINUX:
        return None

    try:
        from dbus_fast import Message, MessageType

        from .dbus_bus import get_bus
    except ImportError:
        _LOGGER.debug("dbus-fast not available, cannot query D-Bus")
        return None

    path = address_to_bluez_path(address, adapter)

    try:
        bus = await get_bus()
        reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=path,
                interface=_PROPERTIES_INTERFACE,
                member="GetAll",
                signature="s",
                body=[_DEVICE_INTERFACE],
            )
        )
        if reply.message_type == MessageType.ERROR:
            return None
        return {k: v.value for k, v in reply.body[0].items()}
    except Exception:
        _LOGGER.debug(
            "Failed to get D-Bus properties for %s on %s",
            address,
            adapter,
            exc_info=True,
        )
        return None


async def is_inactive_connection(
    address: str, adapter: str = "hci0"
) -> bool:
    """Check whether a device has an inactive BLE connection.

    An inactive connection is one where BlueZ D-Bus reports
    ``Connected=True`` but ``ServicesResolved`` is not ``True``.
    This indicates GATT discovery never completed — typically because
    the HCI transport died after the initial connection event.

    Parameters
    ----------
    address:
        The BLE device MAC address (e.g. ``"AA:BB:CC:DD:EE:FF"``).
    adapter:
        The adapter name (e.g. ``"hci0"``).

    Returns ``True`` if the connection is inactive (phantom).
    """
    if not IS_LINUX:
        return False

    props = await _get_device_properties(address, adapter)
    if props is None:
        return False

    connected = props.get("Connected", False)
    if not connected:
        return False

    services_resolved = props.get("ServicesResolved", False)
    if services_resolved is not True:
        _LOGGER.debug(
            "%s: Inactive connection detected — Connected but "
            "ServicesResolved is not True",
            address,
        )
        return True

    return False


async def remove_device(address: str, adapter: str = "hci0") -> bool:
    """Remove a device from BlueZ via the adapter's RemoveDevice method.

    This is the D-Bus equivalent of ``bluetoothctl remove <address>``.
    Clears all cached state including GATT services.

    Returns ``True`` if the device was removed successfully.
    """
    if not IS_LINUX:
        return False

    try:
        from dbus_fast import Message, MessageType

        from .dbus_bus import get_bus
    except ImportError:
        return False

    adapter_path = f"/org/bluez/{adapter}"
    device_path = address_to_bluez_path(address, adapter)

    try:
        bus = await get_bus()
        reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=adapter_path,
                interface=_ADAPTER_INTERFACE,
                member="RemoveDevice",
                signature="o",
                body=[device_path],
            )
        )
        if reply.message_type == MessageType.ERROR:
            _LOGGER.debug(
                "RemoveDevice D-Bus error for %s: %s",
                address,
                reply.body[0] if reply.body else "unknown",
            )
            return False
        _LOGGER.debug("Removed device %s from %s", address, adapter)
        return True
    except Exception:
        _LOGGER.debug(
            "Failed to remove %s from %s",
            address,
            adapter,
            exc_info=True,
        )
        return False


async def disconnect_device(address: str, adapter: str = "hci0") -> bool:
    """Disconnect a device via D-Bus.

    Returns ``True`` if the disconnect was initiated successfully.
    """
    if not IS_LINUX:
        return False

    try:
        from dbus_fast import Message, MessageType

        from .dbus_bus import get_bus
    except ImportError:
        return False

    path = address_to_bluez_path(address, adapter)

    try:
        bus = await get_bus()
        reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=path,
                interface=_DEVICE_INTERFACE,
                member="Disconnect",
            )
        )
        if reply.message_type == MessageType.ERROR:
            _LOGGER.debug(
                "Disconnect D-Bus error for %s: %s",
                address,
                reply.body[0] if reply.body else "unknown",
            )
            return False
        _LOGGER.debug("Disconnected %s on %s via D-Bus", address, adapter)
        return True
    except Exception:
        _LOGGER.debug(
            "Failed to disconnect %s on %s",
            address,
            adapter,
            exc_info=True,
        )
        return False


async def verified_disconnect(
    address: str,
    adapter: str = "hci0",
    timeout: float = 5.0,
    poll_interval: float = 0.5,
) -> bool:
    """Disconnect a device and verify the D-Bus ``Connected`` property is ``False``.

    Unlike :func:`disconnect_device`, this function polls the D-Bus
    ``Connected`` property after initiating the disconnect to confirm
    the device is truly disconnected.  If ``Connected`` remains ``True``
    after *timeout* seconds, it escalates to :func:`remove_device` to
    force-clear the stale BlueZ state.

    This addresses Stuck State 8 (disconnect hang) without using
    ``hcitool`` — purely via D-Bus.

    Parameters
    ----------
    address:
        The BLE device MAC address.
    adapter:
        The adapter name (e.g. ``"hci0"``).
    timeout:
        Maximum seconds to wait for ``Connected`` to become ``False``.
    poll_interval:
        Seconds between D-Bus property polls.

    Returns ``True`` if the device is confirmed disconnected, ``False``
    if the disconnect could not be verified (but cleanup was attempted).
    """
    if not IS_LINUX:
        return True

    # Step 1: Initiate disconnect via D-Bus
    await disconnect_device(address, adapter)

    # Step 2: Poll D-Bus Connected property
    elapsed = 0.0
    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        props = await _get_device_properties(address, adapter)
        if props is None:
            # Device no longer on D-Bus — fully disconnected
            return True
        if not props.get("Connected", False):
            _LOGGER.debug(
                "%s: Verified disconnected on %s after %.1f s",
                address,
                adapter,
                elapsed,
            )
            return True

    # Step 3: Still connected after timeout — escalate to remove
    _LOGGER.warning(
        "%s: Still Connected=True on %s after %.1f s disconnect timeout, "
        "escalating to remove_device",
        address,
        adapter,
        timeout,
    )
    await remove_device(address, adapter)
    return False


# ── Adapter health check for scan readiness ────────────────────────
#
# Pre-scan detection and repair of stale BlueZ adapter state that
# causes ``InProgress`` errors on ``StartDiscovery``.

# Per-adapter cooldown tracking for power-cycle operations.
# Prevents rapid cascading power-cycles when multiple processes
# detect stale state within a short window.
_POWER_CYCLE_COOLDOWN = 30.0
_last_power_cycle: dict[str, float] = {}

# Settle time after power-cycling an adapter.  The adapter goes
# through power-off → power-on → BlueZ re-initialization.
_POWER_CYCLE_SETTLE = 2.0


async def get_adapter_discovering(adapter: str) -> bool | None:
    """Query the ``Discovering`` property of a BlueZ adapter.

    Returns ``True`` if the adapter is actively discovering,
    ``False`` if not, or ``None`` if the query failed.
    """
    if not IS_LINUX:
        return None

    try:
        from dbus_fast import Message, MessageType

        from .dbus_bus import get_bus
    except ImportError:
        return None

    adapter_path = f"/org/bluez/{adapter}"

    try:
        bus = await get_bus()
        reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=adapter_path,
                interface=_PROPERTIES_INTERFACE,
                member="Get",
                signature="ss",
                body=[_ADAPTER_INTERFACE, "Discovering"],
            )
        )
        if reply.message_type == MessageType.ERROR:
            return None
        val = reply.body[0]
        if hasattr(val, "value"):
            val = val.value
        return bool(val)
    except Exception:
        _LOGGER.debug(
            "Failed to query Discovering on %s", adapter, exc_info=True,
        )
        return None


async def power_cycle_adapter(adapter: str) -> bool:
    """Power-cycle a BlueZ adapter via D-Bus ``Powered`` property.

    Sets ``Powered=False``, waits briefly, then ``Powered=True``.
    This drops ALL connections and discovery sessions on the adapter,
    clearing any stale BlueZ state.

    Less disruptive than ``bluetooth-auto-recovery`` (no USB reset
    or rfkill).  Suitable for clearing orphaned discovery sessions
    without needing external tools.

    Returns ``True`` if the power-cycle completed, ``False`` on error.
    """
    if not IS_LINUX:
        return False

    try:
        from dbus_fast import Message, MessageType, Variant

        from .dbus_bus import get_bus
    except ImportError:
        return False

    adapter_path = f"/org/bluez/{adapter}"

    try:
        bus = await get_bus()

        # Powered = False
        reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=adapter_path,
                interface=_PROPERTIES_INTERFACE,
                member="Set",
                signature="ssv",
                body=[_ADAPTER_INTERFACE, "Powered", Variant("b", False)],
            )
        )
        if reply.message_type == MessageType.ERROR:
            _LOGGER.warning(
                "%s: Failed to set Powered=False: %s",
                adapter,
                reply.body[0] if reply.body else "unknown",
            )
            return False

        await asyncio.sleep(0.5)

        # Powered = True
        reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=adapter_path,
                interface=_PROPERTIES_INTERFACE,
                member="Set",
                signature="ssv",
                body=[_ADAPTER_INTERFACE, "Powered", Variant("b", True)],
            )
        )
        if reply.message_type == MessageType.ERROR:
            _LOGGER.warning(
                "%s: Failed to set Powered=True: %s",
                adapter,
                reply.body[0] if reply.body else "unknown",
            )
            return False

        _LOGGER.info("%s: Power-cycled adapter via D-Bus", adapter)
        await asyncio.sleep(_POWER_CYCLE_SETTLE)
        return True

    except Exception:
        _LOGGER.warning(
            "Failed to power-cycle %s", adapter, exc_info=True,
        )
        return False


async def _power_cycle_adapter_with_cooldown(adapter: str) -> bool:
    """Power-cycle an adapter, respecting a per-adapter cooldown.

    If the adapter was power-cycled less than ``_POWER_CYCLE_COOLDOWN``
    seconds ago, skip the cycle and return ``False``.  This prevents
    rapid cascading cycles when multiple processes detect stale state
    within a short window.

    Returns ``True`` if the power-cycle ran, ``False`` if skipped or
    failed.
    """
    now = time.monotonic()
    last = _last_power_cycle.get(adapter, 0.0)
    if (now - last) < _POWER_CYCLE_COOLDOWN:
        _LOGGER.debug(
            "%s: Power-cycle cooldown active (%.0fs remaining)",
            adapter,
            _POWER_CYCLE_COOLDOWN - (now - last),
        )
        return False

    result = await power_cycle_adapter(adapter)
    if result:
        _last_power_cycle[adapter] = time.monotonic()
    return result


async def _probe_start_discovery(adapter: str) -> bool | None:
    """Probe whether ``StartDiscovery`` works on an adapter.

    Calls ``StartDiscovery`` on the shared D-Bus bus.  If it succeeds,
    immediately calls ``StopDiscovery`` to clean up.

    Returns:
    - ``True`` if the adapter is clean (StartDiscovery succeeded).
    - ``False`` if ``InProgress`` was returned (stale internal state).
    - ``None`` on other errors (e.g. adapter not ready).

    Must be called **while holding the scan lock** to prevent
    interference from other processes.
    """
    if not IS_LINUX:
        return None

    try:
        from dbus_fast import Message, MessageType

        from .dbus_bus import get_bus
    except ImportError:
        return None

    adapter_path = f"/org/bluez/{adapter}"

    try:
        bus = await get_bus()

        reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=adapter_path,
                interface=_ADAPTER_INTERFACE,
                member="StartDiscovery",
            )
        )
        if reply.message_type == MessageType.ERROR:
            error_body = reply.body[0] if reply.body else ""
            error_name = getattr(reply, "error_name", "") or ""
            if "InProgress" in error_name or "InProgress" in str(error_body):
                return False
            _LOGGER.debug(
                "%s: StartDiscovery probe error: %s (%s)",
                adapter,
                error_name,
                error_body,
            )
            return None

        # Probe succeeded — adapter is clean.  Stop our test session.
        stop_reply = await bus.call(
            Message(
                destination=_BLUEZ_SERVICE,
                path=adapter_path,
                interface=_ADAPTER_INTERFACE,
                member="StopDiscovery",
            )
        )
        if stop_reply.message_type == MessageType.ERROR:
            _LOGGER.debug(
                "%s: StopDiscovery after probe failed (non-critical)",
                adapter,
            )
        return True

    except Exception:
        _LOGGER.debug(
            "%s: StartDiscovery probe failed", adapter, exc_info=True,
        )
        return None


async def ensure_adapter_scan_ready(adapter: str) -> bool:
    """Pre-scan health check: detect and repair stale adapter state.

    Must be called **while holding the scan lock** so that no other
    process can interfere during the probe.

    Detects two failure modes:

    1. **Discovering=True** — An orphaned discovery session from a
       previous ``BleakScanner`` that was cancelled without calling
       ``StopDiscovery`` (e.g. hard timeout / Stuck State 16).  Since
       BlueZ ties sessions to D-Bus connections, ``StopDiscovery``
       from our bus won't help.  Fix: power-cycle the adapter.

    2. **Discovering=False + InProgress** — The originating D-Bus
       connection died and BlueZ auto-stopped the discovery, but left
       stale internal state.  Detected by probing with a test
       ``StartDiscovery``.  Fix: power-cycle the adapter.

    Returns ``True`` if the adapter is ready for scanning, ``False``
    if the adapter could not be repaired.
    """
    if not IS_LINUX:
        return True

    # ── Check 1: Orphaned Discovering=True ─────────────────────────
    discovering = await get_adapter_discovering(adapter)

    if discovering is True:
        _LOGGER.warning(
            "%s: Discovering=True with scan lock held — "
            "orphaned scan session, power-cycling adapter",
            adapter,
        )
        if not await _power_cycle_adapter_with_cooldown(adapter):
            return False

        discovering = await get_adapter_discovering(adapter)
        if discovering is True:
            _LOGGER.error(
                "%s: Still Discovering=True after power-cycle", adapter,
            )
            return False
        return True

    # ── Check 2: Hidden stale InProgress ───────────────────────────
    probe_ok = await _probe_start_discovery(adapter)

    if probe_ok is False:
        _LOGGER.warning(
            "%s: Discovering=False but StartDiscovery returns InProgress "
            "— stale internal state, power-cycling adapter",
            adapter,
        )
        if not await _power_cycle_adapter_with_cooldown(adapter):
            return False

    return True
