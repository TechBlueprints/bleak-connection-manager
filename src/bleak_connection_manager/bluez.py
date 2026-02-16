"""BlueZ D-Bus utilities for connection state inspection.

Provides functions to detect phantom/inactive BLE connections and
construct BlueZ D-Bus object paths.  Uses ``dbus-fast`` directly
for D-Bus queries — the same library bleak uses internally.

These functions fill a gap: ``bleak-retry-connector`` does not expose
connection state inspection, and ``bleak`` itself does not provide
phantom detection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .const import IS_LINUX

_LOGGER = logging.getLogger(__name__)

# D-Bus constants
_BLUEZ_SERVICE = "org.bluez"
_DEVICE_INTERFACE = "org.bluez.Device1"
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
        from dbus_fast.aio import MessageBus
        from dbus_fast.constants import BusType
    except ImportError:
        _LOGGER.debug("dbus-fast not available, cannot query D-Bus")
        return None

    path = address_to_bluez_path(address, adapter)

    try:
        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        try:
            introspection = await bus.introspect(_BLUEZ_SERVICE, path)
            proxy = bus.get_proxy_object(_BLUEZ_SERVICE, path, introspection)
            props_iface = proxy.get_interface(
                "org.freedesktop.DBus.Properties"
            )
            props = await props_iface.call_get_all(_DEVICE_INTERFACE)
            return {k: v.value for k, v in props.items()}
        finally:
            bus.disconnect()
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
        from dbus_fast.aio import MessageBus
        from dbus_fast.constants import BusType
    except ImportError:
        return False

    adapter_path = f"/org/bluez/{adapter}"
    device_path = address_to_bluez_path(address, adapter)

    try:
        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        try:
            introspection = await bus.introspect(_BLUEZ_SERVICE, adapter_path)
            proxy = bus.get_proxy_object(
                _BLUEZ_SERVICE, adapter_path, introspection
            )
            adapter_iface = proxy.get_interface("org.bluez.Adapter1")
            await adapter_iface.call_remove_device(device_path)
            _LOGGER.debug("Removed device %s from %s", address, adapter)
            return True
        finally:
            bus.disconnect()
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
        from dbus_fast.aio import MessageBus
        from dbus_fast.constants import BusType
    except ImportError:
        return False

    path = address_to_bluez_path(address, adapter)

    try:
        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        try:
            introspection = await bus.introspect(_BLUEZ_SERVICE, path)
            proxy = bus.get_proxy_object(_BLUEZ_SERVICE, path, introspection)
            device_iface = proxy.get_interface(_DEVICE_INTERFACE)
            await device_iface.call_disconnect()
            _LOGGER.debug("Disconnected %s on %s via D-Bus", address, adapter)
            return True
        finally:
            bus.disconnect()
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
