"""Shared D-Bus system bus for BlueZ diagnostic queries.

Provides a single long-lived ``MessageBus`` connection to the system
D-Bus daemon, reused across all BCM diagnostic operations (device
property queries, remove, disconnect, adapter state inspection, etc.).

Why a shared bus?
-----------------

Previously, each diagnostic function created its own temporary
``MessageBus``, used it for one query, and immediately disconnected.
This caused two problems:

1. **Spurious errors**: ``dbus-fast``'s ``get_proxy_object()`` triggers
   ``_init_high_level_client()`` which fire-and-forgets an ``AddMatch``
   for ``NameOwnerChanged``.  If the bus disconnects before the reply
   arrives, the callback logs ``add match request failed`` at ERROR
   level.

2. **Unnecessary overhead**: Each temporary bus creates a Unix socket,
   authenticates, and tears down.  For the shyion service polling 7
   devices per hour, that was 14+ bus connections created and destroyed
   per poll cycle.

The shared bus eliminates both issues: the ``AddMatch`` (if it occurs)
succeeds because the bus stays alive, and all queries reuse one
connection.

All functions use raw ``bus.call(Message(...))`` instead of proxy
objects, which avoids triggering ``_init_high_level_client()`` entirely
and also skips the unnecessary ``bus.introspect()`` round-trip.

Thread / async safety
---------------------

The bus is lazily created on first use and auto-reconnects if the
connection drops.  All access goes through :func:`get_bus`, which is
safe to call from any coroutine in the same event loop.  There is no
cross-thread sharing.
"""

from __future__ import annotations

import asyncio
import logging

from .const import IS_LINUX

_LOGGER = logging.getLogger(__name__)

_bus: object | None = None  # dbus_fast.aio.MessageBus, typed loosely to avoid import on non-Linux
_bus_loop: object | None = None  # The event loop the bus was created on
_bluez_ready = False  # Set True once we've confirmed org.bluez is on D-Bus


async def get_bus():
    """Get the shared system D-Bus connection, creating or reconnecting as needed.

    Returns a connected ``dbus_fast.aio.MessageBus`` instance.

    If the running event loop differs from the one the bus was created
    on, the old bus is discarded and a fresh one is created.  This
    prevents ``Future attached to a different loop`` RuntimeErrors when
    the caller's event loop changes (e.g. ``asyncio.run()`` creates a
    new loop between calls).

    Raises ``ImportError`` if ``dbus-fast`` is not available, or
    ``RuntimeError`` on non-Linux platforms.
    """
    global _bus, _bus_loop

    if not IS_LINUX:
        raise RuntimeError("Shared D-Bus bus is only available on Linux")

    from dbus_fast.aio import MessageBus
    from dbus_fast.constants import BusType

    current_loop = asyncio.get_running_loop()

    if _bus is not None:
        if _bus_loop is not current_loop:
            _LOGGER.debug(
                "Shared D-Bus bus was created on a different event loop, "
                "reconnecting on current loop"
            )
            try:
                _bus.disconnect()
            except Exception:
                pass
            _bus = None
        elif _bus.connected:
            return _bus
        else:
            _LOGGER.debug("Shared D-Bus bus disconnected, reconnecting")

    _bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    _bus_loop = current_loop
    _LOGGER.debug("Shared D-Bus bus connected")
    return _bus


async def _ping_bluez() -> bool:
    """Single fast D-Bus check for ``org.bluez`` availability.

    Returns ``True`` if BlueZ responded (any non-ServiceUnknown reply).
    """
    from dbus_fast import Message, MessageType

    try:
        bus = await get_bus()
        reply = await bus.call(
            Message(
                destination="org.bluez",
                path="/org/bluez",
                interface="org.freedesktop.DBus.Properties",
                member="GetAll",
                signature="s",
                body=["org.bluez.AgentManager1"],
            )
        )
        if reply.message_type != MessageType.ERROR:
            return True
        error_name = reply.error_name or ""
        if "ServiceUnknown" in error_name or "UnknownObject" in error_name:
            return False
        return True
    except Exception:
        return False


async def _poll_bluez(timeout: float, poll_interval: float) -> bool:
    """Poll D-Bus for ``org.bluez`` up to *timeout* seconds.

    Returns ``True`` as soon as BlueZ responds, ``False`` on timeout.
    """
    elapsed = 0.0
    while elapsed < timeout:
        if await _ping_bluez():
            if elapsed > 0:
                _LOGGER.info("BlueZ ready on D-Bus after %.1fs", elapsed)
            else:
                _LOGGER.debug("BlueZ ready on D-Bus")
            return True

        _LOGGER.debug(
            "Waiting for BlueZ on D-Bus (%.1fs / %.0fs)...",
            elapsed, timeout,
        )
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        # Reconnect bus if it dropped while we were waiting
        await get_bus()

    return False


async def wait_for_bluez(
    timeout: float = 30.0,
    poll_interval: float = 1.0,
) -> bool:
    """Wait until ``org.bluez`` is available on the system D-Bus.

    On Venus OS, BLE services may start before ``bluetoothd`` has
    registered on D-Bus.  Any BlueZ operation attempted before
    ``org.bluez`` is available fails with::

        org.freedesktop.DBus.Error.ServiceUnknown:
        The name org.bluez was not provided by any .service files

    This function polls D-Bus until ``org.bluez`` is reachable or
    *timeout* seconds have elapsed.

    If the initial poll times out, the function attempts to restart
    ``bluetoothd`` via ``/etc/init.d/bluetooth start`` and polls for
    an additional 10 seconds.

    The ``_bluez_ready`` cache is validated with a single D-Bus ping
    on every call, so a crashed ``bluetoothd`` is detected even when
    the cache says BlueZ was previously healthy.

    Returns ``True`` if BlueZ became available, ``False`` on timeout.
    """
    global _bluez_ready

    if not IS_LINUX:
        return True

    # Validate the cache: if we previously marked BlueZ as ready,
    # do a quick ping to confirm it's still alive.  This catches
    # bluetoothd crashes that happened since the last successful check.
    if _bluez_ready:
        if await _ping_bluez():
            return True
        _LOGGER.warning(
            "BlueZ was previously available but is no longer responding "
            "on D-Bus — bluetoothd may have crashed"
        )
        _bluez_ready = False

    # Normal poll loop
    if await _poll_bluez(timeout, poll_interval):
        _bluez_ready = True
        return True

    # Poll exhausted — attempt to restart bluetoothd
    _LOGGER.warning(
        "BlueZ did not appear on D-Bus after %.0fs — "
        "attempting to restart bluetoothd",
        timeout,
    )

    from .recovery import restart_bluetoothd

    if await restart_bluetoothd():
        if await _poll_bluez(10.0, poll_interval):
            _bluez_ready = True
            return True
        _LOGGER.error(
            "bluetoothd restarted but BlueZ still not on D-Bus after 10s"
        )
    else:
        _LOGGER.error(
            "Failed to restart bluetoothd — BLE operations will fail"
        )

    return False


async def close_bus() -> None:
    """Disconnect the shared bus if it's open.

    Safe to call even if no bus was ever created.
    """
    global _bus, _bus_loop
    if _bus is not None:
        try:
            _bus.disconnect()
        except Exception:
            pass
        _bus = None
        _bus_loop = None
