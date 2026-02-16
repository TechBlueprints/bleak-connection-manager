"""Stuck-state diagnosis and targeted recovery for BLE connections.

Provides :func:`diagnose_stuck_state` to determine why a BLE connection
is stuck, and :func:`clear_stuck_state` to apply the minimal targeted
fix for each diagnosis.

Diagnosis uses two layers:

1. **D-Bus** (via ``bluez`` module) — BlueZ's cached view of the world.
2. **HCI** (via ``hci`` module) — the kernel's ground-truth connection
   list.

Cross-referencing these two layers is the only reliable way to detect
phantom connections and stale handles.  No shell tools
(``bluetoothctl``, ``hcitool``) are required.
"""

from __future__ import annotations

import logging
from enum import Enum

from .bluez import _get_device_properties, disconnect_device, remove_device
from .const import IS_LINUX
from .hci import disconnect_by_address, find_connection_by_address

_LOGGER = logging.getLogger(__name__)


class StuckState(Enum):
    """Diagnosis of why a BLE connection is stuck.

    Each value maps to a specific recovery action in
    :func:`clear_stuck_state`.
    """

    NOT_STUCK = "not_stuck"
    INACTIVE_CONNECTION = "inactive_connection"
    STALE_CACHE = "stale_cache"
    ORPHAN_HCI_HANDLE = "orphan_hci_handle"
    PHANTOM_NO_HANDLE = "phantom_no_handle"


async def diagnose_stuck_state(
    address: str,
    adapter: str,
    adapters: list[str] | None = None,
) -> StuckState:
    """Determine what kind of stuck state a device is in.

    Cross-references D-Bus properties with HCI connection state:

    - **PHANTOM_NO_HANDLE**: D-Bus ``Connected=True`` but no HCI handle
      exists on any adapter.  This is the classic phantom — BlueZ
      believes the device is connected but there is no radio link.
      (Stuck State 1)
    - **INACTIVE_CONNECTION**: D-Bus ``Connected=True``,
      ``ServicesResolved`` is not ``True``, and an HCI handle exists.
      The radio link is up but GATT never resolved — dead handle.
      (Stuck State 2)
    - **ORPHAN_HCI_HANDLE**: D-Bus does NOT show the device as connected,
      but an HCI handle exists.  The service that created the connection
      died without cleaning up, and the peripheral is stuck in connected
      mode (won't advertise).  (Stuck State 20)
    - **STALE_CACHE**: Device exists on D-Bus but is not connected and
      has no HCI handle.  Stale BlueZ cache that may cause
      ``InProgress`` errors.  (Stuck State 13)
    - **NOT_STUCK**: Device appears healthy or is not present.

    Parameters
    ----------
    address:
        The BLE device MAC address.
    adapter:
        The primary adapter name (e.g. ``"hci0"``).
    adapters:
        All available adapters.  If provided, HCI connections are
        checked across all of them.  If ``None``, only *adapter* is
        checked.
    """
    if not IS_LINUX:
        return StuckState.NOT_STUCK

    # Layer 1: Check HCI (ground truth)
    search_adapters = adapters if adapters else [adapter]
    hci_conn = find_connection_by_address(
        address, adapters=search_adapters
    )

    # Layer 2: Check D-Bus (BlueZ's cached view)
    props = await _get_device_properties(address, adapter)

    dbus_connected = False
    dbus_exists = props is not None
    if props is not None:
        dbus_connected = props.get("Connected", False)

    # Cross-reference the two layers
    if dbus_connected:
        services_resolved = props.get("ServicesResolved", False)  # type: ignore[union-attr]
        if hci_conn is None:
            # D-Bus says connected, HCI says no handle → PHANTOM
            return StuckState.PHANTOM_NO_HANDLE
        if services_resolved is not True:
            # D-Bus connected + HCI handle exists + GATT not resolved
            return StuckState.INACTIVE_CONNECTION
        # Both layers agree: connected with services resolved → healthy
        return StuckState.NOT_STUCK

    # D-Bus does NOT show connected
    if hci_conn is not None:
        # D-Bus says not connected (or not present), but HCI handle
        # exists → orphan handle from a crashed service.  The peripheral
        # is stuck in connected mode and won't advertise.
        return StuckState.ORPHAN_HCI_HANDLE

    if dbus_exists:
        # Device in D-Bus cache but not connected, no HCI handle
        return StuckState.STALE_CACHE

    # Not in D-Bus, no HCI handle → clean state
    return StuckState.NOT_STUCK


async def clear_stuck_state(
    address: str,
    adapter: str,
    state: StuckState,
    adapters: list[str] | None = None,
) -> bool:
    """Apply the targeted fix for a diagnosed stuck state.

    - **PHANTOM_NO_HANDLE**: Disconnect + remove via D-Bus (clears the
      stale BlueZ state since there's no HCI handle to clear).
    - **INACTIVE_CONNECTION**: Disconnect via D-Bus (which sends HCI
      Disconnect since a handle exists), then remove.
    - **ORPHAN_HCI_HANDLE**: Disconnect at the HCI level to free the
      peripheral, then remove from D-Bus.  This is the fix for
      Stuck State 20 -- D-Bus ``RemoveDevice`` alone does NOT clear
      HCI handles.
    - **STALE_CACHE**: Remove the device from BlueZ cache.

    Returns ``True`` if the cleanup action was executed.
    """
    if state == StuckState.NOT_STUCK:
        return True

    if not IS_LINUX:
        return False

    if state == StuckState.PHANTOM_NO_HANDLE:
        _LOGGER.info(
            "%s: Clearing phantom connection (D-Bus Connected=True "
            "but no HCI handle)",
            address,
        )
        await disconnect_device(address, adapter)
        await remove_device(address, adapter)
        return True

    if state == StuckState.INACTIVE_CONNECTION:
        _LOGGER.info(
            "%s: Clearing inactive connection (Connected but "
            "ServicesResolved is not True)",
            address,
        )
        await disconnect_device(address, adapter)
        await remove_device(address, adapter)
        return True

    if state == StuckState.ORPHAN_HCI_HANDLE:
        _LOGGER.info(
            "%s: Clearing orphan HCI handle (peripheral stuck in "
            "connected mode, won't advertise)",
            address,
        )
        search_adapters = adapters if adapters else [adapter]
        disconnect_by_address(address, adapters=search_adapters)
        await remove_device(address, adapter)
        return True

    if state == StuckState.STALE_CACHE:
        _LOGGER.info(
            "%s: Removing stale BlueZ cache entry on %s",
            address,
            adapter,
        )
        await remove_device(address, adapter)
        return True

    return False
