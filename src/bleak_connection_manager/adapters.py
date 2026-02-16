"""Adapter enumeration and rotation for multi-adapter BLE systems.

Wraps ``bluetooth-adapters`` for enumeration and adds round-robin
rotation logic for distributing connection attempts across adapters.

When multiple USB BLE adapters are available, rotating between them
avoids saturating a single adapter and works around per-adapter
connection limits in BlueZ.
"""

from __future__ import annotations

import logging
from typing import Any

from bleak.backends.device import BLEDevice

from .const import IS_LINUX

_LOGGER = logging.getLogger(__name__)


def discover_adapters() -> list[str]:
    """Discover available BLE adapters on the system.

    Uses ``bluetooth-adapters`` when available, falls back to
    ``/sys/class/bluetooth/`` enumeration.

    Returns a sorted list of adapter names (e.g. ``["hci0", "hci1"]``).
    Returns ``["hci0"]`` as a safe default if no adapters are found.
    """
    if not IS_LINUX:
        return ["hci0"]

    # Try bluetooth-adapters first (more reliable, handles USB adapters)
    try:
        from bluetooth_adapters import get_adapters_from_hci

        adapters_from_hci = get_adapters_from_hci()
        if adapters_from_hci:
            names = sorted(a["name"] for a in adapters_from_hci.values())
            if names:
                _LOGGER.debug("Discovered adapters via bluetooth-adapters: %s", names)
                return names
    except Exception:
        _LOGGER.debug(
            "bluetooth-adapters enumeration failed, trying /sys",
            exc_info=True,
        )

    # Fallback: /sys/class/bluetooth/
    try:
        import pathlib

        bt_path = pathlib.Path("/sys/class/bluetooth")
        if bt_path.exists():
            adapters = sorted(
                d.name for d in bt_path.iterdir() if d.name.startswith("hci")
            )
            if adapters:
                _LOGGER.debug("Discovered adapters via /sys: %s", adapters)
                return adapters
    except Exception:
        _LOGGER.debug("Failed to enumerate /sys/class/bluetooth", exc_info=True)

    return ["hci0"]


def pick_adapter(
    adapters: list[str],
    attempt: int,
) -> str:
    """Pick an adapter for the current attempt using round-robin.

    Parameters
    ----------
    adapters:
        List of available adapter names (e.g. ``["hci0", "hci1"]``).
    attempt:
        The current attempt number (1-based).

    Returns the adapter to use for this attempt.
    """
    if not adapters:
        return "hci0"
    idx = (attempt - 1) % len(adapters)
    return adapters[idx]


def make_device_for_adapter(
    device: BLEDevice,
    adapter: str,
) -> BLEDevice:
    """Create a BLEDevice targeting a specific adapter.

    Constructs a new ``BLEDevice`` with the D-Bus path pointing to the
    chosen adapter, so that ``bleak-retry-connector`` connects through
    it.

    Parameters
    ----------
    device:
        The original BLEDevice (from scanning).
    adapter:
        The adapter to target (e.g. ``"hci0"``).

    Returns a new BLEDevice with the adapter-specific path.
    """
    from .bluez import address_to_bluez_path

    path = address_to_bluez_path(device.address, adapter)
    details: dict[str, Any] = {"path": path}

    if isinstance(device.details, dict):
        # Preserve any extra details from the original device
        details.update(
            {k: v for k, v in device.details.items() if k != "path"}
        )

    return BLEDevice(
        device.address,
        device.name,
        details,
    )
