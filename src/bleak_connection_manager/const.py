"""Constants and configuration dataclasses for bleak-connection-manager."""

from __future__ import annotations

import platform
from dataclasses import dataclass

IS_LINUX = platform.system() == "Linux"

# Thread-level safety timer timeout (seconds).  Must be less than
# BLEAK_SAFETY_TIMEOUT (60 s in bleak-retry-connector) so the asyncio
# timeout remains the primary mechanism and the thread timer is only
# a fallback for a stuck event loop.
THREAD_SAFETY_TIMEOUT = 45.0

# How long to wait for a disconnect to complete before giving up.
DISCONNECT_TIMEOUT = 5.0

# Default number of outer retry attempts.
DEFAULT_MAX_ATTEMPTS = 4


@dataclass
class LockConfig:
    """Configuration for cross-process BLE serialization locks.

    On multi-service systems (e.g. Venus OS / Cerbo GX) several processes
    may compete for the same BLE adapter, causing ``InProgress`` errors
    on ~40% of connection attempts.  Slot-based file locking limits
    concurrent connection attempts per adapter across all processes.

    All services sharing adapters on the same host **must** use the same
    *lock_dir* and *lock_template* to coordinate.

    Parameters
    ----------
    enabled:
        Whether cross-process locking is active.
    lock_dir:
        Directory for lock files.  Defaults to ``/run`` — cleared on
        reboot so stale locks cannot survive reboots.
    lock_template:
        Template with ``{adapter}`` and ``{slot}`` placeholders.
    lock_timeout:
        Maximum seconds to wait for slot acquisition.  If exceeded, the
        connection attempt proceeds without a slot (graceful
        degradation).
    max_slots:
        Maximum concurrent connection attempts allowed per adapter.
        Each slot is a separate lock file.  ``1`` gives strict
        serialization (old behavior).  ``2``-``3`` is typical for a
        single adapter on a Cerbo GX.  Higher values suit systems
        with multiple USB adapters.  ``flock`` is crash-safe — if a
        process dies, the kernel releases its slot automatically.
    """

    enabled: bool = False
    lock_dir: str = "/run"
    lock_template: str = "bleak-cm-{adapter}-slot-{slot}.lock"
    lock_timeout: float = 15.0
    max_slots: int = 2

    def path_for_slot(self, adapter: str | None, slot: int) -> str:
        """Return the full lock file path for a given adapter slot."""
        name = adapter or "default"
        filename = self.lock_template.format(adapter=name, slot=slot)
        return f"{self.lock_dir}/{filename}"

    def path_for_adapter(self, adapter: str | None) -> str:
        """Return the lock file path for slot 0 (backwards compatibility)."""
        return self.path_for_slot(adapter, 0)


@dataclass
class ScanLockConfig:
    """Configuration for cross-process BLE scan serialization.

    BlueZ allows only **one** active scan (``StartDiscovery``) per adapter
    at a time.  When multiple processes call ``BleakScanner.discover()`` or
    ``BleakScanner.find_device_by_address()`` on the same adapter, all but
    the first receive ``org.bluez.Error.InProgress`` (Stuck State 4).

    This config controls a per-adapter exclusive file lock that ensures
    only one process scans on a given adapter at a time.  All services
    on the same host **must** use the same *lock_dir* and *lock_template*
    to coordinate.

    Unlike :class:`LockConfig` (which supports N concurrent slots),
    scan locking is strictly exclusive — BlueZ does not support
    concurrent scans on a single adapter.

    Parameters
    ----------
    enabled:
        Whether cross-process scan locking is active.
    lock_dir:
        Directory for lock files.  Defaults to ``/run`` — cleared on
        reboot so stale locks cannot survive reboots.
    lock_template:
        Template with ``{adapter}`` placeholder.
    lock_timeout:
        Maximum seconds to wait for scan lock acquisition.  If exceeded,
        the scan proceeds without holding the lock (graceful degradation)
        so the caller can still attempt the scan and handle the
        ``InProgress`` error itself.
    """

    enabled: bool = False
    lock_dir: str = "/run"
    lock_template: str = "bleak-cm-{adapter}-scan.lock"
    lock_timeout: float = 30.0

    def path_for_adapter(self, adapter: str | None) -> str:
        """Return the full lock file path for a given adapter."""
        name = adapter or "default"
        filename = self.lock_template.format(adapter=name)
        return f"{self.lock_dir}/{filename}"
