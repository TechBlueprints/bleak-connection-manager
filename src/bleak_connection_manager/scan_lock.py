"""Cross-process exclusive scan lock for BLE adapters.

BlueZ allows only **one** ``StartDiscovery`` per adapter.  When multiple
processes call ``BleakScanner.discover()`` on the same adapter, all but
the first get ``org.bluez.Error.InProgress`` (Stuck State 4).

This module provides a per-adapter exclusive file lock (``fcntl.flock``)
so that only one process scans a given adapter at a time.  It is
designed to be used alongside the connection slot locks in
:mod:`bleak_connection_manager.lock`, which manage *connection* concurrency.
Scan and connection locks are independent resources — holding a scan lock
does not block connection attempts, and vice versa.

**How it works:**

Each adapter gets **one** lock file (e.g. ``/run/bleak-cm-hci0-scan.lock``).
Before calling ``BleakScanner.start()`` / ``discover()`` /
``find_device_by_address()``, a process acquires the lock.  When
scanning is complete, the lock is released.

**Usage patterns:**

1. **Low-level acquire / release** — for maximum control::

    fd = await acquire_scan_lock(config, "hci0")
    try:
        device = await BleakScanner.find_device_by_address(addr, ...)
    finally:
        release_scan_lock(fd)

2. **Async context manager** — for convenience::

    async with ScanLock(config, "hci0"):
        device = await BleakScanner.find_device_by_address(addr, ...)

**Graceful degradation:**

If the lock cannot be acquired within the configured timeout, the caller
proceeds without the lock.  This avoids deadlocking the entire BLE
subsystem when a process crashes while holding the lock (although
``flock`` is kernel-managed and crash-safe, the timeout is still useful
for processes that hold the lock for too long due to BlueZ hangs).

**Why this can't deadlock:**

Same reasoning as connection slot locks:

- Every ``flock`` call is non-blocking (``LOCK_NB``).  If the lock
  isn't free, we release, sleep, retry.
- ``flock`` is kernel-managed: if a process crashes, the kernel
  releases the lock automatically.  No stale counters.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .const import ScanLockConfig

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

_LOGGER = logging.getLogger(__name__)

_RETRY_INTERVAL = 0.25


async def acquire_scan_lock(
    config: ScanLockConfig,
    adapter: str | None,
) -> int | None:
    """Acquire an exclusive scan lock for the given adapter.

    Tries to ``flock(LOCK_EX | LOCK_NB)`` the adapter's scan lock file.
    If the lock is held by another process, retries until *config.lock_timeout*
    expires.

    Parameters
    ----------
    config:
        Scan lock configuration.
    adapter:
        Adapter name (e.g. ``"hci0"``).

    Returns
    -------
    int | None
        An open file descriptor holding the scan lock, or ``None`` if
        the lock could not be acquired (timeout or unavailable).
    """
    if not _HAS_FCNTL or not config.enabled:
        return None

    lock_path = config.path_for_adapter(adapter)
    elapsed = 0.0

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        except OSError:
            _LOGGER.debug(
                "Failed to open scan lock file %s",
                lock_path,
                exc_info=True,
            )
            return None

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _LOGGER.debug(
                "Acquired scan lock for %s (%s)",
                adapter,
                lock_path,
            )
            return fd
        except OSError:
            # Lock is held — close and retry after a sleep
            os.close(fd)

        elapsed += _RETRY_INTERVAL
        if elapsed >= config.lock_timeout:
            _LOGGER.warning(
                "Timed out waiting for scan lock on %s after %.1f s "
                "— proceeding without lock",
                adapter,
                elapsed,
            )
            return None

        await asyncio.sleep(_RETRY_INTERVAL)


def release_scan_lock(fd: int | None) -> None:
    """Release a previously acquired scan lock.

    Safe to call with ``None`` (no-op).
    """
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        _LOGGER.debug("Failed to release scan lock", exc_info=True)


class ScanLock:
    """Async context manager for per-adapter scan locking.

    Usage::

        async with ScanLock(config, "hci0"):
            device = await BleakScanner.find_device_by_address(addr, ...)

    If the lock cannot be acquired, the context manager still enters
    (graceful degradation) but logs a warning.

    Parameters
    ----------
    config:
        Scan lock configuration.
    adapter:
        Adapter name (e.g. ``"hci0"``).
    """

    __slots__ = ("_config", "_adapter", "_fd")

    def __init__(self, config: ScanLockConfig, adapter: str | None = "hci0") -> None:
        self._config = config
        self._adapter = adapter
        self._fd: int | None = None

    async def __aenter__(self) -> "ScanLock":
        self._fd = await acquire_scan_lock(self._config, self._adapter)
        return self

    async def __aexit__(self, *exc: object) -> None:
        release_scan_lock(self._fd)
        self._fd = None

    @property
    def acquired(self) -> bool:
        """Whether the scan lock is currently held."""
        return self._fd is not None
