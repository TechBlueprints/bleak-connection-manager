"""Managed BLE scanning with adapter rotation, locking, and InProgress retry.

This is the scan counterpart to :mod:`bleak_connection_manager.connection`.
It wraps ``BleakScanner`` with:

- **Per-adapter scan lock** — cross-process ``fcntl.flock`` so only one
  process scans a given adapter at a time (Stuck State 4 fix).
- **Adapter rotation** — if the scan lock on the current adapter can't be
  acquired quickly, or if BlueZ returns ``InProgress``, automatically
  rotate to the next adapter.
- **Hard timeout** — wraps every ``BleakScanner`` call in
  ``asyncio.wait_for()`` to guard against Stuck State 16 (scanner hangs
  ignoring its own timeout parameter).
- **Retry with backoff** — transient scan failures get retried across
  adapters before giving up.

Each workaround is independently removable.  When upstream or BlueZ
fixes the root cause, the corresponding code path can be deleted.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

from .adapters import discover_adapters, pick_adapter
from .const import IS_LINUX, ScanLockConfig
from .scan_lock import acquire_scan_lock, release_scan_lock

_LOGGER = logging.getLogger(__name__)

# Extra seconds added to the BleakScanner timeout to form the hard
# asyncio.wait_for timeout.  This gives BleakScanner a chance to
# return normally before the hard timeout fires.
_HARD_TIMEOUT_BUFFER = 5.0

# How long to try acquiring the scan lock on a single adapter before
# giving up and rotating.  Short — we'd rather try another adapter
# than wait a long time for one.
_LOCK_ATTEMPT_TIMEOUT = 2.0

# Brief pause between retries on different adapters.
_RETRY_BACKOFF = 0.25


def _is_inprogress(exc: BaseException) -> bool:
    """Check if an exception is an InProgress error."""
    err_str = str(exc).lower()
    return "inprogress" in err_str or "in progress" in err_str


async def find_device(
    address: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = 3,
    adapters: list[str] | None = None,
    scan_lock_config: ScanLockConfig | None = None,
    **scanner_kwargs: Any,
) -> BLEDevice | None:
    """Find a BLE device by address with automatic adapter rotation.

    Drop-in replacement for ``BleakScanner.find_device_by_address()``
    with scan lock, adapter rotation, and InProgress retry built in.

    Parameters
    ----------
    address:
        The BLE device MAC address to find.
    timeout:
        Per-attempt scan timeout in seconds.  Each attempt on each
        adapter gets this much time.
    max_attempts:
        Maximum total attempts across all adapters.
    adapters:
        List of adapters to rotate through.  If ``None``, auto-discovered.
    scan_lock_config:
        Cross-process scan lock configuration.  If ``None`` or
        ``enabled=False``, no locking is performed.
    **scanner_kwargs:
        Additional keyword arguments passed to
        ``BleakScanner.find_device_by_address()`` (e.g. ``scanning_mode``).

    Returns
    -------
    BLEDevice | None
        The found device, or ``None`` if not found after all attempts.
    """
    if adapters is None and IS_LINUX:
        adapters = discover_adapters()
    effective_adapters = adapters or ["hci0"]

    hard_timeout = timeout + _HARD_TIMEOUT_BUFFER
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        adapter = pick_adapter(effective_adapters, attempt)

        # --- Try to acquire the scan lock for this adapter ---
        fd: int | None = None
        if scan_lock_config is not None and scan_lock_config.enabled:
            lock_cfg_for_attempt = ScanLockConfig(
                enabled=True,
                lock_dir=scan_lock_config.lock_dir,
                lock_template=scan_lock_config.lock_template,
                lock_timeout=min(_LOCK_ATTEMPT_TIMEOUT, scan_lock_config.lock_timeout),
            )
            fd = await acquire_scan_lock(lock_cfg_for_attempt, adapter)
            if fd is None and len(effective_adapters) > 1:
                _LOGGER.debug(
                    "%s: Scan lock busy on %s, rotating (attempt %d/%d)",
                    address,
                    adapter,
                    attempt,
                    max_attempts,
                )
                continue

        try:
            _LOGGER.debug(
                "%s: Scanning on %s (attempt %d/%d, timeout=%.1f s)",
                address,
                adapter,
                attempt,
                max_attempts,
                timeout,
            )

            kwargs: dict[str, Any] = dict(scanner_kwargs)
            if IS_LINUX:
                kwargs.setdefault("adapter", adapter)

            device = await asyncio.wait_for(
                BleakScanner.find_device_by_address(
                    address,
                    timeout=timeout,
                    **kwargs,
                ),
                timeout=hard_timeout,
            )

            if device is not None:
                _LOGGER.debug(
                    "%s: Found on %s (attempt %d/%d)",
                    address,
                    adapter,
                    attempt,
                    max_attempts,
                )
                return device

            _LOGGER.debug(
                "%s: Not found on %s (attempt %d/%d)",
                address,
                adapter,
                attempt,
                max_attempts,
            )

        except (BleakError, asyncio.TimeoutError) as exc:
            last_error = exc
            if _is_inprogress(exc):
                _LOGGER.debug(
                    "%s: InProgress on %s, rotating (attempt %d/%d)",
                    address,
                    adapter,
                    attempt,
                    max_attempts,
                )
            elif isinstance(exc, asyncio.TimeoutError):
                _LOGGER.warning(
                    "%s: Scanner hard timeout on %s after %.0f s "
                    "(Stuck State 16), rotating (attempt %d/%d)",
                    address,
                    adapter,
                    hard_timeout,
                    attempt,
                    max_attempts,
                )
            else:
                _LOGGER.debug(
                    "%s: Scan error on %s: %s (attempt %d/%d)",
                    address,
                    adapter,
                    exc,
                    attempt,
                    max_attempts,
                )

        finally:
            release_scan_lock(fd)

        if attempt < max_attempts:
            await asyncio.sleep(_RETRY_BACKOFF)

    if last_error is not None:
        _LOGGER.warning(
            "%s: All %d scan attempts failed, last error: %s",
            address,
            max_attempts,
            last_error,
        )
    return None


async def discover(
    *,
    timeout: float = 5.0,
    max_attempts: int = 2,
    adapters: list[str] | None = None,
    scan_lock_config: ScanLockConfig | None = None,
    **scanner_kwargs: Any,
) -> list[BLEDevice]:
    """Discover BLE devices with automatic adapter rotation.

    Drop-in replacement for ``BleakScanner.discover()`` with scan lock,
    adapter rotation, and InProgress retry built in.

    Parameters
    ----------
    timeout:
        Per-attempt scan duration in seconds.
    max_attempts:
        Maximum total attempts across all adapters.
    adapters:
        List of adapters to rotate through.  If ``None``, auto-discovered.
    scan_lock_config:
        Cross-process scan lock configuration.  If ``None`` or
        ``enabled=False``, no locking is performed.
    **scanner_kwargs:
        Additional keyword arguments passed to
        ``BleakScanner.discover()`` (e.g. ``scanning_mode``).

    Returns
    -------
    list[BLEDevice]
        List of discovered devices.  May be empty if all attempts fail.
    """
    if adapters is None and IS_LINUX:
        adapters = discover_adapters()
    effective_adapters = adapters or ["hci0"]

    hard_timeout = timeout + _HARD_TIMEOUT_BUFFER

    for attempt in range(1, max_attempts + 1):
        adapter = pick_adapter(effective_adapters, attempt)

        fd: int | None = None
        if scan_lock_config is not None and scan_lock_config.enabled:
            lock_cfg_for_attempt = ScanLockConfig(
                enabled=True,
                lock_dir=scan_lock_config.lock_dir,
                lock_template=scan_lock_config.lock_template,
                lock_timeout=min(_LOCK_ATTEMPT_TIMEOUT, scan_lock_config.lock_timeout),
            )
            fd = await acquire_scan_lock(lock_cfg_for_attempt, adapter)
            if fd is None and len(effective_adapters) > 1:
                _LOGGER.debug(
                    "Scan lock busy on %s, rotating (attempt %d/%d)",
                    adapter,
                    attempt,
                    max_attempts,
                )
                continue

        try:
            _LOGGER.debug(
                "Discovering on %s (attempt %d/%d, timeout=%.1f s)",
                adapter,
                attempt,
                max_attempts,
                timeout,
            )

            kwargs: dict[str, Any] = dict(scanner_kwargs)
            if IS_LINUX:
                kwargs.setdefault("adapter", adapter)

            devices = await asyncio.wait_for(
                BleakScanner.discover(
                    timeout=timeout,
                    **kwargs,
                ),
                timeout=hard_timeout,
            )

            _LOGGER.debug(
                "Discovered %d devices on %s (attempt %d/%d)",
                len(devices),
                adapter,
                attempt,
                max_attempts,
            )
            return list(devices)

        except (BleakError, asyncio.TimeoutError) as exc:
            if _is_inprogress(exc):
                _LOGGER.debug(
                    "InProgress on %s, rotating (attempt %d/%d)",
                    adapter,
                    attempt,
                    max_attempts,
                )
            elif isinstance(exc, asyncio.TimeoutError):
                _LOGGER.warning(
                    "Scanner hard timeout on %s after %.0f s "
                    "(Stuck State 16), rotating (attempt %d/%d)",
                    adapter,
                    hard_timeout,
                    attempt,
                    max_attempts,
                )
            else:
                _LOGGER.debug(
                    "Scan error on %s: %s (attempt %d/%d)",
                    adapter,
                    exc,
                    attempt,
                    max_attempts,
                )

        finally:
            release_scan_lock(fd)

        if attempt < max_attempts:
            await asyncio.sleep(_RETRY_BACKOFF)

    return []
