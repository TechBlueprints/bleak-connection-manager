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
from .diagnostics import StuckState, clear_stuck_state, diagnose_stuck_state
from .scan_lock import acquire_scan_lock, release_scan_lock

_LOGGER = logging.getLogger(__name__)

# Time to wait after clearing a phantom connection to let the
# peripheral's supervision timer expire and start advertising again.
# The clear itself includes a power cycle with its own wait, so this
# additional wait is just for the peripheral side.
_PHANTOM_CLEAR_WAIT = 3.0

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



async def _find_in_bluez_cache(address: str) -> BLEDevice | None:
    """Check if BlueZ already has a cached device entry from another scanner."""
    try:
        from dbus_fast.aio import MessageBus
        from dbus_fast import BusType
    except ImportError:
        return None

    addr_path_suffix = address.upper().replace(":", "_")
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    try:
        introspect = await bus.introspect("org.bluez", "/")
        proxy = bus.get_proxy_object("org.bluez", "/", introspect)
        iface = proxy.get_interface("org.freedesktop.DBus.ObjectManager")
        objects = await iface.call_get_managed_objects()

        for path, interfaces in objects.items():
            if not path.endswith(addr_path_suffix):
                continue
            dev_props = interfaces.get("org.bluez.Device1", {})
            if not dev_props:
                continue
            dev_addr = dev_props.get("Address", {})
            if hasattr(dev_addr, "value"):
                dev_addr = dev_addr.value
            dev_name = dev_props.get("Name", {})
            if hasattr(dev_name, "value"):
                dev_name = dev_name.value
            dev_rssi = dev_props.get("RSSI", {})
            if hasattr(dev_rssi, "value"):
                dev_rssi = dev_rssi.value
            if isinstance(dev_addr, str) and dev_addr.upper() == address.upper():
                return BLEDevice(
                    address=dev_addr,
                    name=dev_name if isinstance(dev_name, str) else None,
                    rssi=dev_rssi if isinstance(dev_rssi, int) else 0,
                    details={"path": path},
                )
    finally:
        bus.disconnect()

    return None


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

    # Pre-scan phantom check: if BlueZ has a stale "Connected" entry
    # for this device but HCI shows no actual connection, clear it.
    # Without this, the peripheral thinks it's still connected and
    # won't advertise, making all scan attempts fail.
    if IS_LINUX:
        try:
            primary_adapter = effective_adapters[0]
            state = await diagnose_stuck_state(
                address, primary_adapter, adapters=effective_adapters,
            )
            if state in (
                StuckState.PHANTOM_NO_HANDLE,
                StuckState.INACTIVE_CONNECTION,
                StuckState.ORPHAN_HCI_HANDLE,
            ):
                _LOGGER.info(
                    "%s: Pre-scan detected %s, clearing before scan",
                    address,
                    state.value,
                )
                await clear_stuck_state(
                    address, primary_adapter, state,
                    adapters=effective_adapters,
                )
                # Give the peripheral time to notice the link is gone
                # and start advertising again (supervision timeout).
                await asyncio.sleep(_PHANTOM_CLEAR_WAIT)
        except Exception:
            _LOGGER.debug(
                "%s: Pre-scan phantom check failed, proceeding",
                address,
                exc_info=True,
            )

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

    # Last resort: if all scans hit InProgress, check BlueZ cache
    if IS_LINUX and last_error is not None and _is_inprogress(last_error):
        _LOGGER.info(
            "%s: All scans failed with InProgress, checking BlueZ cache",
            address,
        )
        try:
            cached = await _find_in_bluez_cache(address)
            if cached is not None:
                _LOGGER.info(
                    "%s: Found in BlueZ cache (bypassed InProgress)",
                    address,
                )
                return cached
        except Exception:
            _LOGGER.debug(
                "%s: BlueZ cache lookup failed", address, exc_info=True,
            )

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
