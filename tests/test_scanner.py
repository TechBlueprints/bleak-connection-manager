"""Tests for scanner module — managed scanning with rotation and locking."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

from bleak_connection_manager.const import AdapterScanState, ScanLockConfig
from bleak_connection_manager.scanner import (
    _find_in_bluez_cache,
    _poll_cache_while_locked,
    discover,
    find_device,
)


def _make_device(address="AA:BB:CC:DD:EE:FF", name="TestDevice"):
    return BLEDevice(
        address,
        name,
        {"path": f"/org/bluez/hci0/dev_{address.replace(':', '_')}"},
    )


# ── find_device tests ─────────────────────────────────────────────


# ── Cache-first tests ─────────────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._find_in_bluez_cache")
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_cache_hit_skips_scan(
    mock_scanner_cls, mock_cache
):
    """When device is in BlueZ cache, scanning is skipped entirely."""
    mock_device = _make_device()
    mock_cache.return_value = mock_device

    result = await find_device(
        "AA:BB:CC:DD:EE:FF", max_attempts=3, adapters=["hci0"],
    )

    assert result is mock_device
    # BleakScanner should NEVER have been called
    mock_scanner_cls.find_device_by_address.assert_not_called()


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._find_in_bluez_cache")
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_cache_miss_falls_through_to_scan(
    mock_scanner_cls, mock_cache
):
    """When cache misses, falls through to normal scanning."""
    mock_device = _make_device()
    mock_cache.return_value = None  # cache miss
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    result = await find_device(
        "AA:BB:CC:DD:EE:FF", max_attempts=1, adapters=["hci0"],
        timeout=0.1,
    )

    assert result is mock_device
    mock_scanner_cls.find_device_by_address.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._find_in_bluez_cache")
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_cache_error_falls_through(
    mock_scanner_cls, mock_cache
):
    """Cache lookup error falls through to scanning gracefully."""
    mock_device = _make_device()
    mock_cache.side_effect = Exception("D-Bus error")
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    result = await find_device(
        "AA:BB:CC:DD:EE:FF", max_attempts=1, adapters=["hci0"],
        timeout=0.1,
    )

    assert result is mock_device


# ── Cache polling tests ───────────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._find_in_bluez_cache")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_poll_cache_while_locked_finds_device(mock_cache):
    """Polling cache finds device on second poll."""
    mock_device = _make_device()
    # First poll: miss.  Second poll: hit.
    mock_cache.side_effect = [None, mock_device]

    result = await _poll_cache_while_locked("AA:BB:CC:DD:EE:FF", 2.0)
    assert result is mock_device


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._find_in_bluez_cache")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_poll_cache_while_locked_timeout(mock_cache):
    """Polling cache times out when device is never found."""
    mock_cache.return_value = None

    result = await _poll_cache_while_locked("AA:BB:CC:DD:EE:FF", 0.6)
    assert result is None


# ── Original find_device tests ────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_find_device_success(mock_scanner_cls):
    """Basic find_device returns a device."""
    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    result = await find_device("AA:BB:CC:DD:EE:FF", max_attempts=3)
    assert result is mock_device
    mock_scanner_cls.find_device_by_address.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_find_device_not_found(mock_scanner_cls):
    """Returns None when device is not found."""
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=None)

    result = await find_device(
        "AA:BB:CC:DD:EE:FF", max_attempts=2, timeout=0.1
    )
    assert result is None
    assert mock_scanner_cls.find_device_by_address.call_count == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_find_device_inprogress_rotates(mock_scanner_cls):
    """InProgress error on first attempt triggers rotation."""
    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(
        side_effect=[
            BleakError("org.bluez.Error.InProgress"),
            mock_device,
        ]
    )

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=3,
        adapters=["hci0", "hci1"],
        timeout=0.1,
    )

    assert result is mock_device
    assert mock_scanner_cls.find_device_by_address.call_count == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_passes_adapter_on_linux(mock_scanner_cls):
    """On Linux, the adapter kwarg is passed through to BleakScanner."""
    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        adapters=["hci1"],
        timeout=0.1,
    )

    assert result is mock_device
    call_kwargs = mock_scanner_cls.find_device_by_address.call_args.kwargs
    assert call_kwargs.get("adapter") == "hci1"


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_find_device_timeout_rotates(mock_scanner_cls):
    """Hard timeout triggers rotation to next adapter."""
    mock_device = _make_device()
    call_count = {"n": 0}

    async def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            await asyncio.sleep(100)  # hang on first attempt
        return mock_device

    mock_scanner_cls.find_device_by_address = AsyncMock(side_effect=side_effect)

    # The hard timeout buffer is 5s but we use a tiny scan timeout
    # to make the test fast.  Patch the buffer for the test.
    with patch("bleak_connection_manager.scanner._HARD_TIMEOUT_BUFFER", 0.1):
        result = await find_device(
            "AA:BB:CC:DD:EE:FF",
            max_attempts=3,
            adapters=["hci0", "hci1"],
            timeout=0.1,
        )

    assert result is mock_device
    assert call_count["n"] == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_find_device_adapter_rotation_order(mock_scanner_cls):
    """Adapters rotate across attempts: hci0, hci1, hci0, ..."""
    adapters_used: list[str] = []

    async def track_adapter(*args, **kwargs):
        adapters_used.append(kwargs.get("adapter", "none"))
        return None  # Not found

    mock_scanner_cls.find_device_by_address = AsyncMock(side_effect=track_adapter)

    with patch("bleak_connection_manager.scanner.IS_LINUX", True):
        await find_device(
            "AA:BB:CC:DD:EE:FF",
            max_attempts=4,
            adapters=["hci0", "hci1"],
            timeout=0.1,
        )

    assert adapters_used == ["hci0", "hci1", "hci0", "hci1"]


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_find_device_generic_bleak_error_retries(mock_scanner_cls):
    """Generic BleakError (not InProgress) also retries."""
    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(
        side_effect=[
            BleakError("some other error"),
            mock_device,
        ]
    )

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=3,
        timeout=0.1,
    )

    assert result is mock_device


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_find_device_extra_kwargs_passed(mock_scanner_cls):
    """Extra scanner kwargs are forwarded."""
    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        scanning_mode="passive",
        timeout=0.1,
    )

    call_kwargs = mock_scanner_cls.find_device_by_address.call_args.kwargs
    assert call_kwargs.get("scanning_mode") == "passive"


# ── find_device with scan lock tests ──────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
@patch("bleak_connection_manager.scanner.acquire_scan_lock")
@patch("bleak_connection_manager.scanner.release_scan_lock")
async def test_find_device_with_scan_lock(
    mock_release, mock_acquire, mock_scanner_cls
):
    """Scan lock is acquired and released around the scan."""
    mock_acquire.return_value = 42  # fake fd
    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    cfg = ScanLockConfig(enabled=True)
    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        scan_lock_config=cfg,
        timeout=0.1,
    )

    assert result is mock_device
    mock_acquire.assert_called_once()
    mock_release.assert_called_once_with(42)


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
@patch("bleak_connection_manager.scanner.acquire_scan_lock")
@patch("bleak_connection_manager.scanner.release_scan_lock")
async def test_find_device_lock_contention_rotates(
    mock_release, mock_acquire, mock_scanner_cls
):
    """Lock contention on first adapter rotates to second."""
    mock_device = _make_device()
    # First lock attempt fails (None = contention), second succeeds
    mock_acquire.side_effect = [None, 99]
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    cfg = ScanLockConfig(enabled=True)
    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=3,
        adapters=["hci0", "hci1"],
        scan_lock_config=cfg,
        timeout=0.1,
    )

    assert result is mock_device
    # Two lock attempts: hci0 (failed), hci1 (succeeded)
    assert mock_acquire.call_count == 2
    # Lock released for the successful attempt
    mock_release.assert_called_with(99)


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
@patch("bleak_connection_manager.scanner.acquire_scan_lock")
@patch("bleak_connection_manager.scanner.release_scan_lock")
async def test_find_device_lock_released_on_error(
    mock_release, mock_acquire, mock_scanner_cls
):
    """Lock is released even when the scan raises."""
    mock_acquire.return_value = 42
    mock_scanner_cls.find_device_by_address = AsyncMock(
        side_effect=BleakError("boom")
    )

    cfg = ScanLockConfig(enabled=True)
    await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        scan_lock_config=cfg,
        timeout=0.1,
    )

    mock_release.assert_called_with(42)


# ── discover tests ────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_discover_success(mock_scanner_cls):
    """Basic discover returns a list of devices."""
    mock_devices = [_make_device("AA:BB:CC:DD:EE:01"), _make_device("AA:BB:CC:DD:EE:02")]
    mock_scanner_cls.discover = AsyncMock(return_value=mock_devices)

    result = await discover(max_attempts=1, timeout=0.1)
    assert len(result) == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_discover_inprogress_rotates(mock_scanner_cls):
    """InProgress on discover rotates to next adapter."""
    mock_devices = [_make_device()]
    mock_scanner_cls.discover = AsyncMock(
        side_effect=[
            BleakError("org.bluez.Error.InProgress"),
            mock_devices,
        ]
    )

    result = await discover(
        max_attempts=3,
        adapters=["hci0", "hci1"],
        timeout=0.1,
    )

    assert len(result) == 1
    assert mock_scanner_cls.discover.call_count == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_discover_all_fail_returns_empty(mock_scanner_cls):
    """All attempts fail returns empty list."""
    mock_scanner_cls.discover = AsyncMock(
        side_effect=BleakError("always fails")
    )

    result = await discover(max_attempts=2, timeout=0.1)
    assert result == []


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_discover_passes_adapter_on_linux(mock_scanner_cls):
    """On Linux, adapter kwarg is passed to BleakScanner.discover()."""
    mock_scanner_cls.discover = AsyncMock(return_value=[])

    await discover(
        max_attempts=1,
        adapters=["hci1"],
        timeout=0.1,
    )

    call_kwargs = mock_scanner_cls.discover.call_args.kwargs
    assert call_kwargs.get("adapter") == "hci1"


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", False)
async def test_discover_extra_kwargs_passed(mock_scanner_cls):
    """Extra kwargs are forwarded to BleakScanner.discover()."""
    mock_scanner_cls.discover = AsyncMock(return_value=[])

    await discover(
        max_attempts=1,
        scanning_mode="passive",
        timeout=0.1,
    )

    call_kwargs = mock_scanner_cls.discover.call_args.kwargs
    assert call_kwargs.get("scanning_mode") == "passive"


# ── Pre-scan adapter health check tests ────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._power_cycle_adapter_with_cooldown", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_pre_scan_check_repairs(
    mock_scanner_cls, mock_cache, mock_ready, mock_cycle,
):
    """Pre-scan check detects stale state, repairs, then scan succeeds."""
    mock_cache.return_value = None  # Cache miss
    mock_ready.return_value = AdapterScanState.READY

    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        adapters=["hci0"],
        scan_lock_config=ScanLockConfig(enabled=False),
        timeout=0.1,
    )

    assert result is mock_device
    mock_ready.assert_called_once_with("hci0")


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._power_cycle_adapter_with_cooldown", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_pre_scan_stuck_rotates(
    mock_scanner_cls, mock_cache, mock_ready, mock_cycle,
):
    """Pre-scan check returns STUCK, adapter is skipped."""
    mock_cache.return_value = None
    mock_ready.side_effect = [AdapterScanState.STUCK, AdapterScanState.READY]

    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(return_value=mock_device)

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=2,
        adapters=["hci0", "hci1"],
        scan_lock_config=ScanLockConfig(enabled=False),
        timeout=0.1,
    )

    assert result is mock_device
    assert mock_scanner_cls.find_device_by_address.call_count == 1


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._poll_cache_while_locked", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_external_scan_polls_cache(
    mock_scanner_cls, mock_cache, mock_ready, mock_poll,
):
    """EXTERNAL_SCAN falls back to cache polling, skips BleakScanner."""
    mock_cache.return_value = None
    mock_ready.return_value = AdapterScanState.EXTERNAL_SCAN
    mock_device = _make_device()
    mock_poll.return_value = mock_device

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        adapters=["hci0"],
        scan_lock_config=ScanLockConfig(enabled=False),
        timeout=5.0,
    )

    assert result is mock_device
    mock_poll.assert_called_once_with("AA:BB:CC:DD:EE:FF", 5.0)
    mock_scanner_cls.find_device_by_address.assert_not_called()


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._last_resort_power_cycle", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._poll_cache_while_locked", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_external_scan_no_power_cycle(
    mock_scanner_cls, mock_cache, mock_ready, mock_poll, mock_last_resort,
):
    """EXTERNAL_SCAN never triggers power-cycling."""
    mock_cache.return_value = None
    mock_ready.return_value = AdapterScanState.EXTERNAL_SCAN
    mock_poll.return_value = None
    mock_last_resort.return_value = None

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=2,
        adapters=["hci0", "hci1"],
        scan_lock_config=ScanLockConfig(enabled=False),
        timeout=0.1,
    )

    assert result is None
    mock_last_resort.assert_not_called()
    mock_scanner_cls.find_device_by_address.assert_not_called()


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._last_resort_power_cycle", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._try_recover_adapter", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_post_timeout_tries_recover(
    mock_scanner_cls, mock_cache, mock_ready, mock_recover, mock_last_resort,
):
    """Hard timeout calls _try_recover_adapter, then _last_resort_power_cycle."""
    mock_cache.return_value = None
    mock_ready.return_value = AdapterScanState.READY
    mock_last_resort.return_value = None

    mock_scanner_cls.find_device_by_address = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        adapters=["hci0"],
        scan_lock_config=ScanLockConfig(enabled=False),
        timeout=0.1,
    )

    assert result is None
    mock_recover.assert_called_once()
    assert mock_recover.call_args[0][0] == "hci0"
    mock_last_resort.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._last_resort_power_cycle", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._try_recover_adapter", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_post_inprogress_tries_recover(
    mock_scanner_cls, mock_cache, mock_ready, mock_recover, mock_last_resort,
):
    """InProgress error calls _try_recover_adapter, then _last_resort_power_cycle."""
    mock_cache.return_value = None
    mock_ready.return_value = AdapterScanState.READY
    mock_last_resort.return_value = None

    mock_scanner_cls.find_device_by_address = AsyncMock(
        side_effect=BleakError("org.bluez.Error.InProgress")
    )

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=1,
        adapters=["hci0"],
        scan_lock_config=ScanLockConfig(enabled=False),
        timeout=0.1,
    )

    assert result is None
    mock_recover.assert_called_once()
    assert mock_recover.call_args[0][0] == "hci0"
    mock_last_resort.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.release_scan_lock")
@patch("bleak_connection_manager.scanner.acquire_scan_lock")
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_lock_released_on_cancelled_during_health_check(
    mock_scanner_cls, mock_cache, mock_ready, mock_acquire, mock_release,
):
    """Scan lock is released when CancelledError fires during health check."""
    mock_cache.return_value = None
    mock_acquire.return_value = 42
    mock_ready.side_effect = asyncio.CancelledError()

    cfg = ScanLockConfig(enabled=True)
    with pytest.raises(asyncio.CancelledError):
        await find_device(
            "AA:BB:CC:DD:EE:FF",
            max_attempts=1,
            adapters=["hci0"],
            scan_lock_config=cfg,
            timeout=0.1,
        )

    mock_release.assert_called_with(42)


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner.release_scan_lock")
@patch("bleak_connection_manager.scanner.acquire_scan_lock")
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_discover_lock_released_on_cancelled_during_health_check(
    mock_scanner_cls, mock_ready, mock_acquire, mock_release,
):
    """Scan lock is released when CancelledError fires during discover health check."""
    mock_acquire.return_value = 42
    mock_ready.side_effect = asyncio.CancelledError()

    cfg = ScanLockConfig(enabled=True)
    with pytest.raises(asyncio.CancelledError):
        await discover(
            max_attempts=1,
            adapters=["hci0"],
            scan_lock_config=cfg,
            timeout=0.1,
        )

    mock_release.assert_called_with(42)


@pytest.mark.asyncio
@patch("bleak_connection_manager.scanner._last_resort_power_cycle", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._try_recover_adapter", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.ensure_adapter_scan_ready", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner._find_in_bluez_cache", new_callable=AsyncMock)
@patch("bleak_connection_manager.scanner.BleakScanner")
@patch("bleak_connection_manager.scanner.IS_LINUX", True)
async def test_find_device_no_power_cycle_when_other_adapter_available(
    mock_scanner_cls, mock_cache, mock_ready, mock_recover, mock_last_resort,
):
    """With two adapters, InProgress on one rotates without power-cycle."""
    mock_cache.return_value = None
    mock_ready.return_value = AdapterScanState.READY
    mock_last_resort.return_value = None

    mock_device = _make_device()
    mock_scanner_cls.find_device_by_address = AsyncMock(
        side_effect=[BleakError("org.bluez.Error.InProgress"), mock_device]
    )

    result = await find_device(
        "AA:BB:CC:DD:EE:FF",
        max_attempts=2,
        adapters=["hci0", "hci1"],
        scan_lock_config=ScanLockConfig(enabled=False),
        timeout=0.1,
    )

    assert result is mock_device
    mock_recover.assert_called_once()
    mock_last_resort.assert_not_called()
