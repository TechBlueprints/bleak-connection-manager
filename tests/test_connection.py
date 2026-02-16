"""Tests for connection module — the outer retry loop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak import BleakClient
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

from bleak_connection_manager.connection import (
    _get_device_lock,
    establish_connection,
)
from bleak_connection_manager.const import LockConfig
from bleak_connection_manager.recovery import EscalationConfig, EscalationPolicy


def _make_device(address="AA:BB:CC:DD:EE:FF", name="TestDevice"):
    return BLEDevice(
        address,
        name,
        {"path": f"/org/bluez/hci0/dev_{address.replace(':', '_')}"},
    )


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_successful_connection(mock_brc):
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.return_value = mock_client

    device = _make_device()
    client = await establish_connection(
        BleakClient, device, "TestDevice", max_attempts=3
    )

    assert client is mock_client
    mock_brc.assert_called_once()
    call_kwargs = mock_brc.call_args
    assert call_kwargs.kwargs["max_attempts"] == 1


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_retry_on_failure(mock_brc):
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.side_effect = [
        BleakError("transient failure"),
        mock_client,
    ]

    device = _make_device()
    client = await establish_connection(
        BleakClient, device, "TestDevice", max_attempts=3
    )

    assert client is mock_client
    assert mock_brc.call_count == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_exhausted_attempts(mock_brc):
    mock_brc.side_effect = BleakError("always fails")

    device = _make_device()
    with pytest.raises(Exception, match="Failed to connect"):
        await establish_connection(
            BleakClient, device, "TestDevice", max_attempts=2
        )

    assert mock_brc.call_count == 2


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection._handle_inprogress")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_inprogress_handling(mock_inprogress, mock_brc):
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.side_effect = [
        BleakError("org.bluez.Error.InProgress"),
        mock_client,
    ]

    device = _make_device()
    client = await establish_connection(
        BleakClient, device, "TestDevice", max_attempts=3
    )

    assert client is mock_client
    mock_inprogress.assert_called_once()
    # Verify adapter is passed through
    call_args = mock_inprogress.call_args
    assert call_args.args[0] is device
    assert call_args.args[1] == "hci0"


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_validate_connection_success(mock_brc):
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.return_value = mock_client

    validator = AsyncMock(return_value=True)

    device = _make_device()
    client = await establish_connection(
        BleakClient,
        device,
        "TestDevice",
        max_attempts=3,
        validate_connection=validator,
    )

    assert client is mock_client
    validator.assert_called_once_with(mock_client)


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection._clear_stale_state")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_validate_connection_failure_retries(mock_clear, mock_brc):
    mock_client_bad = MagicMock(spec=BleakClient)
    mock_client_bad.disconnect = AsyncMock()
    mock_client_good = MagicMock(spec=BleakClient)

    mock_brc.side_effect = [mock_client_bad, mock_client_good]
    validator = AsyncMock(side_effect=[False, True])

    device = _make_device()
    client = await establish_connection(
        BleakClient,
        device,
        "TestDevice",
        max_attempts=3,
        validate_connection=validator,
    )

    assert client is mock_client_good
    assert mock_brc.call_count == 2
    mock_client_bad.disconnect.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection._clear_stale_state")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_validate_connection_exception_treated_as_false(mock_clear, mock_brc):
    mock_client_bad = MagicMock(spec=BleakClient)
    mock_client_bad.disconnect = AsyncMock()
    mock_client_good = MagicMock(spec=BleakClient)

    mock_brc.side_effect = [mock_client_bad, mock_client_good]

    async def validator(client):
        if client is mock_client_bad:
            raise RuntimeError("GATT read failed")
        return True

    device = _make_device()
    client = await establish_connection(
        BleakClient,
        device,
        "TestDevice",
        max_attempts=3,
        validate_connection=validator,
    )

    assert client is mock_client_good


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_escalation_policy_on_failure(mock_brc):
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.side_effect = [
        BleakError("fail 1"),
        BleakError("fail 2"),
        mock_client,
    ]

    config = EscalationConfig(reset_adapter=False)
    policy = EscalationPolicy(["hci0"], config=config)

    device = _make_device()
    client = await establish_connection(
        BleakClient,
        device,
        "TestDevice",
        max_attempts=4,
        escalation_policy=policy,
    )

    assert client is mock_client


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", True)
@patch("bleak_connection_manager.connection.discover_adapters", return_value=["hci0", "hci1"])
@patch("bleak_connection_manager.connection._clear_inactive_connections")
async def test_adapter_rotation(mock_clear, mock_discover, mock_brc):
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.side_effect = [
        BleakError("fail on hci0"),
        mock_client,
    ]

    device = _make_device()
    client = await establish_connection(
        BleakClient,
        device,
        "TestDevice",
        max_attempts=3,
        close_inactive_connections=True,
    )

    assert client is mock_client
    assert mock_brc.call_count == 2
    # Second attempt should use a different adapter
    first_call_device = mock_brc.call_args_list[0].args[1]
    second_call_device = mock_brc.call_args_list[1].args[1]
    assert "hci0" in first_call_device.details["path"]
    assert "hci1" in second_call_device.details["path"]


# ── Per-device in-process lock ─────────────────────────────────────


@pytest.mark.asyncio
async def test_get_device_lock_same_address():
    """Same address returns the same lock instance."""
    lock1 = await _get_device_lock("AA:BB:CC:DD:EE:FF")
    lock2 = await _get_device_lock("AA:BB:CC:DD:EE:FF")
    assert lock1 is lock2


@pytest.mark.asyncio
async def test_get_device_lock_case_insensitive():
    """Address lookup is case-insensitive."""
    lock1 = await _get_device_lock("aa:bb:cc:dd:ee:ff")
    lock2 = await _get_device_lock("AA:BB:CC:DD:EE:FF")
    assert lock1 is lock2


@pytest.mark.asyncio
async def test_get_device_lock_different_addresses():
    """Different addresses return different lock instances."""
    lock1 = await _get_device_lock("AA:BB:CC:DD:EE:FF")
    lock2 = await _get_device_lock("11:22:33:44:55:66")
    assert lock1 is not lock2


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_device_lock_serializes_same_device(mock_brc):
    """Two concurrent connections to the same device are serialized."""
    call_order: list[str] = []

    async def slow_connect(*args, **kwargs):
        call_order.append("start")
        await asyncio.sleep(0.1)
        call_order.append("end")
        return MagicMock(spec=BleakClient)

    mock_brc.side_effect = slow_connect

    device = _make_device()

    # Launch two concurrent connections to the same device
    task1 = asyncio.create_task(
        establish_connection(BleakClient, device, "Dev1", max_attempts=1)
    )
    task2 = asyncio.create_task(
        establish_connection(BleakClient, device, "Dev2", max_attempts=1)
    )

    await asyncio.gather(task1, task2)

    # With the per-device lock, connections should be serialized:
    # start, end, start, end  (not start, start, end, end)
    assert call_order == ["start", "end", "start", "end"]


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_device_lock_allows_different_devices_concurrently(mock_brc):
    """Connections to different devices can proceed concurrently."""
    active_count = {"n": 0, "max": 0}

    async def slow_connect(*args, **kwargs):
        active_count["n"] += 1
        active_count["max"] = max(active_count["max"], active_count["n"])
        await asyncio.sleep(0.1)
        active_count["n"] -= 1
        return MagicMock(spec=BleakClient)

    mock_brc.side_effect = slow_connect

    device_a = _make_device("AA:BB:CC:DD:EE:01", "DevA")
    device_b = _make_device("AA:BB:CC:DD:EE:02", "DevB")

    task1 = asyncio.create_task(
        establish_connection(BleakClient, device_a, "DevA", max_attempts=1)
    )
    task2 = asyncio.create_task(
        establish_connection(BleakClient, device_b, "DevB", max_attempts=1)
    )

    await asyncio.gather(task1, task2)

    # Both should have been active concurrently
    assert active_count["max"] == 2


# ── try_direct_first tests ─────────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_try_direct_first_passes_use_services_cache(mock_brc):
    """First attempt passes use_services_cache=True when try_direct_first=True."""
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.return_value = mock_client

    device = _make_device()
    client = await establish_connection(
        BleakClient, device, "TestDevice",
        max_attempts=3,
        try_direct_first=True,
    )

    assert client is mock_client
    call_kwargs = mock_brc.call_args.kwargs
    assert call_kwargs.get("use_services_cache") is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_try_direct_first_second_attempt_no_cache(mock_brc):
    """Second attempt does NOT pass use_services_cache when first failed."""
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.side_effect = [
        BleakError("cache miss"),
        mock_client,
    ]

    device = _make_device()
    client = await establish_connection(
        BleakClient, device, "TestDevice",
        max_attempts=3,
        try_direct_first=True,
    )

    assert client is mock_client
    assert mock_brc.call_count == 2
    # First attempt should have use_services_cache=True
    first_kwargs = mock_brc.call_args_list[0].kwargs
    assert first_kwargs.get("use_services_cache") is True
    # Second attempt should NOT have use_services_cache set
    second_kwargs = mock_brc.call_args_list[1].kwargs
    assert "use_services_cache" not in second_kwargs


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection.IS_LINUX", False)
async def test_try_direct_first_false_no_cache(mock_brc):
    """Without try_direct_first, use_services_cache is NOT set."""
    mock_client = MagicMock(spec=BleakClient)
    mock_brc.return_value = mock_client

    device = _make_device()
    client = await establish_connection(
        BleakClient, device, "TestDevice",
        max_attempts=3,
        try_direct_first=False,
    )

    assert client is mock_client
    call_kwargs = mock_brc.call_args.kwargs
    assert "use_services_cache" not in call_kwargs


# ── Verified disconnect in validation failure path ─────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.connection._brc_establish_connection")
@patch("bleak_connection_manager.connection._clear_stale_state")
@patch("bleak_connection_manager.connection.verified_disconnect", new_callable=AsyncMock)
@patch("bleak_connection_manager.connection.IS_LINUX", True)
@patch("bleak_connection_manager.connection.discover_adapters", return_value=["hci0"])
async def test_validation_failure_uses_verified_disconnect(
    mock_discover, mock_verified, mock_clear, mock_brc
):
    """When validation fails, verified_disconnect is called to confirm D-Bus state."""
    mock_client_bad = MagicMock(spec=BleakClient)
    mock_client_bad.disconnect = AsyncMock()
    mock_client_good = MagicMock(spec=BleakClient)

    mock_brc.side_effect = [mock_client_bad, mock_client_good]
    validator = AsyncMock(side_effect=[False, True])

    device = _make_device()
    client = await establish_connection(
        BleakClient,
        device,
        "TestDevice",
        max_attempts=3,
        validate_connection=validator,
    )

    assert client is mock_client_good
    mock_verified.assert_called_once()
    # Verify it was called with the device address and adapter
    call_args = mock_verified.call_args
    assert call_args.args[0] == "AA:BB:CC:DD:EE:FF"
    assert call_args.args[1] == "hci0"
