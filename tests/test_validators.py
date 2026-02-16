"""Tests for validators module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from bleak_connection_manager.validators import (
    validate_char_exists,
    validate_gatt_services,
    validate_read_char,
)


def _make_char(uuid: str) -> MagicMock:
    """Create a mock BleakGATTCharacteristic."""
    char = MagicMock()
    char.uuid = uuid
    return char


def _make_service(char_uuids: list[str]) -> MagicMock:
    """Create a mock BleakGATTService with the given characteristics."""
    service = MagicMock()
    service.characteristics = [_make_char(u) for u in char_uuids]
    return service


def _make_client(
    address: str = "AA:BB:CC:DD:EE:FF",
    services: list | None = None,
) -> MagicMock:
    """Create a mock BleakClient with optional services."""
    client = MagicMock()
    client.address = address
    client.services = services if services is not None else []
    return client


# ── validate_gatt_services ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_gatt_services_empty():
    client = _make_client(services=[])
    assert await validate_gatt_services(client) is False


@pytest.mark.asyncio
async def test_gatt_services_none():
    client = _make_client(services=None)
    assert await validate_gatt_services(client) is False


@pytest.mark.asyncio
async def test_gatt_services_present():
    service = _make_service(["00002a19-0000-1000-8000-00805f9b34fb"])
    client = _make_client(services=[service])
    assert await validate_gatt_services(client) is True


# ── validate_char_exists ────────────────────────────────────────────


BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"


@pytest.mark.asyncio
async def test_char_exists_empty_services():
    validator = validate_char_exists(BATTERY_UUID)
    client = _make_client(services=[])
    assert await validator(client) is False


@pytest.mark.asyncio
async def test_char_exists_not_found():
    service = _make_service([NUS_TX_UUID])
    client = _make_client(services=[service])
    validator = validate_char_exists(BATTERY_UUID)
    assert await validator(client) is False


@pytest.mark.asyncio
async def test_char_exists_found():
    service = _make_service([BATTERY_UUID, NUS_TX_UUID])
    client = _make_client(services=[service])
    validator = validate_char_exists(BATTERY_UUID)
    assert await validator(client) is True


@pytest.mark.asyncio
async def test_char_exists_case_insensitive():
    service = _make_service([BATTERY_UUID.upper()])
    client = _make_client(services=[service])
    validator = validate_char_exists(BATTERY_UUID.lower())
    assert await validator(client) is True


@pytest.mark.asyncio
async def test_char_exists_multiple_services():
    service1 = _make_service(["0000180a-0000-1000-8000-00805f9b34fb"])
    service2 = _make_service([BATTERY_UUID])
    client = _make_client(services=[service1, service2])
    validator = validate_char_exists(BATTERY_UUID)
    assert await validator(client) is True


@pytest.mark.asyncio
async def test_char_exists_has_docstring():
    validator = validate_char_exists(BATTERY_UUID)
    assert BATTERY_UUID in validator.__doc__


# ── validate_read_char ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_char_empty_services():
    validator = validate_read_char(BATTERY_UUID)
    client = _make_client(services=[])
    assert await validator(client) is False


@pytest.mark.asyncio
async def test_read_char_char_not_found():
    service = _make_service([NUS_TX_UUID])
    client = _make_client(services=[service])
    validator = validate_read_char(BATTERY_UUID)
    assert await validator(client) is False


@pytest.mark.asyncio
async def test_read_char_success():
    service = _make_service([BATTERY_UUID])
    client = _make_client(services=[service])
    client.read_gatt_char = AsyncMock(return_value=b"\x64")
    validator = validate_read_char(BATTERY_UUID)
    assert await validator(client) is True
    client.read_gatt_char.assert_called_once_with(BATTERY_UUID)


@pytest.mark.asyncio
async def test_read_char_read_fails():
    service = _make_service([BATTERY_UUID])
    client = _make_client(services=[service])
    client.read_gatt_char = AsyncMock(side_effect=Exception("Not connected"))
    validator = validate_read_char(BATTERY_UUID)
    assert await validator(client) is False


@pytest.mark.asyncio
async def test_read_char_read_timeout():
    service = _make_service([BATTERY_UUID])
    client = _make_client(services=[service])

    async def _slow_read(uuid):
        await asyncio.sleep(10.0)
        return b"\x00"

    client.read_gatt_char = _slow_read
    validator = validate_read_char(BATTERY_UUID, timeout=0.1)
    assert await validator(client) is False


@pytest.mark.asyncio
async def test_read_char_has_docstring():
    validator = validate_read_char(BATTERY_UUID)
    assert BATTERY_UUID in validator.__doc__


@pytest.mark.asyncio
async def test_read_char_case_insensitive_lookup():
    service = _make_service([BATTERY_UUID.upper()])
    client = _make_client(services=[service])
    client.read_gatt_char = AsyncMock(return_value=b"\x42")
    validator = validate_read_char(BATTERY_UUID.lower())
    assert await validator(client) is True
