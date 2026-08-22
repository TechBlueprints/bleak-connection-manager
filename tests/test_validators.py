# -*- coding: utf-8 -*-
"""Tests for the post-connect validators.

The module is duck-typed on the client, so the fakes here are plain
objects: a service table shaped like bleak's (services iterate services,
services have .characteristics, characteristics have .uuid) and a backend
that counts GATT re-reads. No bleak, no stubs in sys.modules - importing
the validators must not need either.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bleak_connection_manager import validators  # noqa: E402

ADDRESS = "C8:47:8C:00:00:00"
CHAR = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


class _Char:
    def __init__(self, uuid):
        self.uuid = uuid


class _Service:
    def __init__(self, uuid, chars):
        self.uuid = uuid
        self.characteristics = [_Char(c) for c in chars]


class _Backend:
    """Stands in for bleak's platform backend: services resolve once and are
    cached on the backend, which is what the re-read has to defeat."""

    def __init__(self, tables, read_error=None):
        self._tables = list(tables)
        self.services = self._tables.pop(0)
        self.reads = 0
        self.read_error = read_error

    async def _get_services(self, dangerous_use_bleak_cache=False):
        self.reads += 1
        assert dangerous_use_bleak_cache is False
        if self.read_error is not None:
            raise self.read_error
        self.services = self._tables.pop(0) if self._tables else []


class _Client:
    def __init__(self, tables=([],), reads=None, is_connected=True, read_error=None):
        self._backend = _Backend(tables, read_error)
        self.address = ADDRESS
        self.is_connected = is_connected
        self._reads = list(reads or [])

    @property
    def services(self):
        return self._backend.services

    async def read_gatt_char(self, uuid):
        result = self._reads.pop(0) if self._reads else b"\x64"
        if isinstance(result, BaseException):
            raise result
        return result


def _gatt(*chars):
    return [_Service("fff0", chars)]


# -- the three built-ins ---------------------------------------------------


def test_gatt_services_validator_rejects_an_empty_table():
    assert asyncio.run(validators.validate_gatt_services(_Client())) is False
    assert asyncio.run(validators.validate_gatt_services(_Client([_gatt(CHAR)]))) is True


def test_char_exists_validator_is_case_insensitive_and_wants_that_char():
    validate = validators.validate_char_exists(CHAR.lower())

    assert asyncio.run(validate(_Client([_gatt(CHAR)]))) is True
    assert asyncio.run(validate(_Client([_gatt("00002a19-0000-1000-8000-00805f9b34fb")]))) is False
    assert asyncio.run(validate(_Client())) is False


def test_read_char_validator_needs_the_read_to_succeed():
    validate = validators.validate_read_char(CHAR)

    assert asyncio.run(validate(_Client([_gatt(CHAR)]))) is True
    assert asyncio.run(validate(_Client([_gatt(CHAR)], reads=[Exception("not connected")]))) is False
    assert asyncio.run(validate(_Client([_gatt("fff1")]))) is False


def test_read_char_validator_gives_up_on_a_hung_read():
    async def _hang(uuid):
        await asyncio.sleep(10)

    client = _Client([_gatt(CHAR)])
    client.read_gatt_char = _hang

    assert asyncio.run(validators.validate_read_char(CHAR, timeout=0.01)(client)) is False


# -- late GATT registration ------------------------------------------------


def test_late_gatt_services_pass_on_a_later_re_read(monkeypatch):
    """Telink-style: ServicesResolved fires with only Generic Attribute, the
    vendor services land seconds later."""
    slept = []

    async def _sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(validators.asyncio, "sleep", _sleep)
    client = _Client([[], [], _gatt(CHAR)])
    validate = validators.tolerate_late_gatt(validators.validate_char_exists(CHAR))

    assert asyncio.run(validate(client)) is True
    assert slept == [2.0, 4.0]  # passed on the second wait, never used the third
    assert client._backend.reads == 2


def test_late_gatt_gives_up_after_the_last_wait(monkeypatch):
    async def _sleep(seconds):
        return None

    monkeypatch.setattr(validators.asyncio, "sleep", _sleep)
    client = _Client([[], [], [], []])
    validate = validators.tolerate_late_gatt(validators.validate_gatt_services)

    assert asyncio.run(validate(client)) is False
    assert client._backend.reads == 3  # one per wait, then done


def test_late_gatt_stops_waiting_once_the_link_is_gone(monkeypatch):
    """A really dead link fails the GATT re-read; with the property agreeing,
    the retry is abandoned on the first wait rather than burning all three."""

    async def _sleep(seconds):
        return None

    monkeypatch.setattr(validators.asyncio, "sleep", _sleep)
    client = _Client([[]], is_connected=False, read_error=RuntimeError("Not connected"))
    validate = validators.tolerate_late_gatt(validators.validate_gatt_services)

    assert asyncio.run(validate(client)) is False
    assert client._backend.reads == 1  # probed once, abandoned


def test_late_gatt_does_not_abandon_on_a_stale_is_connected(monkeypatch):
    """The false-negative that costs a good link: after an adapter reset or
    a device object removed and re-added, BlueZ sends no property signal and
    bleak's cached is_connected is stranded False while GATT still works. A
    rejection here would disconnect the link, release its claims and rotate
    the radio, so the re-read - which succeeds - is what decides."""

    async def _sleep(seconds):
        return None

    monkeypatch.setattr(validators.asyncio, "sleep", _sleep)
    client = _Client([[], [_Service("fff0", [CHAR])]], is_connected=False)
    validate = validators.tolerate_late_gatt(validators.validate_char_exists(CHAR))

    assert asyncio.run(validate(client)) is True
    assert client._backend.reads == 1  # the late service arrived, link kept


def test_late_gatt_treats_a_raising_validator_as_a_rejection(monkeypatch):
    async def _sleep(seconds):
        return None

    monkeypatch.setattr(validators.asyncio, "sleep", _sleep)
    calls = []

    async def _boom(client):
        calls.append(client)
        raise RuntimeError("backend gone")

    assert asyncio.run(validators.tolerate_late_gatt(_boom)(_Client())) is False
    assert len(calls) == 4  # the first check plus one per wait


def test_a_passing_validator_never_waits(monkeypatch):
    def _no(*args, **kwargs):
        raise AssertionError("must not sleep")

    monkeypatch.setattr(validators.asyncio, "sleep", _no)
    client = _Client([_gatt(CHAR)])

    assert asyncio.run(validators.tolerate_late_gatt(validators.validate_gatt_services)(client)) is True
    assert client._backend.reads == 0


# -- the GATT re-read itself ----------------------------------------------


def test_refresh_services_clears_the_backend_cache_and_re_resolves():
    client = _Client([[], _gatt(CHAR)])

    assert asyncio.run(validators.refresh_services(client)) is True
    assert client.services == client._backend.services
    assert [c.uuid for s in client.services for c in s.characteristics] == [CHAR]


def test_refresh_services_survives_a_backend_that_cannot_re_read():
    class _Bare:
        address = ADDRESS
        _backend = object()

    assert asyncio.run(validators.refresh_services(_Bare())) is False
    assert asyncio.run(validators.refresh_services(_Client.__new__(_Client))) is False


def test_refresh_services_reports_a_raising_re_read():
    client = _Client([[]])

    async def _boom(**kwargs):
        raise RuntimeError("dbus gone")

    client._backend._get_services = _boom

    assert asyncio.run(validators.refresh_services(client)) is False


def test_refresh_services_tolerates_an_older_get_services_signature():
    calls = []

    class _Old:
        services = None

        async def get_services(self):
            calls.append(True)

    class _Client9:
        _backend = _Old()

    assert asyncio.run(validators.refresh_services(_Client9())) is True
    assert calls == [True]
