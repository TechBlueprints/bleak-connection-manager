# -*- coding: utf-8 -*-
"""Tests for the bleak catcher: process-wide client routing, habluetooth style.

A rich bleak stub mimics the bleak 3.x contract the wrapper relies on:
__init__ builds the platform backend eagerly (which is why the wrapper must
defer it), connect()/disconnect()/is_connected all delegate to
self._backend. A bleak_retry_connector stub carries the two rebound names.
The stubs are swapped into sys.modules while the catcher module is first
imported and while each test runs, so the catcher's captured originals are
the stub classes regardless of what the environment has installed.

Foreign processes are simulated with pid-1 claim files (kill(1, 0) raises
PermissionError, which counts as alive); stale ones with a dead pid and an
aged mtime.
"""

import asyncio
import contextlib
import gc
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ADDRESS = "C8:47:8C:00:00:00"

# scripts the rich stubs consume, reset per test
CONNECT_RESULTS = []
INIT_RESULTS = []
RECORDED_INITS = []
SCANNER_START_RESULTS = []
SCANNER_INIT_RESULTS = []
RECORDED_SCANNER_INITS = []
GATT_RESULTS = []


class _FakeBus:
    """Stands in for the per-session dbus-fast MessageBus bleak opens."""

    def __init__(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    async def wait_for_disconnect(self):
        return None


class _RichBackend:
    def __init__(self, adapter, address):
        self.adapter = adapter
        self.address = address
        self.is_connected = False
        self.disconnects = 0
        self.disconnect_error = None
        self._bus = _FakeBus()
        self.notify_callbacks = {}

    async def connect(self, pair, **kwargs):
        result = CONNECT_RESULTS.pop(0) if CONNECT_RESULTS else True
        if isinstance(result, BaseException):  # CancelledError is not an Exception
            raise result
        self.is_connected = True

    async def disconnect(self):
        self.disconnects += 1
        if self.disconnect_error is not None:
            # bleak's real disconnect() closes its bus in statements AFTER
            # the try/finally, so a raise here leaves _bus attached
            raise self.disconnect_error
        self.is_connected = False
        bus = getattr(self, "_bus", None)
        if bus is not None:
            bus.disconnect()
            self._bus = None

    async def start_notify(self, char_specifier, callback, **kwargs):
        self.notify_callbacks[char_specifier] = callback

    async def _gatt(self):
        result = GATT_RESULTS.pop(0) if GATT_RESULTS else b"\x00"
        if isinstance(result, BaseException):
            raise result
        return result

    async def read_gatt_char(self, char_specifier, **kwargs):
        return await self._gatt()

    async def write_gatt_char(self, char_specifier, data, response=None):
        return await self._gatt()

    async def read_gatt_descriptor(self, handle, **kwargs):
        return await self._gatt()

    async def write_gatt_descriptor(self, handle, data):
        return await self._gatt()


class RichBleakClient:
    """Just like bleak 3.x: the backend is wired up inside __init__."""

    def __init__(self, address_or_ble_device, disconnected_callback=None, services=None, *, timeout=30, pair=False, bluez=None, backend=None, **kwargs):
        if INIT_RESULTS:
            raise INIT_RESULTS.pop(0)
        adapter = (bluez or {}).get("adapter")
        address = getattr(address_or_ble_device, "address", address_or_ble_device)
        RECORDED_INITS.append(
            {
                "address": address,
                "adapter": adapter,
                "services": services,
                "extra": sorted(kwargs),
                "disconnected_callback": disconnected_callback,
            }
        )
        self._backend = _RichBackend(adapter, address)
        self._pair_before_connect = pair

    async def connect(self, **kwargs):
        await self._backend.connect(self._pair_before_connect, **kwargs)

    async def disconnect(self):
        await self._backend.disconnect()

    @property
    def is_connected(self):
        return self._backend.is_connected

    async def start_notify(self, char_specifier, callback, **kwargs):
        await self._backend.start_notify(char_specifier, callback, **kwargs)

    async def read_gatt_char(self, char_specifier, **kwargs):
        return await self._backend.read_gatt_char(char_specifier, **kwargs)

    async def write_gatt_char(self, char_specifier, data, response=None):
        return await self._backend.write_gatt_char(char_specifier, data, response)

    async def read_gatt_descriptor(self, handle, **kwargs):
        return await self._backend.read_gatt_descriptor(handle, **kwargs)

    async def write_gatt_descriptor(self, handle, data):
        return await self._backend.write_gatt_descriptor(handle, data)


class _RichScannerBackend:
    def __init__(self, adapter):
        self.adapter = adapter
        self.scanning = False

    async def start(self):
        result = SCANNER_START_RESULTS.pop(0) if SCANNER_START_RESULTS else True
        if isinstance(result, BaseException):  # CancelledError is not an Exception
            raise result
        self.scanning = True

    async def stop(self):
        self.scanning = False


class RichBleakScanner:
    """Just like bleak 3.x: the backend is wired up inside __init__, and the
    BlueZ backend receives the adapter via the backwards-compat kwarg."""

    def __init__(self, detection_callback=None, service_uuids=None, scanning_mode="active", *, bluez=None, backend=None, **kwargs):
        if SCANNER_INIT_RESULTS:
            raise SCANNER_INIT_RESULTS.pop(0)
        adapter = kwargs.get("adapter", (bluez or {}).get("adapter"))
        RECORDED_SCANNER_INITS.append(
            {
                "adapter": adapter,
                "bluez_adapter": (bluez or {}).get("adapter"),
                "extra": sorted(k for k in kwargs if k != "adapter"),
                "detection_callback": detection_callback,
            }
        )
        self._backend = _RichScannerBackend(adapter)
        self._backend_id = "stub"

    async def start(self):
        await self._backend.start()

    async def stop(self):
        await self._backend.stop()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @property
    def discovered_devices(self):
        return ["stub-device"]


def _make_stub_modules():
    exc = types.ModuleType("bleak.exc")
    exc.BleakError = type("BleakError", (Exception,), {})
    exc.BleakDeviceNotFoundError = type("BleakDeviceNotFoundError", (Exception,), {})
    bleak = types.ModuleType("bleak")
    bleak.BleakClient = RichBleakClient
    bleak.BleakScanner = RichBleakScanner
    bleak.exc = exc

    brc = types.ModuleType("bleak_retry_connector")
    brc.BleakClient = RichBleakClient

    class StubServiceCacheClient(RichBleakClient):
        def set_cached_services(self, services):
            return None

    brc.BleakClientWithServiceCache = StubServiceCacheClient
    return bleak, exc, brc


STUB_BLEAK, STUB_BLEAK_EXC, STUB_BRC = _make_stub_modules()


@contextlib.contextmanager
def _stubs_installed():
    names = ("bleak", "bleak.exc", "bleak_retry_connector")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["bleak"] = STUB_BLEAK
    sys.modules["bleak.exc"] = STUB_BLEAK_EXC
    sys.modules["bleak_retry_connector"] = STUB_BRC
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_catcher():
    """Import the catcher bound to the stubs, whatever the env has installed."""
    with _stubs_installed():
        import bleak_connection_manager.catcher as module
    return module


catcher = _load_catcher()


def _foreign_file(claim_dir, name, pid=1, aged=None):
    path = os.path.join(claim_dir, name)
    os.makedirs(claim_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{pid} foreign-svc {int(time.time())}\n")
    if aged:
        old = time.time() - aged
        os.utime(path, (old, old))
    return path


OWNER = "test-svc"


def _soft_name(adapter, mac=ADDRESS):
    return f"{adapter}.use.{OWNER}-{os.getpid()}.{mac.replace(':', '')}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Stubs in sys.modules, fresh rotation and warning state, a tmp claim
    directory, no adapters "present" unless a test says otherwise."""
    CONNECT_RESULTS.clear()
    INIT_RESULTS.clear()
    RECORDED_INITS.clear()
    SCANNER_START_RESULTS.clear()
    SCANNER_INIT_RESULTS.clear()
    RECORDED_SCANNER_INITS.clear()
    GATT_RESULTS.clear()
    catcher._warned_bare_connect_addresses.clear()
    monkeypatch.setattr(catcher, "_rotation", catcher.BleAdapterRotation())
    monkeypatch.setattr(catcher, "_connect_failures", {})
    monkeypatch.setattr(catcher, "_scan_failures", {})
    monkeypatch.setattr(catcher, "present_adapters", lambda: set())

    def install(adapters=(), link_caps=None, wrap_scanner=False, scan_to_score=False, validate_connection=None, adapter_config_path=None):
        catcher.install_bleak_catcher(
            OWNER,
            adapters=adapters,
            link_caps=link_caps,
            adapter_config_path=adapter_config_path,
            claim_dir=str(tmp_path),
            wrap_scanner=wrap_scanner,
            scan_to_score=scan_to_score,
            validate_connection=validate_connection,
        )

    with _stubs_installed():
        yield types.SimpleNamespace(install=install, dir=str(tmp_path))
        catcher.uninstall_bleak_catcher()


# -- install/uninstall and wrapper basics (ported from the prototype) ------


def test_install_rebinds_bleak_and_brc_and_uninstall_restores(env):
    brc = sys.modules["bleak_retry_connector"]
    assert sys.modules["bleak"].BleakClient is RichBleakClient
    env.install()
    assert sys.modules["bleak"].BleakClient is catcher.BLEConnection
    assert brc.BleakClient is catcher.BLEConnection
    assert brc.BleakClientWithServiceCache is catcher.BLEConnectionWithServiceCache
    env.install()  # idempotent
    assert sys.modules["bleak"].BleakClient is catcher.BLEConnection
    catcher.uninstall_bleak_catcher()
    assert sys.modules["bleak"].BleakClient is RichBleakClient
    assert brc.BleakClient is RichBleakClient
    assert brc.BleakClientWithServiceCache is STUB_BRC.BleakClientWithServiceCache


def test_a_late_importer_gets_the_wrapper_and_placeholders_are_inert(env):
    """
    aiobmsble's BaseBMS does `from bleak import BleakClient` when it is first
    imported and builds a placeholder client immediately. After install, that
    binding must be the wrapper, and the placeholder must be inert: nothing
    constructed, not connected, but still answering is_connected and address.
    """
    env.install()
    BleakClient = sys.modules["bleak"].BleakClient  # what a late import binds
    device = types.SimpleNamespace(address=ADDRESS)
    placeholder = BleakClient(device, disconnected_callback=None, services=["fff0", "180a"])

    assert isinstance(placeholder, catcher.BLEConnection)
    assert RECORDED_INITS == []  # nothing built before connect
    assert placeholder.is_connected is False
    assert placeholder.address == ADDRESS
    asyncio.run(placeholder.disconnect())  # noop, must not raise


def test_the_captured_originals_cannot_recurse(env):
    env.install()
    assert catcher._ORIGINAL_BLEAK_CLIENT is RichBleakClient
    assert catcher._ORIGINAL_BRC_CLIENT is RichBleakClient


def test_connect_routes_the_selected_adapter_into_bluez_args(env):
    env.install(adapters=("hci7",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert [r["adapter"] for r in RECORDED_INITS] == ["hci7"]
    assert client.is_connected is True


def test_a_failed_connect_walks_to_the_next_adapter_on_the_same_instance(env):
    """
    bleak-retry-connector retries connect() on one client instance; each
    attempt must re-run the selection so the walk continues mid-retry.
    """
    env.install(adapters=("hci5", "hci6"))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    CONNECT_RESULTS.append(RuntimeError("le-connection-abort-by-local"))

    with pytest.raises(RuntimeError):
        asyncio.run(client.connect())
    asyncio.run(client.connect())

    assert [r["adapter"] for r in RECORDED_INITS] == ["hci5", "hci6"]
    assert client.is_connected is True


def test_a_dropped_link_reconnects_on_the_same_adapter(env):
    """A disconnect is not a failed attempt and must not advance the walk."""
    env.install(adapters=("hci5", "hci6"))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())
    asyncio.run(client.disconnect())
    asyncio.run(client.connect())

    assert [r["adapter"] for r in RECORDED_INITS] == ["hci5", "hci5"]


def test_an_adapter_chosen_by_the_caller_is_never_overridden(env):
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True, bluez={"adapter": "hci9"})

    asyncio.run(client.connect())

    assert [r["adapter"] for r in RECORDED_INITS] == ["hci9"]
    # visibility even for explicit choices: the connection soft-claims the
    # card it actually uses, so other processes' occupancy scores see it
    assert os.listdir(env.dir) == [_soft_name("hci9")]


def test_without_configured_adapters_the_wrapper_is_a_passthrough(env):
    env.install(adapters=())
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True, services=["fff0"])

    asyncio.run(client.connect())

    assert len(RECORDED_INITS) == 1
    recorded = RECORDED_INITS[0]
    assert recorded["address"] == ADDRESS
    assert recorded["adapter"] is None
    assert recorded["services"] == ["fff0"]
    assert recorded["extra"] == []
    assert client.is_connected is True
    assert os.listdir(env.dir) == []


UNRETRIED_WARNING = "called without bleak-retry-connector"


def test_a_client_built_by_establish_connection_connects_without_the_warning(env, caplog):
    """
    bleak-retry-connector marks the clients it constructs with
    _is_retry_client=True; the marker must silence the warning and must not
    leak into the real client's kwargs.
    """
    env.install()
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert UNRETRIED_WARNING not in caplog.text
    assert RECORDED_INITS[0]["extra"] == []  # marker was popped, not passed down


def test_a_bare_connect_warns_that_it_has_no_retry(env, caplog):
    env.install()
    client = sys.modules["bleak"].BleakClient(ADDRESS)

    asyncio.run(client.connect())

    assert UNRETRIED_WARNING in caplog.text
    assert ADDRESS in caplog.text


def test_the_bare_connect_warning_fires_once_per_address(env, caplog):
    """A reconnect loop must not repeat this into the log every pass."""
    env.install()
    client = sys.modules["bleak"].BleakClient(ADDRESS)

    asyncio.run(client.connect())
    asyncio.run(client.disconnect())
    asyncio.run(client.connect())
    other = sys.modules["bleak"].BleakClient(ADDRESS)
    asyncio.run(other.connect())

    assert caplog.text.count(UNRETRIED_WARNING) == 1


def test_rotation_state_is_shared_across_client_instances(env):
    """
    A caller that builds a fresh client per attempt must still continue the
    walk instead of restarting it.
    """
    env.install(adapters=("hci5", "hci6"))

    first = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    CONNECT_RESULTS.append(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        asyncio.run(first.connect())

    second = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(second.connect())

    assert [r["adapter"] for r in RECORDED_INITS] == ["hci5", "hci6"]


# -- claims integration: slots, soft claims, releases, avoidance -----------


def test_a_connect_holds_a_link_slot_and_a_soft_claim(env):
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert set(os.listdir(env.dir)) == {"hci5.link.0", _soft_name("hci5")}


def test_an_uncapped_adapter_takes_no_slot_but_writes_a_soft_claim(env):
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert os.listdir(env.dir) == [_soft_name("hci5")]


def test_a_full_adapter_is_skipped_without_a_failure_penalty(env):
    """Slot exhaustion says nothing about the radio: the next eligible
    adapter is tried and no failure is recorded - a busy card must not be
    scored down like a broken one. (The freed cap-1 card then legitimately
    keeps habluetooth's last-slot penalty, so the uncapped card wins ties.)"""
    env.install(adapters=("hci5", "hci6"), link_caps={"hci5": 1})
    blocker = _foreign_file(env.dir, "hci5.link.0")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())
    assert RECORDED_INITS[-1]["adapter"] == "hci6"
    assert catcher._connect_failures == {}  # exhaustion is not failure

    asyncio.run(client.disconnect())
    os.unlink(blocker)
    asyncio.run(client.connect())
    assert RECORDED_INITS[-1]["adapter"] == "hci6"  # uncapped beats last-slot


def test_every_adapter_full_raises_the_typed_error_with_occupancy(env):
    env.install(adapters=("hci5", "hci6"), link_caps={"hci5": 1, "hci6": 1})
    _foreign_file(env.dir, "hci5.link.0")
    _foreign_file(env.dir, "hci6.link.0")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    with pytest.raises(catcher.OutOfConnectionSlotsError) as excinfo:
        asyncio.run(client.connect())

    message = str(excinfo.value)
    assert "connection slot" in message  # brc's OUT_OF_SLOTS_ERRORS match
    assert "hci5 (1/1 links held)" in message
    assert "hci6 (1/1 links held)" in message
    assert isinstance(excinfo.value, sys.modules["bleak"].exc.BleakError)
    assert RECORDED_INITS == []  # raised before any backend was constructed
    assert catcher._rotation.index(ADDRESS) == 0  # exhaustion is not failure


def test_claims_are_released_on_disconnect(env):
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())
    assert len(os.listdir(env.dir)) == 2
    asyncio.run(client.disconnect())
    assert os.listdir(env.dir) == []


def test_claims_are_released_when_the_link_drops_unexpectedly(env):
    """An unexpected drop must free the slot, via the wrapped raw
    disconnected_callback, before the caller's own callback runs."""
    seen = []
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, seen.append, _is_retry_client=True)

    asyncio.run(client.connect())
    assert len(os.listdir(env.dir)) == 2

    # bleak's backend records the drop, then fires the callback (via
    # functools.partial) - the release path keys on that ordering
    client._backend.is_connected = False
    raw_callback = RECORDED_INITS[-1]["disconnected_callback"]
    raw_callback(client)

    assert os.listdir(env.dir) == []
    assert seen == [client]


def test_claims_are_released_when_the_connect_attempt_fails(env):
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    CONNECT_RESULTS.append(RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        asyncio.run(client.connect())

    assert os.listdir(env.dir) == []
    assert catcher._rotation.index(ADDRESS) == 1  # a real failure DOES advance


def test_a_foreign_scan_claim_steers_selection_away(env):
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.scan")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci6"
    assert catcher._rotation.index(ADDRESS) == 0  # avoidance never moves the walk


def test_our_own_scan_claim_does_not_steer_selection_away(env):
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.scan", pid=os.getpid())
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci5"


def test_all_adapters_scan_claimed_never_gates(env):
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.scan")
    _foreign_file(env.dir, "hci6.scan")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci5"
    assert client.is_connected is True


def test_a_pinned_device_never_falls_back_to_the_pool(env, monkeypatch):
    """The pin's adapter is gone from the kernel: the empty presence filter
    falls back to the pinned list itself, never to the shared pool."""
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    env.install(adapters=(f"{ADDRESS}@hci9", "hci5"))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci9"


def test_selection_filters_against_present_adapters(env, monkeypatch):
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci6"})
    env.install(adapters=("hci5", "hci6"))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci6"


def test_an_empty_presence_filter_falls_back_to_the_configured_order(env, monkeypatch):
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci7"})
    env.install(adapters=("hci5", "hci6"))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci5"


def test_an_explicit_adapter_is_still_slot_gated_when_capped(env):
    """A link cap is physics, not coordination: an explicitly chosen full
    adapter raises the typed error - the connect is doomed and the error
    buys correct pacing. With the slot free, the connect holds it."""
    env.install(adapters=(), link_caps={"hci9": 1})
    blocker = _foreign_file(env.dir, "hci9.link.0")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True, bluez={"adapter": "hci9"})

    with pytest.raises(catcher.OutOfConnectionSlotsError) as excinfo:
        asyncio.run(client.connect())
    assert "hci9 (1/1 links held)" in str(excinfo.value)

    os.unlink(blocker)
    asyncio.run(client.connect())
    assert RECORDED_INITS[-1]["adapter"] == "hci9"
    assert "hci9.link.0" in os.listdir(env.dir)


# -- the service-cache surface ---------------------------------------------


def test_the_service_cache_surface_matches_bleak_retry_connector(env, caplog):
    """set_cached_services is a no-op; clear_cache delegates under a hasattr
    guard and warns + returns False when the underlying bleak (3.x) has no
    clear_cache - the one path establish_connection calls it on."""
    env.install()
    ClientClass = sys.modules["bleak_retry_connector"].BleakClientWithServiceCache
    assert ClientClass is catcher.BLEConnectionWithServiceCache
    client = ClientClass(ADDRESS, _is_retry_client=True)

    assert client.set_cached_services(["fff0"]) is None
    assert asyncio.run(client.clear_cache()) is False
    assert "clear_cache not implemented" in caplog.text


# -- phase 2: the adapter-bound, hard-claiming scanner ---------------------


def test_scanner_wrapping_is_opt_in(env):
    env.install()
    assert sys.modules["bleak"].BleakScanner is RichBleakScanner
    env.install(wrap_scanner=True)
    assert sys.modules["bleak"].BleakScanner is catcher.BLEScanner
    env.install()  # re-install without the flag reverts the scanner rebind
    assert sys.modules["bleak"].BleakScanner is RichBleakScanner
    env.install(wrap_scanner=True)
    catcher.uninstall_bleak_catcher()
    assert sys.modules["bleak"].BleakScanner is RichBleakScanner


def test_scanner_placeholders_are_inert(env):
    env.install(adapters=("hci5",), wrap_scanner=True)
    scanner = sys.modules["bleak"].BleakScanner()
    assert isinstance(scanner, catcher.BLEScanner)
    assert RECORDED_SCANNER_INITS == []  # nothing built before start
    assert scanner.discovered_devices == []


def test_a_scan_binds_an_adapter_and_holds_the_hard_claim_while_scanning(env):
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        assert "hci5.scan" in os.listdir(env.dir)
        assert scanner.discovered_devices == ["stub-device"]
        await scanner.stop()

    asyncio.run(scenario())
    # both spellings with a fresh bluez dict: older bleak backends read the
    # adapter kwarg, current bleak reads bluez["adapter"] - and passing the
    # kwarg alone trips current bleak's shared-mutable-default poison
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci5"
    assert RECORDED_SCANNER_INITS[-1]["bluez_adapter"] == "hci5"
    assert "hci5.scan" not in os.listdir(env.dir)  # held per scan activity


def test_the_scanner_context_manager_claims_and_releases(env):
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        async with sys.modules["bleak"].BleakScanner() as scanner:
            assert "hci5.scan" in os.listdir(env.dir)
            assert scanner._backend.scanning is True

    asyncio.run(scenario())
    assert "hci5.scan" not in os.listdir(env.dir)


def test_scan_placement_prefers_the_less_occupied_adapter(env):
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.use.other-svc")

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci6"


def test_a_scan_avoids_a_foreign_scan_claim(env):
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        assert "hci6.scan" in os.listdir(env.dir)
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci6"


def test_every_card_scan_claimed_scans_anyway_unclaimed(env):
    """Coordination never gates: with every card claimed by live foreign
    scanners, the scan proceeds on the best-ranked card without a claim."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")
    _foreign_file(env.dir, "hci6.scan")

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        assert scanner._backend.scanning is True
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci5"
    assert set(os.listdir(env.dir)) == {"hci5.scan", "hci6.scan"}  # both still foreign


def test_a_failed_scan_start_releases_the_claim(env):
    env.install(adapters=("hci5",), wrap_scanner=True)
    SCANNER_START_RESULTS.append(RuntimeError("org.bluez.Error.InProgress"))

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        with pytest.raises(RuntimeError):
            await scanner.start()

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []


def test_a_scanner_adapter_chosen_by_the_caller_is_never_overridden(env):
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner(adapter="hci9")
        await scanner.start()
        assert "hci9.scan" in os.listdir(env.dir)  # claim still taken, best effort
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci9"
    assert "hci9.scan" not in os.listdir(env.dir)


def test_a_scan_with_no_pool_uses_the_union_of_pinned_adapters(env):
    env.install(adapters=(f"{ADDRESS}@hci9",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci9"


def test_an_unconfigured_scanner_is_a_passthrough(env):
    env.install(adapters=(), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] is None
    assert RECORDED_SCANNER_INITS[-1]["extra"] == []
    assert os.listdir(env.dir) == []


# -- scored placement (habluetooth parity, cross-process) ------------------


def test_an_unconfigured_install_scores_over_every_present_adapter(env, monkeypatch):
    """Default is everything: with no pool configured, placement scores all
    adapters the kernel exposes; the pool config acts as an allowlist."""
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5", "hci6"})
    env.install(adapters=())
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci5"  # tie -> lowest hci number
    assert _soft_name("hci5") in os.listdir(env.dir)


def test_scoring_spreads_unpinned_devices_away_from_occupied_adapters(env):
    """The cross-process generalization of habluetooth's in-progress
    penalty: live soft claims from any process push new placements to the
    least-occupied card."""
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.use.other-svc.AABBCC")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci6"


def test_a_recovered_adapter_competes_again_after_a_success_elsewhere(env):
    """Failure penalties are per (adapter, address) and a success clears
    only that adapter's count - so the preferred card's single old failure
    keeps it demoted only until the counts even out."""
    env.install(adapters=("hci5", "hci6"))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    CONNECT_RESULTS.append(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        asyncio.run(client.connect())      # hci5 fails, penalized
    asyncio.run(client.connect())          # lands on hci6
    asyncio.run(client.disconnect())
    CONNECT_RESULTS.append(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        asyncio.run(client.connect())      # hci6 fails too, now tied
    asyncio.run(client.connect())

    assert [r["adapter"] for r in RECORDED_INITS] == ["hci5", "hci6", "hci6", "hci5"]


def test_hci_names_sort_numerically():
    assert sorted(["hci10", "hci2", "hci1"], key=catcher._hci_sort_key) == ["hci1", "hci2", "hci10"]


def test_an_unconfigured_scanner_scores_over_every_present_adapter(env, monkeypatch):
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci4"})
    env.install(adapters=(), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        assert "hci4.scan" in os.listdir(env.dir)
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci4"


# -- resolved BLEDevices: the device's own adapter is the truth ------------


def test_a_resolved_ble_devices_path_adapter_is_treated_as_explicit(env):
    """bleak's BlueZ backend connects via device.details["path"] and ignores
    the adapter argument for such devices - so claims, cap gating and tuning
    must land on the device's own adapter, and selection must not run."""
    env.install(adapters=("hci5",), link_caps={"hci9": 2})
    device = types.SimpleNamespace(address=ADDRESS, details={"path": "/org/bluez/hci9/dev_C8_47_8C_00_00_00"})
    client = sys.modules["bleak"].BleakClient(device, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] is None  # nothing injected to be ignored
    assert set(os.listdir(env.dir)) == {"hci9.link.0", _soft_name("hci9")}

    asyncio.run(client.disconnect())
    assert os.listdir(env.dir) == []


def test_a_resolved_ble_device_on_a_full_adapter_raises_the_typed_error(env):
    env.install(adapters=(), link_caps={"hci9": 1})
    _foreign_file(env.dir, "hci9.link.0")
    device = types.SimpleNamespace(address=ADDRESS, details={"path": "/org/bluez/hci9/dev_C8_47_8C_00_00_00"})
    client = sys.modules["bleak"].BleakClient(device, _is_retry_client=True)

    with pytest.raises(catcher.OutOfConnectionSlotsError) as excinfo:
        asyncio.run(client.connect())
    assert "hci9 (1/1 links held)" in str(excinfo.value)


# -- dead adapters: zero-MAC filtering and scan-failure memory -------------


def test_zero_mac_adapters_are_dropped_from_selection(env, monkeypatch):
    """A dead onboard controller stays listed in /sys/class/bluetooth with
    an all-zeros address forever; it must not win every tie."""
    monkeypatch.setattr(catcher.recovery, "adapter_mac", lambda a: catcher.recovery.UNKNOWN_MAC if a == "hci0" else "AA:BB:CC:DD:EE:FF")
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci0", "hci1"})
    env.install(adapters=(), wrap_scanner=True)

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    assert RECORDED_INITS[-1]["adapter"] == "hci1"

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        await scanner.stop()

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci1"


def test_every_adapter_zero_mac_never_gates(env, monkeypatch):
    monkeypatch.setattr(catcher.recovery, "adapter_mac", lambda a: catcher.recovery.UNKNOWN_MAC)
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci5"


def test_a_failed_scanner_start_steers_the_next_scan_elsewhere(env):
    """Scan selection has no failure-driven walk, so a dead-but-listed
    adapter needs start-failure memory or it wins every tie forever."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)

    async def scenario():
        SCANNER_START_RESULTS.append(RuntimeError("adapter 'hci5' not found"))
        first = sys.modules["bleak"].BleakScanner()
        with pytest.raises(RuntimeError):
            await first.start()
        second = sys.modules["bleak"].BleakScanner()
        await second.start()
        await second.stop()

    asyncio.run(scenario())
    assert [r["adapter"] for r in RECORDED_SCANNER_INITS] == ["hci5", "hci6"]


# -- scan_to_score: swept RSSI feeding the placement score -----------------


def test_swept_rssi_becomes_the_score_base():
    order = catcher._score_order(["hci5", "hci6"], "K", {}, {}, rssi={"hci5": -80, "hci6": -60})
    assert order == ["hci6", "hci5"]  # stronger signal wins


def test_rssi_penalties_are_charged_in_units_of_the_path_spread():
    """habluetooth's two-pass trick: penalties are multiples of how much
    better the best path is. One occupant (1.01 units) always overturns
    the RSSI advantage - that is the herd-spreading - while a failure
    (0.51 units) takes two to do it."""
    occupied = {"hci6": {"soft": 1, "links": 0}}
    assert catcher._score_order(["hci5", "hci6"], "K", occupied, {}, rssi={"hci5": -90, "hci6": -60}) == ["hci5", "hci6"]

    catcher._connect_failures[("hci6", "K")] = 1
    try:
        one_failure = catcher._score_order(["hci5", "hci6"], "K", {}, {}, rssi={"hci5": -61, "hci6": -60})
        assert one_failure == ["hci6", "hci5"]  # half a unit does not flip
        catcher._connect_failures[("hci6", "K")] = 2
        two_failures = catcher._score_order(["hci5", "hci6"], "K", {}, {}, rssi={"hci5": -61, "hci6": -60})
        assert two_failures == ["hci5", "hci6"]  # a full unit does
    finally:
        catcher._connect_failures.pop(("hci6", "K"), None)


def test_an_unseen_adapter_scores_no_rssi_value():
    order = catcher._score_order(["hci5", "hci6"], "K", {}, {}, rssi={"hci6": -90})
    assert order == ["hci6", "hci5"]  # -90 still beats never-seen (-127)


def test_sweeper_samples_go_stale(env, monkeypatch):
    env.install(adapters=("hci5",), scan_to_score=True)
    sweeper = catcher._config.sweeper
    sweeper.record("hci5", ADDRESS, -60)
    assert sweeper.rssi_for(["hci5"], catcher._address_key(ADDRESS)) == {"hci5": -60}
    monkeypatch.setattr(catcher, "_monotonic", lambda: time.monotonic() + catcher.RSSI_STALE_SECONDS + 1)
    assert sweeper.rssi_for(["hci5"], catcher._address_key(ADDRESS)) == {}


def test_a_sweep_takes_the_scan_claim_and_releases_it(env, monkeypatch):
    monkeypatch.setattr(catcher, "RSSI_SWEEP_DURATION", 0)
    env.install(adapters=("hci5",), scan_to_score=True)
    sweeper = catcher._config.sweeper

    async def scenario():
        await sweeper._sweep_adapter("hci5")

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS[-1]["adapter"] == "hci5"  # original scanner, bound
    assert "hci5.scan" not in os.listdir(env.dir)  # claim released after the window


def test_a_sweep_never_interrupts_a_foreign_scanner(env, monkeypatch):
    monkeypatch.setattr(catcher, "RSSI_SWEEP_DURATION", 0)
    env.install(adapters=("hci5",), scan_to_score=True)
    _foreign_file(env.dir, "hci5.scan")
    sweeper = catcher._config.sweeper

    async def scenario():
        await sweeper._sweep_adapter("hci5")

    asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS == []  # no scanner was built


def test_least_used_mode_runs_no_sweeper(env):
    env.install(adapters=("hci5",))
    assert catcher._config.sweeper is None

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    assert RECORDED_SCANNER_INITS == []  # nothing ever scans in this mode


def test_scan_to_score_starts_the_sweeper_on_first_connect(env):
    env.install(adapters=("hci5",), scan_to_score=True)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        sweeper = catcher._config.sweeper
        assert sweeper._task is not None and not sweeper._task.done()
        sweeper.stop()

    asyncio.run(scenario())


# -- connection-parameter tuning around connect ----------------------------


def test_conn_params_load_fast_before_connect_and_medium_after(env, monkeypatch):
    loads = []
    monkeypatch.setattr(catcher.mgmt, "load_fast", lambda adapter, address: loads.append(("fast", adapter)))
    monkeypatch.setattr(catcher.mgmt, "load_medium", lambda adapter, address: loads.append(("medium", adapter)))
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert loads == [("fast", "hci5"), ("medium", "hci5")]


def test_a_failed_connect_never_loads_the_medium_params(env, monkeypatch):
    loads = []
    monkeypatch.setattr(catcher.mgmt, "load_fast", lambda adapter, address: loads.append("fast"))
    monkeypatch.setattr(catcher.mgmt, "load_medium", lambda adapter, address: loads.append("medium"))
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    CONNECT_RESULTS.append(RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        asyncio.run(client.connect())

    assert loads == ["fast"]


def test_conn_param_tuning_can_be_disabled_and_skips_passthrough(env, monkeypatch):
    loads = []
    monkeypatch.setattr(catcher.mgmt, "load_fast", lambda adapter, address: loads.append("fast"))
    monkeypatch.setattr(catcher.mgmt, "load_medium", lambda adapter, address: loads.append("medium"))

    catcher.install_bleak_catcher(OWNER, adapters=("hci5",), claim_dir=env.dir, tune_conn_params=False)
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    assert loads == []  # disabled

    env.install(adapters=())  # unconfigured: no adapter known, nothing to tune
    other = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(other.connect())
    assert loads == []


# -- the scanner watchdog and claims-gated recovery ------------------------


def _adv(**kwargs):
    base = {"local_name": None, "manufacturer_data": {}, "service_data": {}, "service_uuids": []}
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def test_the_watchdog_clock_only_counts_nonempty_advertisements(env):
    """A wedged adapter can keep emitting empty advertisements; they must
    not read as signs of life (habluetooth's rule)."""
    seen = []
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner(seen.append and (lambda d, a: seen.append(a)))
        await scanner.start()
        stamp = RECORDED_SCANNER_INITS[-1]["detection_callback"]
        scanner._catcher_last_detection -= 50

        before = scanner._catcher_last_detection
        stamp("device", _adv())  # empty
        assert scanner._catcher_last_detection == before
        stamp("device", _adv(local_name="BMS"))
        assert scanner._catcher_last_detection > before
        assert len(seen) == 2  # the caller's callback saw both
        await scanner.stop()

    asyncio.run(scenario())


def test_a_quiet_scanner_restarts_without_a_hardware_reset(env, monkeypatch):
    resets = []

    async def fake_reset(adapter, claims_manager=None, force=False, gone_silent=False):
        resets.append(adapter)
        return True

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        now = catcher._monotonic()
        scanner._catcher_start_time = now - 300  # it did see things once
        scanner._catcher_last_detection = now - 100  # quiet 100s: restart tier
        await scanner._watchdog_restart()
        assert scanner._backend.scanning is True
        assert "hci5.scan" in os.listdir(env.dir)  # re-claimed after restart
        await scanner.stop()

    asyncio.run(scenario())
    assert len(RECORDED_SCANNER_INITS) == 2  # scanner was rebuilt
    assert resets == []  # 100s < the 120s escalation threshold


def test_a_scanner_quiet_past_escalation_hardware_resets_the_adapter(env, monkeypatch):
    resets = []

    async def fake_reset(adapter, claims_manager=None, force=False, gone_silent=False):
        resets.append((adapter, gone_silent, claims_manager is not None))
        return True

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        now = catcher._monotonic()
        scanner._catcher_start_time = now - 300
        scanner._catcher_last_detection = now - 200  # past the 120s threshold
        await scanner._watchdog_restart()
        await scanner.stop()

    asyncio.run(scenario())
    assert resets == [("hci5", True, True)]  # gone_silent, claims-gated


# -- claim-leak backstops: init failures and silent drops ------------------


def test_claims_are_released_when_the_real_client_init_fails(env):
    """An exception between claim acquisition and backend construction must
    not strand heartbeat-live claims."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    INIT_RESULTS.append(TypeError("unexpected keyword argument"))

    with pytest.raises(TypeError):
        asyncio.run(client.connect())

    assert os.listdir(env.dir) == []
    assert catcher._rotation.index(ADDRESS) == 0  # not a radio failure


def test_the_scan_claim_is_released_when_the_real_scanner_init_fails(env):
    env.install(adapters=("hci5",), wrap_scanner=True)
    SCANNER_INIT_RESULTS.append(TypeError("unexpected keyword argument"))

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        with pytest.raises(TypeError):
            await scanner.start()

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []


def test_a_cancelled_connect_releases_claims_without_advancing_the_walk(env):
    """bleak-retry-connector's timeout machinery cancels in-flight connects;
    CancelledError is a BaseException, so an except-Exception release path
    would strand the claims. Cancellation says nothing about the radio, so
    the walk index must not move either."""
    env.install(adapters=("hci5", "hci6"), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    CONNECT_RESULTS.append(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(client.connect())

    assert os.listdir(env.dir) == []
    assert catcher._rotation.index(ADDRESS) == 0
    assert client._backend is None  # no partially-initialised backend held


def test_a_cancelled_scan_start_releases_the_claim(env):
    env.install(adapters=("hci5",), wrap_scanner=True)
    SCANNER_START_RESULTS.append(asyncio.CancelledError())

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        with pytest.raises(asyncio.CancelledError):
            await scanner.start()
        assert scanner._backend is None

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []


def test_a_silent_drop_frees_claims_on_the_next_heartbeat(env):
    """A link that dies without the disconnected callback ever firing must
    not hold its slot and soft claim until process exit: the heartbeat's
    validity check releases them within one beat."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    assert len(os.listdir(env.dir)) == 2

    catcher._config.claims._beat_once()
    assert len(os.listdir(env.dir)) == 2  # still connected: the beat keeps them

    client._backend.is_connected = False  # dropped; no callback delivered
    catcher._config.claims._beat_once()
    assert os.listdir(env.dir) == []


# -- reconnect-path claim survival (field 2026-08-21: /run/bt-claims went
# -- empty while the LE link stayed up with data flowing) -------------------


def test_a_spurious_disconnected_callback_with_the_link_alive_keeps_the_claims(env):
    """A disconnect event that arrives while the wrapper's own view still
    says connected is not a drop: releasing on it zeroes the accounting for
    a link that is still up, and no new connect() ever re-claims. The caller
    still hears the event - its semantics are bleak's business - but the
    claims survive, and the heartbeat keeps them alive."""
    seen = []
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, seen.append, _is_retry_client=True)
    asyncio.run(client.connect())

    callback = RECORDED_INITS[-1]["disconnected_callback"]
    callback(client)  # backend still reports connected: spurious

    assert len(os.listdir(env.dir)) == 2
    assert seen == [client]
    catcher._config.claims._beat_once()
    assert len(os.listdir(env.dir)) == 2


def test_a_stale_callback_from_a_torn_down_backend_cannot_strip_a_reconnect(env):
    """The field mechanism: bleak-retry-connector retries connect() on one
    instance, and every generation's disconnected callback closes over the
    same wrapper. A late disconnect event from the torn-down previous
    backend must not release the claims a newer connect() acquired - not
    after it succeeded, and not while its attempt is still in flight."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    stale = RECORDED_INITS[0]["disconnected_callback"]

    client._backend.is_connected = False
    stale(client)  # the real drop: its own generation, link down - releases
    assert os.listdir(env.dir) == []

    asyncio.run(client.connect())  # the reconnect re-claims
    assert len(os.listdir(env.dir)) == 2
    stale(client)  # late duplicate event; the link is up
    assert len(os.listdir(env.dir)) == 2

    client._backend.is_connected = False  # a third attempt mid-flight
    stale(client)  # stale generation: these claims were never its to free
    assert len(os.listdir(env.dir)) == 2


def test_claims_survive_the_wrapper_being_collected_while_the_link_lives(env):
    """Validity must track the link, not the wrapper object: when the
    wrapper is dropped while the backend keeps the BlueZ link, the claims
    stay - and release only once the backend itself says disconnected."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        return client._backend

    backend = asyncio.run(scenario())  # the wrapper goes out of scope
    # the stub's recorded disconnected callback closes over the wrapper;
    # drop the harness's own reference so only production references remain
    RECORDED_INITS.clear()
    gc.collect()

    catcher._config.claims._beat_once()
    assert len(os.listdir(env.dir)) == 2  # link truth: still connected

    backend.is_connected = False
    catcher._config.claims._beat_once()
    assert os.listdir(env.dir) == []


def test_notification_traffic_re_arms_claims_lost_while_the_link_lives(env):
    """The reconnect-path field bug end to end: a disconnect event fires
    with the wrapper's view already broken (is_connected False) while the
    BlueZ link survives - claims are released and nothing re-claims. Data
    still flowing through the notification path is proof of a live link,
    so the tap re-acquires the slot and soft claim, and the evidence keeps
    the heartbeat from sweeping them again."""
    received = []
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", lambda sender, data: received.append(data))
        return client

    client = asyncio.run(scenario())
    tap = client._backend.notify_callbacks["fff4"]

    client._backend.is_connected = False  # bleak's view breaks; link is up
    RECORDED_INITS[-1]["disconnected_callback"](client)
    assert os.listdir(env.dir) == []  # the field state: empty claim dir

    tap("fff4", b"\x01")  # a notification arrives: the link is alive
    assert set(os.listdir(env.dir)) == {"hci5.link.0", _soft_name("hci5")}
    assert received == [b"\x01"]  # the caller's callback still ran

    catcher._config.claims._beat_once()  # evidence outvotes is_connected
    assert len(os.listdir(env.dir)) == 2


def test_stale_link_evidence_stops_protecting_a_disconnected_client(env, monkeypatch):
    """Evidence is a grace, not immortality: once the data stops and the
    wrapper still reads disconnected, the heartbeat releases as before."""
    received = []
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", lambda sender, data: received.append(data))
        return client

    client = asyncio.run(scenario())
    client._backend.notify_callbacks["fff4"]("fff4", b"\x01")
    client._backend.is_connected = False

    catcher._config.claims._beat_once()
    assert len(os.listdir(env.dir)) == 2  # fresh evidence holds the claims

    monkeypatch.setattr(catcher, "_monotonic", lambda: time.monotonic() + catcher.LINK_EVIDENCE_SECONDS + 1)
    catcher._config.claims._beat_once()
    assert os.listdir(env.dir) == []  # silence + disconnected: swept


def test_a_stray_notification_after_disconnect_does_not_re_arm(env):
    """An intentional disconnect is the end of the accounting: a straggler
    notification racing the teardown must not resurrect the claims."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", lambda sender, data: None)
        await client.disconnect()
        return client

    client = asyncio.run(scenario())
    assert os.listdir(env.dir) == []

    client._backend.notify_callbacks["fff4"]("fff4", b"\x01")
    assert os.listdir(env.dir) == []


def test_the_notification_tap_preserves_async_callbacks(env):
    """bleak decides sync-vs-async handling by inspecting the callback it
    receives; the tap must mirror the caller's coroutine-ness or async
    consumer callbacks would silently never run."""
    received = []
    env.install(adapters=("hci5",))

    async def consumer(sender, data):
        received.append(data)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", consumer)
        tap = client._backend.notify_callbacks["fff4"]
        assert asyncio.iscoroutinefunction(tap)
        await tap("fff4", b"\x02")

    asyncio.run(scenario())
    assert received == [b"\x02"]


def test_a_polled_read_re_arms_claims_lost_while_the_link_lives(env):
    """The prod field bug (2026-08-22, dbus-easytouchrv): the same lost-claim
    state as the notification case, but on a consumer that never subscribes
    to anything - it only polls read_gatt_char. Notification traffic can
    never re-arm such a link, so the completed read has to."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        client._backend.is_connected = False  # bleak's view breaks; link is up
        RECORDED_INITS[-1]["disconnected_callback"](client)
        assert os.listdir(env.dir) == []  # the field state: empty claim dir
        return client, await client.read_gatt_char("fff1")

    client, value = asyncio.run(scenario())
    assert set(os.listdir(env.dir)) == {"hci5.link.0", _soft_name("hci5")}
    assert value == b"\x00"  # the caller still gets its data

    catcher._config.claims._beat_once()  # evidence outvotes is_connected
    assert len(os.listdir(env.dir)) == 2


def test_a_polled_write_is_link_evidence_too(env):
    """Write-only consumers exist (a relay driver that never reads back);
    a completed write is the same proof of life as a read."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        client._backend.is_connected = False
        RECORDED_INITS[-1]["disconnected_callback"](client)
        assert os.listdir(env.dir) == []
        await client.write_gatt_char("fff2", b"\x01", True)
        return client

    asyncio.run(scenario())
    assert set(os.listdir(env.dir)) == {"hci5.link.0", _soft_name("hci5")}


def test_a_failed_gatt_read_is_not_proof_of_life(env):
    """Evidence is noted after the await, never before: a read that raised
    says the link is gone, not that it is alive, and must not re-arm."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    GATT_RESULTS.append(RuntimeError("Not connected"))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        client._backend.is_connected = False
        RECORDED_INITS[-1]["disconnected_callback"](client)
        with pytest.raises(RuntimeError):
            await client.read_gatt_char("fff1")
        return client

    client = asyncio.run(scenario())
    assert os.listdir(env.dir) == []
    assert not client._recent_link_evidence()


def test_a_polled_read_holds_claims_the_heartbeat_would_otherwise_sweep(env, monkeypatch):
    """The other half of the fix: for a polling consumer the read is also
    what keeps the validity check from sweeping live claims in the first
    place - and once the polling stops, the sweep resumes as before."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.read_gatt_char("fff1")
        client._backend.is_connected = False  # broken view, link still up
        return client

    client = asyncio.run(scenario())
    catcher._config.claims._beat_once()
    assert len(os.listdir(env.dir)) == 2  # fresh poll holds the claims

    monkeypatch.setattr(catcher, "_monotonic", lambda: time.monotonic() + catcher.LINK_EVIDENCE_SECONDS + 1)
    catcher._config.claims._beat_once()
    assert os.listdir(env.dir) == []  # polling stopped and still disconnected


def test_a_polled_read_after_disconnect_does_not_re_arm(env):
    """An intentional teardown settles the accounting for the polled path
    exactly as it does for notifications: a late read racing disconnect()
    must not resurrect the claims."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.disconnect()
        assert os.listdir(env.dir) == []
        await client.read_gatt_char("fff1")

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []


def test_an_abandoned_running_scanners_claim_frees_on_the_heartbeat(env):
    """A started scanner that is garbage collected without stop() must not
    mark the card as scanning forever."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        assert "hci5.scan" in os.listdir(env.dir)
        del scanner

    asyncio.run(scenario())
    # the stub's recorded detection callback closes over the wrapper; drop
    # the test harness's own reference so only production references remain
    RECORDED_SCANNER_INITS.clear()
    gc.collect()
    catcher._config.claims._beat_once()
    assert "hci5.scan" not in os.listdir(env.dir)


# -- post-connect validation ----------------------------------------------


def _validator(result, seen=None):
    """A validator that records the client it saw and returns/raises `result`."""

    async def _validate(client):
        if seen is not None:
            seen.append(client)
        if isinstance(result, BaseException):
            raise result
        return result

    return _validate


def test_a_rejected_link_is_torn_down_and_counted_as_a_failed_connect(env):
    """v1's contract: validation failing IS a connection failure - the link
    goes down, the claims go with it, and the adapter takes the blame."""
    env.install(adapters=("hci5", "hci6"), link_caps={"hci5": 2})
    backends = []
    client = sys.modules["bleak"].BleakClient(
        ADDRESS,
        _is_retry_client=True,
        validate_connection=_validator(False, backends),
    )

    with pytest.raises(catcher.ConnectionValidationError):
        asyncio.run(client.connect())

    assert backends and backends[0] is client  # the validator sees the client
    assert client.is_connected is False
    assert os.listdir(env.dir) == []  # slot and soft claim both released
    assert catcher._rotation.index(ADDRESS) == 1

    with pytest.raises(catcher.ConnectionValidationError):
        asyncio.run(client.connect())  # the retry above us walks on
    assert [r["adapter"] for r in RECORDED_INITS] == ["hci5", "hci6"]


def test_the_underlying_link_is_disconnected_before_the_failure_is_raised(env):
    """A rejected link left up is the phantom the caller was validating
    against; BlueZ would hold it until something else cleared it."""
    env.install(adapters=("hci5",))
    backends = []

    async def _validate(client):
        backends.append(client._backend)
        assert client._backend.is_connected is True
        return False

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True, validate_connection=_validate)

    with pytest.raises(catcher.ConnectionValidationError):
        asyncio.run(client.connect())

    assert backends[0].is_connected is False
    assert client._backend is None


def test_a_validator_that_raises_counts_as_a_rejection(env):
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(
        ADDRESS,
        _is_retry_client=True,
        validate_connection=_validator(RuntimeError("read failed")),
    )

    with pytest.raises(catcher.ConnectionValidationError):
        asyncio.run(client.connect())

    assert os.listdir(env.dir) == []


def test_a_passing_validator_leaves_the_connection_and_its_claims_alone(env):
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(
        ADDRESS,
        _is_retry_client=True,
        validate_connection=_validator(True),
    )

    asyncio.run(client.connect())

    assert client.is_connected is True
    assert set(os.listdir(env.dir)) == {"hci5.link.0", _soft_name("hci5")}
    assert catcher._rotation.index(ADDRESS) == 0


def test_the_validator_kwarg_never_reaches_the_real_client_init(env):
    """It rides in on establish_connection's kwargs passthrough; bleak's own
    __init__ would reject it."""
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(
        ADDRESS,
        _is_retry_client=True,
        validate_connection=_validator(True),
    )

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["extra"] == []


def test_a_validator_never_runs_when_the_connect_itself_failed(env):
    env.install(adapters=("hci5",))
    seen = []
    client = sys.modules["bleak"].BleakClient(
        ADDRESS,
        _is_retry_client=True,
        validate_connection=_validator(True, seen),
    )
    CONNECT_RESULTS.append(RuntimeError("le-connection-abort-by-local"))

    with pytest.raises(RuntimeError):
        asyncio.run(client.connect())

    assert seen == []


def test_the_installed_validator_applies_to_clients_that_carry_none(env):
    """The point of a process-wide validator: connections made deep inside a
    library the driver never calls directly still get validated."""
    env.install(adapters=("hci5",), validate_connection=_validator(False))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    with pytest.raises(catcher.ConnectionValidationError):
        asyncio.run(client.connect())


def test_a_client_validator_overrides_the_installed_one(env):
    env.install(adapters=("hci5",), validate_connection=_validator(False))
    client = sys.modules["bleak"].BleakClient(
        ADDRESS,
        _is_retry_client=True,
        validate_connection=_validator(True),
    )

    asyncio.run(client.connect())

    assert client.is_connected is True


def test_a_per_call_validator_overrides_both(env):
    env.install(adapters=("hci5",), validate_connection=_validator(True))
    client = sys.modules["bleak"].BleakClient(
        ADDRESS,
        _is_retry_client=True,
        validate_connection=_validator(True),
    )

    with pytest.raises(catcher.ConnectionValidationError):
        asyncio.run(client.connect(validate_connection=_validator(False)))


def test_an_unvalidated_install_connects_exactly_as_before(env):
    env.install(adapters=("hci5",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert client.is_connected is True


# -- drain (convention 0.3) ------------------------------------------------


def test_a_live_drain_steers_connects_away(env):
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.drain")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci6"


def test_every_adapter_draining_never_gates_a_connect(env):
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.drain")
    _foreign_file(env.dir, "hci6.drain")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci5"  # steers, never refuses


def test_a_live_drain_steers_scans_away(env):
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.drain")

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        adapter = scanner._catcher_adapter
        await scanner.stop()
        return adapter

    assert asyncio.run(scenario()) == "hci6"


def test_the_drain_watcher_migrates_a_movable_client(env):
    """The cooperative half of a coordinated reset: a foreign drain appears
    on the card this client is connected on, another card exists, so the
    watcher disconnects it - releasing its claims for the resetter - and
    the driver's retry loop above is what reconnects it elsewhere."""
    env.install(adapters=("hci5", "hci6"), link_caps={"hci5": 2, "hci6": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        adapter = client._catcher_adapter_used
        _foreign_file(env.dir, f"{adapter}.drain")
        catcher._drain_watch()  # what the heartbeat thread runs
        for _ in range(3):
            await asyncio.sleep(0)
        return client, adapter

    client, adapter = asyncio.run(scenario())
    assert client.is_connected is False
    assert client._catcher_drain_kicked == adapter
    remaining = [n for n in os.listdir(env.dir) if not n.endswith(".drain")]
    assert remaining == []  # claims released for the resetter


def test_the_drain_watcher_leaves_a_client_with_nowhere_to_go(env):
    """"If possible" is literal: on a one-card deployment the client stays,
    and its live claims keep vetoing the reset."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        _foreign_file(env.dir, "hci5.drain")
        catcher._drain_watch()
        for _ in range(3):
            await asyncio.sleep(0)
        return client

    client = asyncio.run(scenario())
    assert client.is_connected is True
    assert client._catcher_drain_kicked is None
    assert any(n.startswith("hci5.link") for n in os.listdir(env.dir))


def test_the_drain_watcher_respects_an_explicit_adapter(env):
    env.install(adapters=("hci5", "hci6"))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True, adapter="hci5")
        await client.connect()
        _foreign_file(env.dir, "hci5.drain")
        catcher._drain_watch()
        for _ in range(3):
            await asyncio.sleep(0)
        return client

    client = asyncio.run(scenario())
    assert client.is_connected is True  # the caller chose; we do not move it


def test_the_drain_watcher_kicks_once_per_adapter_per_connect(env):
    """A migration that lands back on the draining card (nothing else
    worked) must not be bounced forever."""
    env.install(adapters=("hci5", "hci6"))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        adapter = client._catcher_adapter_used
        _foreign_file(env.dir, f"{adapter}.drain")
        catcher._drain_watch()
        for _ in range(3):
            await asyncio.sleep(0)
        # simulate the retry loop reconnecting onto the SAME (only working)
        # card: is_connected again, kicked marker preserved by connect? No -
        # connect resets it, which is correct: a fresh connect is fresh
        # consent. Here we re-mark manually to test the guard itself.
        client._backend.is_connected = True
        client._catcher_settled = False
        client._catcher_drain_kicked = adapter
        catcher._drain_watch()
        for _ in range(3):
            await asyncio.sleep(0)
        return client

    client = asyncio.run(scenario())
    assert client.is_connected is True  # second watch did not kick again


def test_the_drain_watcher_moves_a_running_scanner(env):
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        first = scanner._catcher_adapter
        _foreign_file(env.dir, f"{first}.drain")
        catcher.present_adapters = lambda: {"hci5", "hci6"}
        try:
            catcher._drain_watch()
            for _ in range(6):
                await asyncio.sleep(0)
        finally:
            catcher.present_adapters = lambda: set()
        moved = scanner._catcher_adapter
        await scanner.stop()
        return first, moved

    first, moved = asyncio.run(scenario())
    assert first == "hci5"
    assert moved == "hci6"  # restarted off the draining card


def test_a_duplicate_claimant_from_another_pid_is_flagged(env, caplog):
    """The orphaned-process signature (prod 2026-08-22): a second live
    instance of the same service claiming the same device produced a 45
    minute connect/disconnect flap that read as radio failure. The claim
    files name the culprit; the catcher should say so."""
    import logging as _logging

    env.install(adapters=("hci5",))
    mac = ADDRESS.replace(":", "")
    _foreign_file(env.dir, f"hci5.use.{OWNER}-99999.{mac}")  # orphan: same service, other pid

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    with caplog.at_level(_logging.WARNING):
        asyncio.run(client.connect())

    assert any("another live instance of this service" in r.message for r in caplog.records)


def test_an_unrelated_services_claim_is_not_flagged(env, caplog):
    import logging as _logging

    env.install(adapters=("hci5",))
    mac = ADDRESS.replace(":", "")
    _foreign_file(env.dir, f"hci5.use.other-svc-99999.{mac}")  # different service, same device

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    with caplog.at_level(_logging.WARNING):
        asyncio.run(client.connect())

    assert not any("another live instance" in r.message for r in caplog.records)


# -- adapter identity by MAC (convention 0.4) ------------------------------


def _kernel_adapters(monkeypatch, mapping):
    """Present hciN -> MAC, for both the catcher and the claims layer."""
    monkeypatch.setattr(catcher.claims, "_mac_cache", {})
    monkeypatch.setattr(catcher.claims, "present_hci_names", lambda: sorted(mapping))
    monkeypatch.setattr(catcher.claims, "_read_adapter_mac", lambda a: mapping.get(a, catcher.claims.UNKNOWN_MAC))
    monkeypatch.setattr(catcher, "present_adapters", lambda: set(mapping))
    catcher._observed_identities.clear()


def test_an_adapter_configured_by_mac_routes_to_its_current_number(env, monkeypatch):
    """The point of the change: config names the card, and the hciN it is
    handed to bleak is looked up at connect time."""
    env.install(adapters=("AA:BB:CC:DD:EE:FF",))
    _kernel_adapters(monkeypatch, {"hci7": "AA:BB:CC:DD:EE:FF"})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci7"


def test_a_renumbered_card_follows_its_mac(env, monkeypatch):
    """Same config, same card, different number after a USB reset - the
    connect follows the card rather than the stale number."""
    env.install(adapters=("aabbccddeeff",))  # a third spelling, no colons
    _kernel_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    assert RECORDED_INITS[-1]["adapter"] == "hci3"

    _kernel_adapters(monkeypatch, {"hci9": "AA:BB:CC:DD:EE:FF"})
    client2 = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client2.connect())
    assert RECORDED_INITS[-1]["adapter"] == "hci9"


def test_a_mac_that_no_present_card_answers_to_is_skipped(env, monkeypatch):
    """An unplugged card is simply absent - selection falls through to the
    adapters that are there rather than handing bleak a name it cannot use."""
    env.install(adapters=("AA:BB:CC:DD:EE:FF", "hci5"))
    _kernel_adapters(monkeypatch, {"hci5": "11:22:33:44:55:66"})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci5"


def test_link_caps_may_be_keyed_by_mac(env, monkeypatch):
    """A cap follows the card too: keyed by MAC, honored on whatever number
    the card currently has."""
    env.install(adapters=("hci4",), link_caps={"AA:BB:CC:DD:EE:FF": 1})
    _kernel_adapters(monkeypatch, {"hci4": "AA:BB:CC:DD:EE:FF"})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    asyncio.run(client.connect())

    assert any(n.startswith("AABBCCDDEEFF.link.") for n in os.listdir(env.dir))


def test_an_hci_entry_is_rewritten_to_its_mac_in_the_config(env, monkeypatch, tmp_path):
    """First successful read of the card rewrites the config entry to the
    MAC it proved to be, with a comment recording the substitution."""
    conf = tmp_path / "driver.conf"
    conf.write_text(
        "# BLE settings\n"
        "adapters = hci4,hci5\n"
        "[caps]\n"
        "hci4 = 5\n"
        "# hci4 in a comment is left alone\n"
    )
    env.install(adapters=("hci4",), adapter_config_path=str(conf))
    _kernel_adapters(monkeypatch, {"hci4": "AA:BB:CC:DD:EE:FF", "hci5": "11:22:33:44:55:66"})

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())

    text = conf.read_text()
    assert "# bcm: hci4 was detected as AA:BB:CC:DD:EE:FF and rewritten" in text
    assert "adapters = AA:BB:CC:DD:EE:FF,hci5" in text  # hci5 untouched: not read yet
    assert "AA:BB:CC:DD:EE:FF = 5" in text  # the cap key follows the card
    assert "# hci4 in a comment is left alone" in text


def test_the_config_rewrite_respects_token_boundaries(tmp_path):
    """hci1 must never match inside hci10."""
    conf = tmp_path / "c.conf"
    conf.write_text("adapters = hci1,hci10\n")

    assert catcher.rewrite_adapter_config(str(conf), {"hci1": "AA:BB:CC:DD:EE:FF"}) is True

    assert "adapters = AA:BB:CC:DD:EE:FF,hci10" in conf.read_text()


def test_the_config_rewrite_is_best_effort(tmp_path):
    """An unreadable or unwritable config is never worth breaking a
    connection over."""
    assert catcher.rewrite_adapter_config(str(tmp_path / "nope.conf"), {"hci1": "AA:BB:CC:DD:EE:FF"}) is False
    conf = tmp_path / "c.conf"
    conf.write_text("adapters = hci9\n")
    assert catcher.rewrite_adapter_config(str(conf), {"hci1": "AA:BB:CC:DD:EE:FF"}) is False  # no hit


def test_failure_memory_follows_the_card_not_the_number(env, monkeypatch):
    """The same hazard as claim keys, in the placement score: after a
    renumber, penalties keyed by hciN would follow the NUMBER - a healthy
    card inheriting a bad one's number would inherit its record."""
    env.install(adapters=("hci3",))
    _kernel_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    catcher._scan_finished("hci3", False)
    catcher._connect_finished("hci3", ADDRESS, False)

    assert catcher._scan_failures.get("AABBCCDDEEFF") == 1
    # the card comes back as hci7 after a reset; its record follows it
    _kernel_adapters(monkeypatch, {"hci7": "AA:BB:CC:DD:EE:FF"})
    assert catcher._scan_failures.get(catcher.claims.adapter_key("hci7")) == 1
    # and a DIFFERENT card that lands on the old number inherits nothing
    _kernel_adapters(monkeypatch, {"hci3": "11:22:33:44:55:66"})
    assert catcher._scan_failures.get(catcher.claims.adapter_key("hci3")) is None


def test_a_successful_reset_forgets_the_cards_failure_record(env, monkeypatch):
    """A power-cycled card should not keep paying for what the old one did -
    and for scans the penalty is self-reinforcing: ranked last, so never
    selected to earn the success that would clear it."""
    env.install(adapters=("hci3",))
    _kernel_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    catcher._scan_finished("hci3", False)
    catcher._connect_finished("hci3", ADDRESS, False)
    assert catcher._scan_failures and catcher._connect_failures

    catcher.forget_adapter_failures("hci3")

    assert catcher._scan_failures == {}
    assert catcher._connect_failures == {}


def _ble_device(address, path):
    """A cache-resolved BLEDevice: its BlueZ path names its adapter, which
    is what sends a connect down the explicit path."""
    return types.SimpleNamespace(address=address, details={"path": path})


# -- identity boundary: claims() keys by MAC, the catcher works in hciN ----
#
# Every lookup below crosses that boundary. A raw .get returns nothing on
# any host where MACs actually resolve - which is every real deployment and
# was none of the tests, which is how the whole class of bug survived a
# green suite (field 2026-08-22, both Cerbos, reported by a consumer).


def test_the_explicit_path_honors_mac_keyed_caps(env, monkeypatch):
    """_claim_explicit is the DOMINANT path - most real connects arrive as
    cache-resolved BLEDevices carrying a device path - and its cap lookup
    missed entirely when caps were keyed by MAC, so link gating silently
    stopped existing and OutOfConnectionSlotsError could never fire."""
    env.install(adapters=("hci4",), link_caps={"00:1A:7D:DA:71:06": 1})
    _kernel_adapters(monkeypatch, {"hci4": "00:1A:7D:DA:71:06"})
    device = _ble_device(ADDRESS, "/org/bluez/hci4/dev_C8_47_8C_00_00_00")

    client = sys.modules["bleak"].BleakClient(device, _is_retry_client=True)
    asyncio.run(client.connect())

    assert any(n.startswith("001A7DDA7106.link.") for n in os.listdir(env.dir))


def test_the_explicit_path_raises_out_of_slots_with_mac_keyed_caps(env, monkeypatch):
    """The other half: with the cap invisible, the typed error that buys
    bleak-retry-connector its out-of-slots pacing could never be raised."""
    env.install(adapters=("hci4",), link_caps={"00:1A:7D:DA:71:06": 1})
    _kernel_adapters(monkeypatch, {"hci4": "00:1A:7D:DA:71:06"})
    _foreign_file(env.dir, "001A7DDA7106.link.0")  # the card's only slot, taken
    device = _ble_device(ADDRESS, "/org/bluez/hci4/dev_C8_47_8C_00_00_00")

    client = sys.modules["bleak"].BleakClient(device, _is_retry_client=True)
    with pytest.raises(catcher.OutOfConnectionSlotsError):
        asyncio.run(client.connect())


def test_occupancy_scoring_sees_mac_keyed_claims(env, monkeypatch):
    """Placement spreads load by counting other processes' claims. Against
    a MAC-keyed snapshot an hciN lookup sees zero occupancy everywhere, so
    least-used placement silently stops spreading anything."""
    env.install(adapters=("hci3", "hci4"))
    _kernel_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF", "hci4": "11:22:33:44:55:66"})
    for k in range(3):
        _foreign_file(env.dir, f"AABBCCDDEEFF.use.other-svc-{k}")  # hci3 is busy

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci4"  # steered to the idle card


def test_a_foreign_scan_claim_steers_away_when_keyed_by_mac(env, monkeypatch):
    env.install(adapters=("hci3", "hci4"))
    _kernel_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF", "hci4": "11:22:33:44:55:66"})
    _foreign_file(env.dir, "AABBCCDDEEFF.scan")

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci4"


def test_drain_steering_survives_mac_keyed_claims(env, monkeypatch):
    env.install(adapters=("hci3", "hci4"))
    _kernel_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF", "hci4": "11:22:33:44:55:66"})
    _foreign_file(env.dir, "AABBCCDDEEFF.drain")

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci4"


def test_a_disconnect_event_during_the_handshake_keeps_the_claims(env):
    """Field 2026-08-23 (prod shyion, 4 of 28 transient sessions): a
    disconnected callback fires mid-handshake, before the link has ever
    been up. Both earlier guards pass - it IS the current generation, and
    is_connected IS False because the link is not up YET - so the claims
    connect() had just taken were released a second before first traffic
    re-armed them. Between the two, the numbered link slot reads free
    while a connect is in flight on it, so a capped adapter can be
    OVER-subscribed, not merely under-counted."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    fired = []

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        real_connect = client._backend.connect if client._backend else None

        async def connect_firing_disconnect(pair, **kwargs):
            # the event arrives while the handshake is still running
            RECORDED_INITS[-1]["disconnected_callback"](client)
            fired.append(True)
            client._backend.is_connected = True

        original_init = RichBleakClient.__init__

        def patched_init(self, *a, **k):
            original_init(self, *a, **k)
            self._backend.connect = connect_firing_disconnect

        RichBleakClient.__init__ = patched_init
        try:
            await client.connect()
        finally:
            RichBleakClient.__init__ = original_init
        return client

    client = asyncio.run(scenario())
    assert fired == [True]  # the event really did fire mid-handshake
    assert client.is_connected is True
    held = sorted(os.listdir(env.dir))
    assert "hci5.link.0" in held  # the slot was never surrendered
    assert _soft_name("hci5") in held


def test_a_failed_connect_still_releases_after_a_handshake_event(env):
    """The in-flight guard must not leak claims when the connect then
    fails - connect()'s own finally is what releases there."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    CONNECT_RESULTS.append(RuntimeError("le-connection-abort-by-local"))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        with pytest.raises(RuntimeError):
            await client.connect()
        RECORDED_INITS[-1]["disconnected_callback"](client)  # late straggler
        return client

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []


# -- the evidence window adapts to the consumer's cadence -----------------


def _notify_at(client, tap, monkeypatch, times):
    """Drive notifications at given monotonic offsets."""
    base = time.monotonic()
    for offset in times:
        monkeypatch.setattr(catcher, "_monotonic", lambda o=offset: base + o)
        tap("fff4", b"\x01")
    return base


def test_a_thirty_second_cadence_is_not_a_coin_flip(env, monkeypatch):
    """Field 2026-08-23: power-watchdog's device notifies about every 30s
    against a 30s floor - the same number, so any jitter expired the
    evidence before the traffic that would refresh it, while the driver
    itself treats silence up to 120s as healthy."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", lambda s, d: None)
        return client

    client = asyncio.run(scenario())
    tap = client._backend.notify_callbacks["fff4"]
    base = _notify_at(client, tap, monkeypatch, [0, 30, 60, 90])

    client._backend.is_connected = False  # the stranded property this exists for
    # 45s of silence: late for a 30s cadence, but far inside the driver's
    # own 120s liveness tolerance. The old fixed floor swept here.
    monkeypatch.setattr(catcher, "_monotonic", lambda: base + 90 + 45)
    catcher._config.claims._beat_once()

    assert len(os.listdir(env.dir)) == 2  # claims held


def test_a_dead_link_still_frees_within_the_adapted_window(env, monkeypatch):
    """Adapting is a grace, not immortality: past the window the sweep
    releases exactly as before."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", lambda s, d: None)
        return client

    client = asyncio.run(scenario())
    tap = client._backend.notify_callbacks["fff4"]
    base = _notify_at(client, tap, monkeypatch, [0, 30, 60])

    client._backend.is_connected = False
    monkeypatch.setattr(catcher, "_monotonic", lambda: base + 60 + catcher.LINK_EVIDENCE_MAX + 1)
    catcher._config.claims._beat_once()

    assert os.listdir(env.dir) == []


def test_a_fast_consumer_keeps_the_floor(env, monkeypatch):
    """A 10s poller must not have its window shrunk below the floor - the
    adaptation only ever widens."""
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", lambda s, d: None)
        return client

    client = asyncio.run(scenario())
    tap = client._backend.notify_callbacks["fff4"]
    _notify_at(client, tap, monkeypatch, [0, 2, 4, 6])

    assert client._evidence_window() == catcher.LINK_EVIDENCE_SECONDS


def test_the_window_is_capped(env, monkeypatch):
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        await client.start_notify("fff4", lambda s, d: None)
        return client

    client = asyncio.run(scenario())
    tap = client._backend.notify_callbacks["fff4"]
    _notify_at(client, tap, monkeypatch, [0, 3600, 7200])

    assert client._evidence_window() == catcher.LINK_EVIDENCE_MAX


def test_a_reconnect_retires_the_previous_backend(env):
    """Field 2026-08-24, prod: bleak's BlueZ client owns a private system-bus
    connection per session and closes it ONLY in disconnect() - the
    _cleanup_all() that runs when a link drops leaves it attached. This
    wrapper re-runs the real __init__ on every connect, so a reconnect that
    did not pass through disconnect() orphaned the previous backend WITH its
    bus: one leaked connection per retry, against a per-user ceiling of 256.
    One driver retrying an unreachable device reached 148 and saturated the
    system bus for every root process on the box."""
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        first = client._backend
        assert first.disconnects == 0
        # the link drops on its own - no disconnect() call anywhere - and the
        # consumer's retry loop simply connects again on the same client
        first.is_connected = False
        await client.connect()
        return first, client._backend

    first, second = asyncio.run(scenario())
    assert first is not second           # a new backend, as the design intends
    assert first.disconnects == 1        # and the old one was closed, not orphaned


def test_retiring_a_wedged_backend_does_not_block_the_reconnect(env, monkeypatch):
    """A predecessor that will not close must never cost the caller the
    reconnect it actually asked for."""
    env.install(adapters=("hci5",))
    monkeypatch.setattr(catcher, "BACKEND_RETIRE_TIMEOUT", 0.05)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()

        async def never_returns():
            await asyncio.sleep(3600)

        client._backend.disconnect = never_returns
        await client.connect()          # must still complete
        return client

    client = asyncio.run(scenario())
    assert client.is_connected is True


def test_a_failed_connect_does_not_reach_the_retire_path(env):
    """bleak closes the bus itself on a failed connect (only the success path
    calls stack.pop_all()), and our finally clears the backend - so there is
    nothing to retire and no double disconnect."""
    env.install(adapters=("hci5",))
    CONNECT_RESULTS.append(RuntimeError("boom"))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        with pytest.raises(RuntimeError):
            await client.connect()
        assert client._backend is None
        await client.connect()
        return client

    client = asyncio.run(scenario())
    assert client.is_connected is True


def test_a_raising_disconnect_does_not_strand_its_dbus_connection(env):
    """Field 2026-08-24, fleet-wide: bleak closes the session bus in three
    statements AFTER its try/finally, so a Disconnect that answers
    "Not connected" - the ordinary case when the peer is already gone -
    skips them and strands the connection. _cleanup_all never recovers it.
    One leaked connection per failed disconnect, against a ceiling of 256."""
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        bus = client._backend._bus
        client._backend.disconnect_error = RuntimeError("[org.bluez.Error.Failed] Not connected")
        with pytest.raises(RuntimeError):
            await client.disconnect()      # the error still reaches the caller
        return bus, client._backend

    bus, backend = asyncio.run(scenario())
    assert bus.connected is False          # closed rather than stranded
    assert backend._bus is None            # and bleak's own guards see it closed


def test_a_raising_disconnect_during_retirement_is_also_closed(env):
    """The same defect on the reconnect path, where the exception is
    swallowed - tidying the reference away without closing the bus would
    leak it silently, which is worse than the raise."""
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        first = client._backend
        bus = first._bus
        first.disconnect_error = RuntimeError("[org.bluez.Error.Failed] Not connected")
        first.is_connected = False
        await client.connect()             # reconnect retires the predecessor
        return bus, first

    bus, first = asyncio.run(scenario())
    assert bus.connected is False
    assert first._bus is None


def test_a_clean_disconnect_leaves_nothing_to_close(env):
    """When bleak completes its own teardown the helper is a no-op - it must
    not double-close or log on the ordinary path."""
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        bus = client._backend._bus
        await client.disconnect()
        return bus, client._backend

    bus, backend = asyncio.run(scenario())
    assert bus.connected is False
    assert backend._bus is None


def test_the_helper_degrades_when_bleak_has_no_such_attribute(env):
    """getattr-guarded throughout: a bleak that fixes or renames this must
    make the helper a no-op, not an error."""
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        del client._backend._bus
        assert await client._close_orphaned_bus() is False
        await client.disconnect()
        return True

    assert asyncio.run(scenario()) is True
