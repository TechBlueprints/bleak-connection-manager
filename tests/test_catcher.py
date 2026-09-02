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

A standing question for anything added here, because this suite has twice
been green over a live defect for the same underlying reason: WHAT FACT IS
THIS ASSERTION RESTING ON, AND DOES PRODUCTION SUPPLY THAT FACT?

- Adapter identity (2026-08-22): claims are keyed by a card's MAC and the
  catcher works in hciN, so every lookup crosses that boundary. The tests
  run where no MAC is readable, so adapter_key("hci5") degraded to "hci5"
  and both sides of every comparison agreed - for a reason unrelated to
  correctness. Seven raw lookups were broken on every real box. Tests that
  cross that boundary must use MACs that actually resolve (_kernel_adapters).
- Cancellation (2026-08-25): cleanup placed after an await is skipped when
  a CancelledError passes through `except Exception`. Every test exercised
  the RAISING path, which is the one where that handler works, so the hole
  opened exactly where the tests did not look. Cancellation tests must
  assert on the CLEANUP, never on the exception - asserting that
  CancelledError propagates passes against broken and fixed code alike.

Both were found from outside this repo, by people standing where the fact
was visible. A new test earns its keep by failing against the code before
the fix; run it that way before trusting it.

The same shape shows up in production predicates, not just in tests, and
it is worth a name: THE ADJACENT PREDICATE - a check that is correct about
something true and nearby, but not about the thing that matters. It is
harder to spot than a wrong check, because reading it confirms the intent
while the code answers a different question. The scan claim's validity was
`ref() is not None`: a true and useful test of "has this scanner been
collected", written for an abandoned scanner nobody can stop, silently
standing in for "is a scan running" - so a finished scan went on claiming
an exclusive card forever (2026-08-25). Ask what question the predicate
answers, then ask whether that is the question you meant.
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

    # mirrors the real class: .dbus_error is the first constructor arg
    class BleakDBusError(exc.BleakError):
        def __init__(self, dbus_error, error_body):
            super().__init__(dbus_error, *error_body)

        @property
        def dbus_error(self):
            return self.args[0]

    exc.BleakDBusError = BleakDBusError
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


def _locks(claim_dir):
    """Claim names with the 0.5 bookkeeping files hidden: a hardlinked
    exclusive claim is a lock plus a holder file sharing one inode, and every
    assertion here is about the lock."""
    return sorted(n for n in os.listdir(claim_dir) if ".holder." not in n)


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
    catcher._recovery_attempts.clear()
    catcher._recovering.clear()
    catcher._quick_drop_streaks.clear()
    catcher._connect_failures.clear()
    catcher._daemon_dead_at = None
    catcher._last_daemon_check = None
    # wrappers from earlier tests linger in the weaksets until GC and carry
    # their (closed) loops with them; the heartbeat check borrows a loop
    # from exactly these, so a stale one makes it silently target a dead loop
    catcher._live_clients.clear()
    catcher._live_scanners.clear()
    catcher._last_loop = None
    monkeypatch.setattr(catcher, "_rotation", catcher.BleAdapterRotation())
    monkeypatch.setattr(catcher, "_connect_failures", {})
    monkeypatch.setattr(catcher, "_scan_failures", {})
    monkeypatch.setattr(catcher, "_scan_failure_since", {})
    # Card cycling defaults to ENABLED again (0.5: R2 and R3 make a reset
    # fire only on a card that emptied voluntarily). Point the disable flag
    # at a path that does not exist, and OUTSIDE the claim dir (which is
    # tmp_path itself) so the many "claims released, dir empty" assertions
    # still see an empty directory. The switch's own tests override this.
    monkeypatch.setattr(catcher, "CYCLE_DISABLE_FLAG", str(tmp_path.parent / f"no-card-cycle-{tmp_path.name}"))
    catcher._cycle_suppressed.clear()
    monkeypatch.setattr(catcher, "present_adapters", lambda: set())

    def install(adapters=(), link_caps=None, wrap_scanner=False, scan_to_score=False, validate_connection=None, adapter_config_path=None, gatt_timeout=catcher.GATT_OP_TIMEOUT):
        catcher.install_bleak_catcher(
            OWNER,
            adapters=adapters,
            link_caps=link_caps,
            adapter_config_path=adapter_config_path,
            gatt_timeout=gatt_timeout,
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


def test_every_card_scan_claimed_waits_and_never_scans_unclaimed(env, monkeypatch):
    """R1: the hard claim is a GATE. The fallthrough that used to live here -
    "every card is claimed, scan the best one anyway" - is what made
    2026-08-26 an outage: a second discovery on a busy card draws InProgress,
    InProgress was read as a radio failure, and three of those power-cycled a
    healthy adapter. The scan waits, and while it waits it must not have
    touched the backend at all."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")
    _foreign_file(env.dir, "hci6.scan")
    monkeypatch.setattr(catcher, "SCAN_CLAIM_WAIT", 0.25)
    monkeypatch.setattr(catcher, "SCAN_CLAIM_POLL", 0.02)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        with pytest.raises(catcher.ScanSlotWaitTimeout):
            await scanner.start()
        return scanner

    scanner = asyncio.run(scenario())
    assert RECORDED_SCANNER_INITS == [], "built a backend for a scan it never got a card for"
    assert SCANNER_START_RESULTS == []
    assert scanner._backend is None
    assert scanner._catcher_scanning is False
    # the foreign claims are untouched and no queue ticket was left behind
    assert set(os.listdir(env.dir)) == {"hci5.scan", "hci6.scan"}


def test_the_wait_timeout_names_every_holder_and_our_queue_position(env, monkeypatch):
    """A scan that gave up after 30s is only actionable if it says who it was
    waiting on: a holder that never lets go is a bug in that holder, and
    nobody can find it from "scan failed"."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")
    _foreign_file(env.dir, "hci6.scan")
    monkeypatch.setattr(catcher, "SCAN_CLAIM_WAIT", 0.25)
    monkeypatch.setattr(catcher, "SCAN_CLAIM_POLL", 0.02)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        with pytest.raises(catcher.ScanSlotWaitTimeout) as caught:
            await scanner.start()
        return str(caught.value)

    message = asyncio.run(scenario())
    assert "hci5" in message and "hci6" in message
    assert "foreign-svc" in message                 # the holder, by name
    assert "pid 1" in message
    assert "queued 1 of 1" in message               # and where we stood
    # NOT the out-of-slots substring: this is not a connect, and brc's
    # out-of-slots backoff is not the right pacing for it
    assert not message.startswith("connection slot")


def test_a_release_in_this_process_wakes_a_waiting_scanner(env, monkeypatch):
    """The poll is the correctness floor; the release event is what makes a
    handover take milliseconds. Polled at 5s here so only the event can
    explain a fast acquire."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "SCAN_CLAIM_WAIT", 5.0)
    monkeypatch.setattr(catcher, "SCAN_CLAIM_POLL", 5.0)

    async def scenario():
        holder = catcher._config.claims.claim_hard("hci5")
        scanner = sys.modules["bleak"].BleakScanner()
        started = asyncio.get_running_loop().create_task(scanner.start())
        await asyncio.sleep(0.05)
        assert not started.done(), "scanned while another claim was live"
        began = time.monotonic()
        catcher._config.claims.release(holder)
        await asyncio.wait_for(started, timeout=2.0)
        elapsed = time.monotonic() - began
        adapter = scanner._catcher_adapter
        await scanner.stop()
        return adapter, elapsed

    adapter, elapsed = asyncio.run(scenario())
    assert adapter == "hci5"
    assert elapsed < 1.0, "woke on the poll tick, not on the release"


def test_a_constructed_scanner_holds_nothing_until_it_is_started(env):
    """Construction is free of side effects on shared hardware: callers build
    placeholder scanners long before (and often without) starting them."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    scanner = sys.modules["bleak"].BleakScanner()
    assert scanner._catcher_claim is None
    assert scanner._catcher_adapter is None
    assert os.listdir(env.dir) == []


def test_an_explicit_adapter_waits_on_that_card_alone(env, monkeypatch):
    """The caller's choice is never overridden, so "wait" means wait for the
    card they named - not quietly move to a free one."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")            # hci6 is free
    monkeypatch.setattr(catcher, "SCAN_CLAIM_WAIT", 0.25)
    monkeypatch.setattr(catcher, "SCAN_CLAIM_POLL", 0.02)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner(adapter="hci5")
        with pytest.raises(catcher.ScanSlotWaitTimeout) as caught:
            await scanner.start()
        return str(caught.value)

    message = asyncio.run(scenario())
    assert "hci5" in message
    assert "hci6" not in message, "wandered off the caller's explicit adapter"
    assert RECORDED_SCANNER_INITS == []


def test_the_rssi_sweeper_still_skips_rather_than_queues(env, monkeypatch):
    """Opportunistic work must never queue. A sweep exists to improve a
    score; making it wait 30s for a card would have it holding a scan claim
    for a window it did not need, on behalf of nobody."""
    env.install(adapters=("hci5",), scan_to_score=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    _foreign_file(env.dir, "hci5.scan")

    async def scenario():
        began = time.monotonic()
        await catcher._config.sweeper._sweep_adapter("hci5")
        return time.monotonic() - began

    assert asyncio.run(scenario()) < 1.0
    assert RECORDED_SCANNER_INITS == []


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

    async def fake_reset(adapter, claims_manager=None, force=False, gone_silent=False, **kw):
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

    async def fake_reset(adapter, claims_manager=None, force=False, gone_silent=False, **kw):
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


def test_every_adapter_draining_refuses_the_connect_with_the_typed_error(env):
    """R3: nothing NEW starts on a card someone is emptying, with no
    fallback. Work placed onto a draining card tops the drain back up, and
    the reset that follows is only safe because the card emptied
    voluntarily. The error carries brc's out-of-slots substring so its 4s
    backoff paces the retries across the (bounded, <=60s) drain window."""
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.drain")
    _foreign_file(env.dir, "hci6.drain")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    with pytest.raises(catcher.OutOfConnectionSlotsError) as caught:
        asyncio.run(client.connect())

    assert str(caught.value).startswith("connection slot")
    assert RECORDED_INITS == []
    assert sorted(os.listdir(env.dir)) == ["hci5.drain", "hci6.drain"]


def test_a_pinned_device_gets_no_carve_out_from_a_drain(env):
    """A pin says which radio a device should PREFER. It is not an
    instruction to walk into a reset in progress."""
    env.install(adapters=(f"{ADDRESS}@hci5",))
    _foreign_file(env.dir, "hci5.drain")
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    with pytest.raises(catcher.OutOfConnectionSlotsError):
        asyncio.run(client.connect())

    assert RECORDED_INITS == []


def test_an_explicit_adapter_gets_no_carve_out_from_a_drain(env):
    """The last path that could still put new work on a draining card."""
    env.install(adapters=("hci5", "hci6"))
    _foreign_file(env.dir, "hci5.drain")
    client = sys.modules["bleak"].BleakClient(ADDRESS, adapter="hci5", _is_retry_client=True)

    with pytest.raises(catcher.OutOfConnectionSlotsError):
        asyncio.run(client.connect())

    assert RECORDED_INITS == []


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


def test_the_drain_watcher_never_kicks_a_live_link(env):
    """Clint, 2026-09-02: a card is cycled only when it is EMPTY. The
    version this replaces disconnected a connected client so the resetter
    could proceed - a forced mass teardown, which on BlueZ 5.72 is the
    detonation path of the gatt-client use-after-free. A foreign drain now
    steers new placements away and nothing more; this link stays up, its
    claims stay live, and they keep vetoing the reset."""
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
    assert client.is_connected is True
    assert client._catcher_drain_kicked is None
    remaining = [n for n in os.listdir(env.dir) if not n.endswith(".drain")]
    assert remaining, "the link's claims were released for a reset it must veto"

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
            # await the migration itself rather than guessing how many loop
            # turns it needs - wait_for wraps each backend call in a task,
            # so a fixed yield count is a hostage to internal scheduling
            for _ in range(3):
                await asyncio.sleep(0)
            if catcher._migration_tasks:
                await asyncio.gather(*list(catcher._migration_tasks), return_exceptions=True)
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
        assert client._close_orphaned_bus() is False
        await client.disconnect()
        return True

    assert asyncio.run(scenario()) is True


def test_a_dbus_ceiling_failure_is_not_charged_to_the_radio(env):
    """Field 2026-08-24: at dbus's per-user ceiling every connect fails at
    once, on every adapter, because bleak opens a connection per session and
    the bus refuses it. Scoring those as adapter failures would rank every
    card bad simultaneously and walk pinned devices off working radios, for
    a cause no adapter can fix."""
    env.install(adapters=("hci5", "hci6"))
    CONNECT_RESULTS.append(
        RuntimeError("org.freedesktop.DBus.Error.LimitsExceeded: maximum number of "
                     "active connections for UID reached")
    )

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        with pytest.raises(RuntimeError):
            await client.connect()

    asyncio.run(scenario())

    assert catcher._connect_failures == {}      # no card blamed
    assert catcher._rotation.index(ADDRESS) == 0  # no pin walked


def test_an_ordinary_connect_failure_is_still_charged(env):
    """The discrimination has to be narrow: a real radio failure must still
    penalize its adapter and advance the walk."""
    env.install(adapters=("hci5", "hci6"))
    CONNECT_RESULTS.append(RuntimeError("le-connection-abort-by-local"))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        with pytest.raises(RuntimeError):
            await client.connect()

    asyncio.run(scenario())

    assert catcher._connect_failures != {}


def test_fd_exhaustion_is_also_not_the_radios_fault(env):
    env.install(adapters=("hci5",))
    CONNECT_RESULTS.append(OSError(24, "Too many open files"))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        with pytest.raises(OSError):
            await client.connect()

    asyncio.run(scenario())
    assert catcher._connect_failures == {}


def test_a_cancelled_disconnect_still_closes_the_bus_and_releases_claims(env):
    """The path that leaks most reliably. A cancelled disconnect never
    reaches bleak's own bus teardown, and if our cleanup sat behind an
    await it would not run either: CancelledError is a BaseException, so
    `except Exception` does not catch it and anything sequenced after the
    await is skipped. Both our steps are synchronous for that reason.
    (Raised by the sensors-py session, whose fix is synchronous too.)"""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        bus = client._backend._bus
        assert len(os.listdir(env.dir)) == 2

        async def cancelled_disconnect():
            raise asyncio.CancelledError()

        client._backend.disconnect = cancelled_disconnect
        with pytest.raises(asyncio.CancelledError):
            await client.disconnect()
        return bus, client._backend

    bus, backend = asyncio.run(scenario())
    assert bus.connected is False   # socket released
    assert backend._bus is None     # and bleak's guards see it closed
    assert os.listdir(env.dir) == []  # claims released, not stranded


def test_a_cancelled_retirement_still_closes_the_bus(env):
    """Same hazard on the reconnect path."""
    env.install(adapters=("hci5",))

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        first, bus = client._backend, client._backend._bus

        async def cancelled_disconnect():
            raise asyncio.CancelledError()

        first.disconnect = cancelled_disconnect
        first.is_connected = False
        # cancellation still aborts the reconnect - it just no longer
        # takes a system-bus connection with it
        with pytest.raises(asyncio.CancelledError):
            await client.connect()
        return bus, first

    bus, first = asyncio.run(scenario())
    assert bus.connected is False
    assert first._bus is None


def test_a_mac_entry_is_resolved_against_the_current_numbering(env, monkeypatch):
    """Naming a card by MAC is a statement that its number may change, so
    the lookup must not be served from a cache. Within one TTL a stale
    mapping would place a pinned device on whatever card inherited the
    number - the isolation failure pins exist to prevent."""
    env.install(adapters=("AA:BB:CC:DD:EE:FF",))
    # ONE live mapping, mutated in place: re-running _kernel_adapters would
    # reset _mac_cache itself and the test would pass without the fix -
    # the freshness has to come from the code, not from the fixture
    live = {"hci3": "AA:BB:CC:DD:EE:FF", "hci4": "11:22:33:44:55:66"}
    monkeypatch.setattr(catcher.claims, "_mac_cache", {})
    monkeypatch.setattr(catcher.claims, "present_hci_names", lambda: sorted(live))
    monkeypatch.setattr(catcher.claims, "_read_adapter_mac",
                        lambda a: live.get(a, catcher.claims.UNKNOWN_MAC))
    monkeypatch.setattr(catcher, "present_adapters", lambda: set(live))
    catcher._observed_identities.clear()

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    assert RECORDED_INITS[-1]["adapter"] == "hci3"   # cache now holds hci3 -> our MAC

    # the card renumbers WITHOUT the cache expiring - a USB reset by another
    # process, a replug - and the very next placement must follow the card
    live["hci3"], live["hci4"] = "11:22:33:44:55:66", "AA:BB:CC:DD:EE:FF"
    client2 = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client2.connect())

    assert RECORDED_INITS[-1]["adapter"] == "hci4"


def test_hci_entries_pay_nothing_for_the_fresh_resolve(env, monkeypatch):
    """An adapter written as hciN has nothing to resolve, so a config using
    only numbers must not start paying for an hciconfig call per placement."""
    env.install(adapters=("hci5", "hci6"))
    _kernel_adapters(monkeypatch, {"hci5": "AA:BB:CC:DD:EE:FF", "hci6": "11:22:33:44:55:66"})
    calls = []
    real = catcher.claims.invalidate_adapter_mac
    monkeypatch.setattr(catcher.claims, "invalidate_adapter_mac",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())

    assert calls == []


def test_a_finished_scan_stops_holding_its_hard_claim(env):
    """Field 2026-08-25: a single 12s discovery held a SHARED card's .scan
    claim for minutes after the discovery ended, announcing an exclusive
    scan on a card nobody was scanning on and steering every other process
    off it. The claim says "I am scanning here" and must be valid for
    exactly as long as that is true."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        assert _locks(env.dir) == ["hci5.scan"]
        await scanner.stop()
        return scanner

    scanner = asyncio.run(scenario())
    assert os.listdir(env.dir) == []          # released with the activity
    assert scanner._catcher_scanning is False  # and the scanner is still alive


def test_a_stopped_but_referenced_scanner_is_swept(env):
    """The backstop for the same thing: even if a release were missed, the
    heartbeat must not keep refreshing a claim for a scan that has ended.
    Keying validity on the object's existence covered only collection."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        claim = scanner._catcher_claim
        # simulate the release being missed while the scan itself has ended
        scanner._catcher_scanning = False
        return scanner, claim

    scanner, claim = asyncio.run(scenario())
    assert claim.validity() is False
    catcher._config.claims._beat_once()
    assert os.listdir(env.dir) == []          # swept, not refreshed forever


def test_stop_releases_the_claim_even_with_no_backend(env):
    """stop() early-returned before releasing when there was no backend to
    stop - releasing is the whole point of being asked to stop."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        scanner._backend = None               # nothing left to stop
        await scanner.stop()

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []


def test_a_running_scan_keeps_its_claim(env):
    """The guard must not over-fire: a scan that IS running keeps saying so."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        catcher._config.claims._beat_once()
        held = _locks(env.dir)
        await scanner.stop()
        return held

    assert asyncio.run(scenario()) == ["hci5.scan"]


def test_the_scan_flag_holds_through_bleaks_own_stop(env):
    """Review find: clearing the flag at stop() ENTRY opened a window
    during bleak's stop await where the heartbeat's validity check saw a
    "finished" scan and sweep-released the claim stop() was about to
    release anyway - same outcome, logged as a divergence when nothing
    diverged. The claim is deliberately held until the scan is actually
    stopped."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    observed = {}

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        real_stop = scanner._backend.stop

        async def observing_stop():
            observed["during"] = scanner._catcher_scanning
            observed["claim_valid"] = scanner._catcher_claim.validity()
            await real_stop()

        scanner._backend.stop = observing_stop
        await scanner.stop()
        return scanner

    scanner = asyncio.run(scenario())
    assert observed == {"during": True, "claim_valid": True}
    assert scanner._catcher_scanning is False
    assert os.listdir(env.dir) == []


def test_a_failed_restart_does_not_leave_the_flag_claiming_a_scan(env):
    """The documented invariant is "True only between a successful start()
    and stop()". A successful start followed by a direct re-start() that
    fails used to leave it stale-True - readable by nothing today, which is
    exactly how adjacent predicates get built tomorrow."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        assert scanner._catcher_scanning is True
        SCANNER_START_RESULTS.append(RuntimeError("start failed"))
        with pytest.raises(RuntimeError):
            await scanner.start()
        return scanner

    scanner = asyncio.run(scenario())
    assert scanner._catcher_scanning is False
    assert os.listdir(env.dir) == []


def _advert(name="x"):
    return types.SimpleNamespace(local_name=name, manufacturer_data=None,
                                 service_data=None, service_uuids=None)


def test_a_wedged_card_stops_looking_attractive_to_scan_selection(env, monkeypatch):
    """Field 2026-08-26, prod: two cards scan-wedged. A card that cannot
    scan holds no claims and carries no links, so the least-occupied ranker
    rated it BEST and the watchdog kept migrating discovery back onto it.
    The quiet must be recorded against the card before the restart re-runs
    selection - re-running selection changes nothing if nothing changed its
    inputs."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5", "hci6"})
    # the real reset_adapter runs here, probe included: keep probation short
    monkeypatch.setattr(catcher, "PROBE_SECONDS", 0.2)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        first = scanner._catcher_adapter
        # the card accepted the scan command and then reported nothing
        scanner._catcher_last_detection = scanner._catcher_start_time
        monkeypatch.setattr(scanner, "_quiet_seconds", lambda: 999.0)
        await scanner._watchdog_restart()
        return first, scanner._catcher_adapter

    first, after = asyncio.run(scenario())
    assert catcher._scan_failures.get(catcher.claims.adapter_key(first)) == 1
    assert after != first          # discovery moved off the wedged card


def test_a_scan_that_actually_sees_traffic_is_forgiven(env, monkeypatch):
    """The other half: a card with history that demonstrably scans again
    must be cleared - and by an advertisement, not by start() returning."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    catcher._scan_failures[catcher.claims.adapter_key("hci5")] = 3

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        # start alone must NOT forgive it
        assert catcher._scan_failures.get(catcher.claims.adapter_key("hci5")) == 3
        cb = RECORDED_SCANNER_INITS[-1]["detection_callback"]
        cb(types.SimpleNamespace(address=ADDRESS), _advert())
        return scanner

    asyncio.run(scenario())
    assert catcher._scan_failures.get(catcher.claims.adapter_key("hci5")) is None


def test_empty_advertisements_do_not_forgive_a_wedged_card(env, monkeypatch):
    """A wedged adapter can keep emitting empty advertisements; they are not
    evidence of scanning and must not clear the failure memory."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    catcher._scan_failures[catcher.claims.adapter_key("hci5")] = 2

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        cb = RECORDED_SCANNER_INITS[-1]["detection_callback"]
        cb(types.SimpleNamespace(address=ADDRESS),
           types.SimpleNamespace(local_name=None, manufacturer_data=None,
                                 service_data=None, service_uuids=None))

    asyncio.run(scenario())
    assert catcher._scan_failures.get(catcher.claims.adapter_key("hci5")) == 2


def test_a_hung_start_discovery_does_not_hold_a_card_hostage(env, monkeypatch):
    """Field 2026-08-26, prod: a consumer sat inside BlueZ StartDiscovery on
    a wedged adapter for 2h45m - alive, doing nothing, both its devices dead
    - while holding a HARD exclusive scan claim. The claim is taken before
    that call and its validity armed after it, so the heartbeat had no check
    to apply and refreshed it every beat for the whole 2h45m, denying the
    card to every other process on the box."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "SCAN_OP_TIMEOUT", 0.05)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()

        async def never_answers():
            await asyncio.sleep(3600)

        # patch the backend the wrapper is about to build
        original_init = RichBleakScanner.__init__

        def patched(self, *a, **k):
            original_init(self, *a, **k)
            self._backend.start = never_answers

        RichBleakScanner.__init__ = patched
        try:
            with pytest.raises(asyncio.TimeoutError):
                await scanner.start()
        finally:
            RichBleakScanner.__init__ = original_init
        return scanner

    scanner = asyncio.run(scenario())
    assert os.listdir(env.dir) == []                     # card released, not hostage
    assert catcher._scan_failures.get(catcher.claims.adapter_key("hci5")) == 1
    assert scanner._catcher_scanning is False


def test_a_hung_stop_discovery_cannot_wedge_the_watchdog(env, monkeypatch):
    """The same hazard on the way out, and worse placed: the watchdog stops
    before it restarts, so an unbounded stop would wedge the very machinery
    meant to recover from a wedged card."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "SCAN_OP_TIMEOUT", 0.05)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()

        async def never_answers():
            await asyncio.sleep(3600)

        scanner._backend.stop = never_answers
        with pytest.raises(asyncio.TimeoutError):
            await scanner.stop()
        return scanner

    scanner = asyncio.run(scenario())
    assert os.listdir(env.dir) == []      # released despite the hang


def test_a_hung_gatt_call_becomes_an_error_instead_of_silence(env):
    """An unbounded wait does not just risk the caller, it destroys the
    evidence: a call that hangs forever never becomes an observable
    failure, so "this is stuck" never happens as an event and no retry
    loop, score or recovery can act on it. Bounding it is what turns a
    hang into a fact."""
    env.install(adapters=("hci5",), gatt_timeout=0.05)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()

        async def never_answers(*a, **k):
            await asyncio.sleep(3600)

        client._backend.read_gatt_char = never_answers
        with pytest.raises(asyncio.TimeoutError):
            await client.read_gatt_char("fff1")
        return client

    client = asyncio.run(scenario())
    assert client.is_connected is True      # the link is not condemned by one slow call


def test_a_hung_disconnect_still_releases_its_claims(env):
    """disconnect holds a link slot and a soft claim across an unbounded
    D-Bus call - bleak bounds the event waits inside it but not the call
    itself - so a hang strands a SHARED resource, not just this caller."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2}, gatt_timeout=0.05)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        assert len(os.listdir(env.dir)) == 2

        async def never_answers():
            await asyncio.sleep(3600)

        client._backend.disconnect = never_answers
        with pytest.raises(asyncio.TimeoutError):
            await client.disconnect()

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []        # released despite the hang


def test_a_consumer_can_restore_unbounded_waits(env):
    """The escape hatch: a default that trades cost for a guarantee needs
    one, because the caller who does not want the guarantee is invisible
    from where the default is chosen."""
    env.install(adapters=("hci5",), gatt_timeout=None)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        assert client._op_timeout() is None
        return await client.read_gatt_char("fff1")

    assert asyncio.run(scenario()) == b"\x00"


def test_accumulated_scan_failures_trigger_a_drain_and_cycle(env, monkeypatch):
    """Detecting a wedged card and rotating off it protects the fleet but
    guarantees the card stays dead. Recovery was reachable only from the
    scanner watchdog, which needs a scanner that STARTED - so a card that
    hangs or refuses at StartDiscovery could never reach it. Accumulated
    failure now triggers it."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    attempts = []

    async def fake_reset(adapter, claims_manager=None, force=False, gone_silent=False, **kw):
        attempts.append((adapter, gone_silent))
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)

    async def scenario():
        for i in range(catcher.SCAN_FAILURES_BEFORE_RESET):
            if i == catcher.SCAN_FAILURES_BEFORE_RESET - 1:
                # age the streak past the churn guard: strikes alone are not
                # enough, they must also SPAN time (RECOVERY_STRIKE_SPAN)
                key = catcher.claims.adapter_key("hci5")
                catcher._scan_failure_since[key] -= catcher.RECOVERY_STRIKE_SPAN + 1
            SCANNER_START_RESULTS.append(RuntimeError("Set scan parameters failed"))
            scanner = sys.modules["bleak"].BleakScanner()
            with pytest.raises(RuntimeError):
                await scanner.start()
        if catcher._recovery_tasks:
            await asyncio.gather(*list(catcher._recovery_tasks), return_exceptions=True)

    asyncio.run(scenario())
    assert attempts == [("hci5", True)]      # drained and cycled, exactly once


def test_recovery_does_not_fire_before_the_threshold(env, monkeypatch):
    """One bad scan is not evidence of a wedged card."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    attempts = []

    async def fake_reset(adapter, **kw):
        attempts.append(adapter)
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)

    async def scenario():
        SCANNER_START_RESULTS.append(RuntimeError("transient"))
        scanner = sys.modules["bleak"].BleakScanner()
        with pytest.raises(RuntimeError):
            await scanner.start()
        if catcher._recovery_tasks:
            await asyncio.gather(*list(catcher._recovery_tasks), return_exceptions=True)

    asyncio.run(scenario())
    assert attempts == []


def test_a_card_that_will_not_come_back_is_not_cycled_forever(env, monkeypatch):
    """A physically dead radio must not be power-cycled indefinitely -
    giving up loudly is the signal a human should act on."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    attempts = []

    async def fake_reset(adapter, **kw):
        attempts.append(adapter)
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)

    async def scenario():
        for _ in range(catcher.MAX_RECOVERY_ATTEMPTS + 4):
            assert await catcher._recover_adapter("hci5") is False

    asyncio.run(scenario())
    assert len(attempts) == catcher.MAX_RECOVERY_ATTEMPTS


# (test_a_successful_recovery_clears_the_cards_record removed 2026-08-26:
# it pinned the contract that let a cosmetically-recovering card be reset
# forever. Superseded by test_reset_success_alone_does_not_clear_the_attempt_record
# and test_three_resets_without_proof_end_recovery.)


def test_a_card_that_comes_back_on_its_own_is_eligible_for_recovery_again(
    env, monkeypatch
):
    """Exhausting the recovery attempts is a verdict about a card's CURRENT
    state, not a life sentence. A human replugs the dongle, or the kernel
    re-enumerates it, and the card starts advertising again - that traffic
    is proof the radio works, and it must reopen recovery. Otherwise the
    first wedge a card ever suffers permanently disqualifies it from being
    fixed automatically, and the self-reinforcing trap that
    forget_adapter_failures exists to prevent for scan placement reappears
    one level up: the card is not recovered, so it stays wedged, so it
    accumulates failures, so it is never recovered."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})

    async def dead(adapter, **kw):
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", dead)

    async def scenario():
        for _ in range(catcher.MAX_RECOVERY_ATTEMPTS):
            await catcher._recover_adapter("hci5")

    asyncio.run(scenario())
    assert catcher._recovery_attempts[catcher.claims.adapter_key("hci5")] == (
        catcher.MAX_RECOVERY_ATTEMPTS
    )

    # the card comes back and advertises - the same success path a live
    # scanner's first real detection takes
    catcher._scan_finished("hci5", True)

    revived = []

    async def alive(adapter, **kw):
        revived.append(adapter)
        return True

    monkeypatch.setattr(catcher.recovery, "reset_adapter", alive)
    assert asyncio.run(catcher._recover_adapter("hci5")) is True
    assert revived == ["hci5"], "a card proven alive was never retried"


def test_a_completed_link_also_reopens_recovery(env, monkeypatch):
    """Connecting is proof the radio works just as advertising is."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    key = catcher.claims.adapter_key("hci5")
    catcher._recovery_attempts[key] = catcher.MAX_RECOVERY_ATTEMPTS
    catcher._connect_finished("hci5", "AA:BB:CC:DD:EE:FF", True)
    assert key not in catcher._recovery_attempts


def _inprogress():
    return catcher.BleakDBusError(
        "org.bluez.Error.InProgress", ["Operation already in progress"]
    )


def test_inprogress_on_stop_is_swallowed_and_counted(env):
    """Field 2026-08-26, prod: 22 of these across dbus-shyion-switch's
    hourly polls, every one on the 15s scan-timeout teardown. BlueZ 5.72
    answers a stop with InProgress when the kernel REJECTED it for not
    scanning - the scan is already over and bluetoothd detached the
    session before replying (its stop_discovery_complete removes the
    client before checking status). The stop's intent is satisfied;
    raising only destroyed the caller's find. Swallow it, count it."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()

        async def rejected_by_kernel():
            raise _inprogress()

        scanner._backend.stop = rejected_by_kernel
        await scanner.stop()          # must not raise
        return scanner

    scanner = asyncio.run(scenario())
    assert os.listdir(env.dir) == []                     # claim released
    assert scanner._catcher_scanning is False
    assert catcher._scan_failures.get(catcher.claims.adapter_key("hci5")) == 1


def test_inprogress_stops_never_reach_the_drain_and_cycle(env, monkeypatch):
    """The 0.5 correction, and the single most expensive misreading in this
    package's history. InProgress on a stop means two discovery sessions
    overlapped on one card. With the hard claim gating every start that is
    unreachable, so an instance of it is a report that something escaped the
    claim convention - a bug in this package or in a participant - and the
    one thing it is NOT is evidence that the radio is broken. Reading it as
    radio evidence is what power-cycled healthy adapters on 2026-08-26, and
    each cycle mass-disappeared that card's devices into the BlueZ 5.72
    gatt-client use-after-free. The card is still ranked down; it is never
    charged for somebody else's protocol violation."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    attempts = []

    async def fake_reset(adapter, claims_manager=None, force=False, gone_silent=False, **kw):
        attempts.append((adapter, gone_silent))
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)

    async def scenario():
        for i in range(catcher.SCAN_FAILURES_BEFORE_RESET):
            if i == catcher.SCAN_FAILURES_BEFORE_RESET - 1:
                # age the streak past the churn guard (see
                # RECOVERY_STRIKE_SPAN): count AND span are both required
                key = catcher.claims.adapter_key("hci5")
                catcher._scan_failure_since[key] -= catcher.RECOVERY_STRIKE_SPAN + 1
            scanner = sys.modules["bleak"].BleakScanner()
            await scanner.start()

            async def rejected_by_kernel():
                raise _inprogress()

            scanner._backend.stop = rejected_by_kernel
            await scanner.stop()
        if catcher._recovery_tasks:
            await asyncio.gather(*list(catcher._recovery_tasks), return_exceptions=True)

    asyncio.run(scenario())
    assert attempts == [], "an InProgress stop was charged to the radio"
    # ranked down, though: a card producing these keeps losing ties
    assert catcher._scan_failures[catcher.claims.adapter_key("hci5")] == catcher.SCAN_FAILURES_BEFORE_RESET


def test_other_dbus_errors_on_stop_still_raise(env):
    """The swallow is for one error whose semantics we established at
    source, not a policy of ignoring teardown failures."""
    env.install(adapters=("hci5",), wrap_scanner=True)

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()

        async def refused():
            raise catcher.BleakDBusError("org.bluez.Error.Failed", ["Failed"])

        scanner._backend.stop = refused
        with pytest.raises(catcher.BleakDBusError):
            await scanner.stop()

    asyncio.run(scenario())
    assert os.listdir(env.dir) == []      # released regardless, via finally


def test_a_successful_connection_clears_the_scan_strikes(env):
    """Clint, 2026-08-26: "if we get a successful connection we should
    still clear the 3 strikes." A completed link is proof the radio works;
    strikes held against a demonstrably working card only rank it last."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    key = catcher.claims.adapter_key("hci5")
    catcher._scan_failures[key] = catcher.SCAN_FAILURES_BEFORE_RESET - 1
    catcher._connect_finished("hci5", "AA:BB:CC:DD:EE:FF", True)
    assert key not in catcher._scan_failures


def test_a_dead_daemon_restarts_bluetoothd_not_the_card(env, monkeypatch):
    """2026-08-26 prod incident: bluetoothd crash-looped and the per-card
    strike logic read every daemon-caused scan failure as a wedged radio,
    power-cycling healthy cards 14 times. With the daemon down, recovery
    must restart the daemon, wipe the outage's strikes, and charge no card
    attempt."""
    calls = []
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: False)

    async def fake_restart(*a, **k):
        calls.append("restart-daemon")
        return True

    async def fake_invalidate():
        calls.append("invalidate-dbus")

    async def fake_reset(*a, **k):
        calls.append("card-cycle")
        return True

    monkeypatch.setattr(catcher.recovery, "restart_bluetoothd", fake_restart)
    monkeypatch.setattr(catcher.recovery, "invalidate_dbus_state", fake_invalidate)
    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    catcher._scan_failures[catcher.claims.adapter_key("hci5")] = catcher.SCAN_FAILURES_BEFORE_RESET
    assert asyncio.run(catcher._recover_adapter("hci5")) is True
    assert calls == ["restart-daemon", "invalidate-dbus"]
    assert catcher._scan_failures == {}
    assert catcher._recovery_attempts == {}


def test_daemon_restart_is_rate_limited_to_the_grace_window(env, monkeypatch):
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: False)
    calls = []

    async def fake_restart(*a, **k):
        calls.append("restart-daemon")
        return True

    async def fake_invalidate():
        pass

    monkeypatch.setattr(catcher.recovery, "restart_bluetoothd", fake_restart)
    monkeypatch.setattr(catcher.recovery, "invalidate_dbus_state", fake_invalidate)
    assert asyncio.run(catcher._recover_adapter("hci5")) is True
    assert asyncio.run(catcher._recover_adapter("hci6")) is False
    assert calls == ["restart-daemon"]


def test_card_recovery_stands_down_inside_the_daemon_grace_window(env, monkeypatch):
    """Strikes accumulated while the daemon was dead or re-registering are
    outage residue, not card evidence - a card recovery queued on them must
    not fire, and must not be charged as an attempt."""
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)

    async def fake_reset(*a, **k):
        raise AssertionError("card must not be cycled inside the grace window")

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    monkeypatch.setattr(catcher, "_daemon_dead_at", time.monotonic())
    assert asyncio.run(catcher._recover_adapter("hci5")) is False
    assert catcher._recovery_attempts == {}


def test_no_prior_daemon_death_leaves_card_recovery_untouched(env, monkeypatch):
    """The sentinel matters: a freshly booted box (small monotonic clock)
    with no daemon death on record must run normal card recovery, not a
    phantom grace window."""
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)

    async def fake_reset(*a, **k):
        return True

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    assert asyncio.run(catcher._recover_adapter("hci5")) is True


def test_reset_success_alone_does_not_clear_the_attempt_record(env, monkeypatch):
    """A reset's "success" is cosmetic - the card re-enumerated with a
    readable MAC - and clearing attempts on it made the reset loop
    unbounded (2026-08-26 prod load cascade). Only proof the radio works
    clears the record."""
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)

    async def fake_reset(*a, **k):
        return True

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    key = catcher.claims.adapter_key("hci5")
    assert asyncio.run(catcher._recover_adapter("hci5")) is True
    assert catcher._recovery_attempts[key] == 1
    # proof: a scanner starts on the card - NOW the record clears
    catcher._scan_finished("hci5", True)
    assert key not in catcher._recovery_attempts


def test_three_resets_without_proof_end_recovery(env, monkeypatch):
    """The cosmetic-success loop is bounded: reset "succeeds" every time,
    the radio never proves itself, and the fourth round refuses."""
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)
    resets = []

    async def fake_reset(*a, **k):
        resets.append(1)
        return True

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    for _ in range(catcher.MAX_RECOVERY_ATTEMPTS):
        assert asyncio.run(catcher._recover_adapter("hci5")) is True
    assert asyncio.run(catcher._recover_adapter("hci5")) is False
    assert len(resets) == catcher.MAX_RECOVERY_ATTEMPTS


def test_a_young_strike_streak_does_not_schedule_recovery(env, monkeypatch):
    """Three failures in ten seconds is what a mass restart looks like;
    recovery waits until the streak spans RECOVERY_STRIKE_SPAN."""
    key = catcher.claims.adapter_key("hci5")
    monkeypatch.setattr(catcher, "_scan_failure_since", {})
    for _ in range(catcher.SCAN_FAILURES_BEFORE_RESET):
        catcher._scan_finished("hci5", False)
    assert catcher._scan_failures[key] == catcher.SCAN_FAILURES_BEFORE_RESET

    async def scenario():
        catcher._schedule_recovery("hci5")
        assert key not in catcher._recovering          # young streak: refused
        catcher._scan_failure_since[key] = time.monotonic() - catcher.RECOVERY_STRIKE_SPAN - 1
        monkeypatch.setattr(catcher, "_daemon_dead_at", None)
        monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)

        async def fake_reset(*a, **k):
            return True

        monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
        catcher._schedule_recovery("hci5")
        assert key in catcher._recovering              # aged streak: queued
        while key in catcher._recovering:
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_scan_success_resets_the_streak_clock(env):
    key = catcher.claims.adapter_key("hci5")
    catcher._scan_finished("hci5", False)
    assert key in catcher._scan_failure_since
    catcher._scan_finished("hci5", True)
    assert key not in catcher._scan_failure_since


def test_a_dead_daemon_is_checked_before_every_card_gate(env, monkeypatch):
    """2026-08-26 15:48, observed live: with the daemon dead no scan can
    succeed, so no proof ever clears an attempt record, and a process whose
    cards had burnt their attempts lost its only path to the daemon
    restart - the box went dark a second time. The daemon check now
    precedes the strike threshold, the span gate, and the attempt cap: one
    failure on a burnt card must still restart a dead daemon."""
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: False)
    restarts = []

    async def fake_restart(*a, **k):
        restarts.append(1)
        return True

    async def fake_invalidate():
        pass

    monkeypatch.setattr(catcher.recovery, "restart_bluetoothd", fake_restart)
    monkeypatch.setattr(catcher.recovery, "invalidate_dbus_state", fake_invalidate)
    key = catcher.claims.adapter_key("hci5")
    catcher._scan_failures[key] = 1                                   # below threshold
    catcher._recovery_attempts[key] = catcher.MAX_RECOVERY_ATTEMPTS   # burnt

    async def scenario():
        catcher._schedule_recovery("hci5")
        if catcher._recovery_tasks:
            await asyncio.gather(*list(catcher._recovery_tasks), return_exceptions=True)

    asyncio.run(scenario())
    assert restarts == [1]


def test_a_failed_connect_also_restarts_a_dead_daemon(env, monkeypatch):
    """A consumer that only ever connects (a battery driver with a pinned
    address never scans) must still be able to restart a dead bluetoothd -
    the daemon fails connects and scans alike."""
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: False)
    restarts = []

    async def fake_restart(*a, **k):
        restarts.append(1)
        return True

    async def fake_invalidate():
        pass

    monkeypatch.setattr(catcher.recovery, "restart_bluetoothd", fake_restart)
    monkeypatch.setattr(catcher.recovery, "invalidate_dbus_state", fake_invalidate)

    async def scenario():
        catcher._connect_finished("hci5", "AA:BB:CC:DD:EE:FF", False)
        if catcher._recovery_tasks:
            await asyncio.gather(*list(catcher._recovery_tasks), return_exceptions=True)

    asyncio.run(scenario())
    assert restarts == [1]


def test_cycling_is_armed_by_default_and_the_kill_switch_disarms_it_live(env, monkeypatch, tmp_path):
    """R4: cycling returns as the last rung of recovery, default ON, because
    R2 and R3 make it safe by construction - a reset can only fire on a card
    that emptied voluntarily. A card that is genuinely wedged is also a card
    nobody is using, and steering around it forever guarantees it stays dead;
    recovery that can never fire is not caution, it is a fleet quietly losing
    radios one at a time.

    The switch stays for the operator, and it must work WITHOUT restarting
    anything - restarts re-register D-Bus clients and are themselves part of
    the crash pattern this switch exists around, so a switch that needed one
    would confound the very test it enables (field 2026-08-26)."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    flag = tmp_path.parent / f"no-card-cycle-{tmp_path.name}"
    monkeypatch.setattr(catcher, "CYCLE_DISABLE_FLAG", str(flag))
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)
    attempts = []

    async def fake_reset(adapter, **kw):
        attempts.append(adapter)
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)

    async def scenario():
        assert not flag.exists()                 # default: ARMED (0.5)
        await catcher._recover_adapter("hci5")
        assert attempts == ["hci5"], "cycling did not fire by default"
        flag.write_text("")                      # opt out, live, no restart
        try:
            assert await catcher._recover_adapter("hci5") is False
        finally:
            flag.unlink()

    asyncio.run(scenario())
    assert attempts == ["hci5"], "cycled a card while disarmed"


def test_the_kill_switch_charges_no_attempt(env, monkeypatch, tmp_path):
    """A suppressed cycle is not a failed cycle. If disarming burned the
    3-attempt budget, re-arming would find every card already written off
    as beyond reach."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    flag = tmp_path.parent / f"no-card-cycle-{tmp_path.name}"
    flag.write_text("")
    monkeypatch.setattr(catcher, "CYCLE_DISABLE_FLAG", str(flag))
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)

    async def scenario():
        for _ in range(catcher.MAX_RECOVERY_ATTEMPTS + 2):
            await catcher._recover_adapter("hci5")

    asyncio.run(scenario())
    assert catcher._recovery_attempts == {}


def test_the_kill_switch_still_lets_a_dead_daemon_be_restarted(env, monkeypatch, tmp_path):
    """The switch takes our hands off the CARDS, not off the box. A dead
    bluetoothd takes all BLE down and restarting it touches no hardware."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    flag = tmp_path.parent / f"no-card-cycle-{tmp_path.name}"
    flag.write_text("")
    monkeypatch.setattr(catcher, "CYCLE_DISABLE_FLAG", str(flag))
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: False)
    called = []

    async def fake_daemon(adapter):
        called.append(adapter)
        return True

    monkeypatch.setattr(catcher, "_recover_daemon", fake_daemon)
    asyncio.run(catcher._recover_adapter("hci5"))
    assert called == ["hci5"]


def test_the_heartbeat_notices_a_dead_daemon_with_nothing_failing(env, monkeypatch):
    """Found on dev 2026-08-27: every other daemon check hangs off a
    FAILURE, so an idle process (a consumer in a long reconnect backoff,
    or one whose devices are all disconnected) never looks, and bluetoothd
    stays dead until someone happens to want Bluetooth. The claim
    heartbeat ticks on a timer, so it is the one place an idle process
    still looks at the world."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "_daemon_dead_at", None)
    monkeypatch.setattr(catcher, "_last_daemon_check", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: False)
    restarts = []

    async def fake_restart(*a, **k):
        restarts.append(1)
        return True

    async def fake_invalidate():
        pass

    monkeypatch.setattr(catcher.recovery, "restart_bluetoothd", fake_restart)
    monkeypatch.setattr(catcher.recovery, "invalidate_dbus_state", fake_invalidate)

    async def scenario():
        # a live loop exists, but NOTHING is failing and no client or
        # scanner is alive to borrow it from - the idle-box shape
        catcher._remember_loop(asyncio.get_running_loop())
        assert not catcher._live_clients and not catcher._live_scanners
        catcher._drain_watch()                       # runs on the heartbeat thread
        await asyncio.sleep(0)                       # let the scheduled check run
        if catcher._recovery_tasks:
            await asyncio.gather(*list(catcher._recovery_tasks), return_exceptions=True)

    asyncio.run(scenario())
    assert restarts == [1], "an idle process never noticed the daemon was gone"


def test_the_heartbeat_check_is_silent_with_no_loop_to_use(env, monkeypatch):
    """Before any wrapper has run there is no loop to schedule onto. The
    check must not raise on the heartbeat thread - a hook that throws
    would take the drain watch down with it."""
    env.install(adapters=("hci5",))
    monkeypatch.setattr(catcher, "_last_loop", None)
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: False)
    catcher._drain_watch()          # must not raise


# -- the scanwait queue (convention 0.5) ----------------------------------
#
# The queue exists because a gate without one turns N blocked scanners into
# N pollers all watching the same best-ranked card: the first release wakes
# every one of them, one wins, and the rest have burned a wake for nothing
# while other cards sat idle. Its two failure modes are starvation (a
# newcomer stealing a card the queue was waiting for) and thrash (waiters
# swapping queues forever), and each rule below exists for one of them.


def _foreign_wait(claim_dir, adapter, seq, service="foreign-svc", pid=1):
    return _foreign_file(claim_dir, f"{adapter}.scanwait.{service}-{pid}-{seq}", pid=pid)


def _our_waits(claim_dir):
    return sorted(n for n in os.listdir(claim_dir) if ".scanwait." in n and OWNER in n)


def _snapshot(**queues):
    """A claims() snapshot carrying only the fields the queue rules read."""
    return {
        catcher.claims.adapter_key(adapter): {
            "hard": None, "hard_pid": None, "soft": 0, "soft_owners": [],
            "links": 0, "drain": False, "drain_pid": None, "waiters": waiters,
        }
        for adapter, waiters in queues.items()
    }


def _queue(*names):
    return [(index, "svc", 1, name) for index, name in enumerate(names)]


def _ticket(name):
    return types.SimpleNamespace(path=os.path.join("/run/bt-claims", name))


@contextlib.contextmanager
def _short_wait(monkeypatch, wait=5.0, poll=0.02):
    monkeypatch.setattr(catcher, "SCAN_CLAIM_WAIT", wait)
    monkeypatch.setattr(catcher, "SCAN_CLAIM_POLL", poll)
    yield


async def _start_waiting():
    """A scan placement in flight, parked in its wait loop."""
    task = asyncio.get_running_loop().create_task(catcher._acquire_scan_adapter())
    await asyncio.sleep(0.05)
    return task


async def _abandon(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_an_unqueued_waiter_joins_the_shortest_queue(env):
    snapshot = _snapshot(hci5=_queue("a", "b"), hci6=_queue("x"))
    assert catcher._queue_target(snapshot, ["hci5", "hci6"], None, None, False) == "hci6"


def test_queue_ties_break_by_rank_then_configuration_order(env):
    """`ranked` already carries occupancy-then-config order, so an equal-length
    queue is decided by the same ordering everything else here uses."""
    snapshot = _snapshot(hci5=_queue("a"), hci6=_queue("x"))
    assert catcher._queue_target(snapshot, ["hci5", "hci6"], None, None, False) == "hci5"
    assert catcher._queue_target(snapshot, ["hci6", "hci5"], None, None, False) == "hci6"


def test_migration_needs_a_strictly_shorter_queue(env):
    """The worked example from the design. Third of four on hci5 has two
    waiters ahead of it; hci6's entire queue is one, and 1 < 2, so it moves.
    Then nothing moves again: the waiter that inherited third place on hci5
    also has two ahead, hci6 now totals two, and 2 < 2 is false. The
    strictness is the whole anti-thrash argument - under <= that pair would
    swap queues forever, each seeing the other's as equally good."""
    snapshot = _snapshot(hci5=_queue("a", "b", "me", "d"), hci6=_queue("x"))
    assert catcher._queue_target(snapshot, ["hci5", "hci6"], _ticket("me"), "hci5", True) == "hci6"
    settled = _snapshot(hci5=_queue("a", "b", "d"), hci6=_queue("x", "me"))
    assert catcher._queue_target(settled, ["hci5", "hci6"], _ticket("me"), "hci6", True) is None
    assert catcher._queue_target(settled, ["hci5", "hci6"], _ticket("d"), "hci5", True) is None


def test_migration_is_a_poll_tick_decision_never_an_event(env):
    """The other half of the anti-thrash pair. One release wakes every waiter
    at once; if each of them could migrate on that wake, they would all
    recompute from the same instant's snapshot and stampede the same queue."""
    snapshot = _snapshot(hci5=_queue("a", "b", "me", "d"), hci6=_queue("x"))
    assert catcher._queue_target(snapshot, ["hci5", "hci6"], _ticket("me"), "hci5", False) is None
    assert catcher._queue_target(snapshot, ["hci5", "hci6"], _ticket("me"), "hci5", True) == "hci6"


def test_two_waiters_on_two_queues_settle_where_they_are(env):
    snapshot = _snapshot(hci5=_queue("me"), hci6=_queue("you"))
    assert catcher._queue_target(snapshot, ["hci5", "hci6"], _ticket("me"), "hci5", True) is None
    assert catcher._queue_target(snapshot, ["hci5", "hci6"], _ticket("you"), "hci6", True) is None


def test_a_waiter_whose_card_starts_draining_finds_every_other_better(env):
    """A draining card is not in `ranked` at all, which makes its queue
    infinitely expensive: a longer queue elsewhere still beats it. With no
    elsewhere, the waiter stays put and wakes when the drain releases."""
    snapshot = _snapshot(hci5=_queue("me"), hci6=_queue("x", "y", "z"))
    assert catcher._queue_target(snapshot, ["hci6"], _ticket("me"), "hci5", True) == "hci6"
    assert catcher._queue_target(snapshot, [], _ticket("me"), "hci5", True) is None


def test_a_waiting_scan_queues_on_the_shortest_queue(env, monkeypatch):
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")
    _foreign_file(env.dir, "hci6.scan")
    _foreign_wait(env.dir, "hci5", 1)
    _foreign_wait(env.dir, "hci5", 2)
    _foreign_wait(env.dir, "hci6", 3)

    async def scenario():
        with _short_wait(monkeypatch):
            task = await _start_waiting()
            ours = _our_waits(env.dir)
            await _abandon(task)
            return ours

    assert [n.partition(".")[0] for n in asyncio.run(scenario())] == ["hci6"]
    assert _our_waits(env.dir) == [], "a ticket outlived the scanner waiting on it"


def test_a_younger_waiter_does_not_steal_the_card_it_frees(env, monkeypatch):
    """Starvation, the queue's first failure mode: without FIFO the ticket
    is decorative and whoever waited longest waits forever."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    holder = _foreign_file(env.dir, "hci5.scan")
    _foreign_wait(env.dir, "hci5", 0)          # older than anything we can mint

    async def scenario():
        with _short_wait(monkeypatch):
            task = await _start_waiting()
            os.unlink(holder)                  # the card frees...
            await asyncio.sleep(0.15)          # ...several poll ticks pass
            took_it = task.done()
            await _abandon(task)
            return took_it

    assert asyncio.run(scenario()) is False


def test_the_oldest_ticket_takes_the_card_when_it_frees(env, monkeypatch):
    """The same setup with the other waiter YOUNGER than ours: now it is our
    turn, and the queue must not make us wait for a ticket behind us."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    holder = _foreign_file(env.dir, "hci5.scan")
    _foreign_wait(env.dir, "hci5", 10 ** 9)    # queued after us

    async def scenario():
        with _short_wait(monkeypatch):
            task = await _start_waiting()
            os.unlink(holder)
            adapter, claim = await asyncio.wait_for(task, timeout=2.0)
            catcher._config.claims.release(claim)
            return adapter

    assert asyncio.run(scenario()) == "hci5"
    assert _our_waits(env.dir) == []


def test_a_free_card_with_no_queue_is_taken_from_wherever_we_are_queued(env, monkeypatch):
    """A waiter that queued politely while a card sat idle and unwanted would
    be obeying the queue at the cost of the thing the queue is for."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")
    free_me = _foreign_file(env.dir, "hci6.scan")

    async def scenario():
        with _short_wait(monkeypatch):
            task = await _start_waiting()
            queued_on = [n.partition(".")[0] for n in _our_waits(env.dir)]
            os.unlink(free_me)                 # hci6 frees, with no queue
            adapter, claim = await asyncio.wait_for(task, timeout=2.0)
            catcher._config.claims.release(claim)
            return queued_on, adapter

    queued_on, adapter = asyncio.run(scenario())
    assert queued_on == ["hci5"]               # we were queued elsewhere
    assert adapter == "hci6"
    assert _our_waits(env.dir) == []


def test_a_scan_never_queues_on_a_draining_card_and_wakes_on_its_release(env, monkeypatch):
    """R3 for scans: the sole candidate is draining, so no ticket is written
    at all - a queue on a card being emptied is work waiting to land in the
    blast radius. The wait rides out the drain and takes the card the moment
    the resetter lets go."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    drain = _foreign_file(env.dir, "hci5.drain")

    async def scenario():
        with _short_wait(monkeypatch):
            task = await _start_waiting()
            assert not task.done(), "started on a draining card"
            assert _our_waits(env.dir) == [], "queued on a draining card"
            os.unlink(drain)
            adapter, claim = await asyncio.wait_for(task, timeout=2.0)
            catcher._config.claims.release(claim)
            return adapter

    assert asyncio.run(scenario()) == "hci5"


def test_a_waiter_migrates_off_a_card_that_starts_draining(env, monkeypatch):
    """Already queued when the drain appears: hci6's queue is longer, and it
    still wins, because the draining card's queue can never advance."""
    env.install(adapters=("hci5", "hci6"), wrap_scanner=True)
    _foreign_file(env.dir, "hci5.scan")
    _foreign_file(env.dir, "hci6.scan")
    _foreign_wait(env.dir, "hci6", 1)
    _foreign_wait(env.dir, "hci6", 2)

    async def scenario():
        with _short_wait(monkeypatch):
            task = await _start_waiting()
            before = [n.partition(".")[0] for n in _our_waits(env.dir)]
            _foreign_file(env.dir, "hci5.drain")
            await asyncio.sleep(0.15)          # poll ticks: migration is one
            after = [n.partition(".")[0] for n in _our_waits(env.dir)]
            await _abandon(task)
            return before, after

    before, after = asyncio.run(scenario())
    assert before == ["hci5"]                  # the shorter queue, at first
    assert after == ["hci6"]                   # then anywhere but the drain


def test_a_stolen_hard_claim_stops_the_scan_and_rejoins_the_queue(env):
    """The catcher half of the 0.5 ownership check: a claim proved lost means
    this scanner is scanning a card it does not hold, which is the state the
    gate exists to make impossible. It cannot carry on - the holder is
    entitled to that radio - so it restarts through the normal gate."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    restarted = []

    async def scenario():
        scanner = sys.modules["bleak"].BleakScanner()
        await scanner.start()
        claim = scanner._catcher_claim
        os.unlink(claim.path)                  # somebody takes the name
        other = catcher.claims.ClaimManager(owner="rival", claim_dir=env.dir)
        stolen = other.claim_hard("hci5")
        catcher._config.claims._beat_once()    # the beat notices
        assert claim.lost is True
        scanner._drain_restart = lambda: restarted.append(True) or _noop()
        catcher._lost_claim_watch()
        await asyncio.sleep(0)                 # let call_soon_threadsafe run
        await asyncio.sleep(0)
        other.release(stolen)
        await scanner.stop()

    async def _noop():
        return None

    asyncio.run(scenario())
    assert restarted == [True]


def test_a_cycle_only_proceeds_from_a_card_that_emptied_voluntarily(env, monkeypatch):
    """R4 rests entirely on R2, so the two are worth asserting together and
    end to end: with a live foreign claim on the card, the whole
    strike-drain-cycle ladder runs to the drain and stops there. This is what
    makes default-ON defensible - not that cycling became harmless, but that
    it can no longer reach a card anybody is on."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)
    monkeypatch.setattr(catcher.recovery, "_DRAIN_POLL", 0.02)
    _foreign_file(env.dir, "hci5.link.0")          # somebody is connected here
    native = []
    monkeypatch.setattr(catcher.recovery, "HAS_AUTO_RECOVERY", False)

    async def fake_native(dev_id, adapter, gone_silent):
        native.append(adapter)
        return True

    monkeypatch.setattr(catcher.recovery, "_native_recover", fake_native)

    async def scenario():
        return await catcher.recovery.reset_adapter(
            "hci5", claims_manager=catcher._config.claims, gone_silent=True, drain_timeout=0.1
        )

    assert asyncio.run(scenario()) is False
    assert native == [], "cycled a card with a live foreign link claim"
    assert "hci5.drain" not in os.listdir(env.dir)


def _drop_link(client):
    """A spontaneous drop: the backend notices, then bleak fires the raw
    disconnected callback - the ordering the release path keys on."""
    client._backend.is_connected = False
    RECORDED_INITS[-1]["disconnected_callback"](client)


def test_a_quick_redrop_is_a_failure_not_a_success(env):
    """Field 2026-08-30, prod RS pack 53:20:B7:D7:F9:E7: connected :24,
    dropped :28, connected :04, dropped :06 - ping-pong through an RF
    storm, because only a FAILED establish advanced rotation/backoff and
    each 4-second "success" wiped the adapter's record and re-armed the
    same path. A link that dies within QUICK_REDROP_WINDOW of establish
    never really connected."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    before = catcher._rotation.index(ADDRESS)

    _drop_link(client)          # lived ~0s: spontaneous, young

    akey = catcher._address_key(ADDRESS)
    key = (catcher.claims.adapter_key("hci5"), akey)
    assert catcher._connect_failures.get(key) == 1
    assert catcher._quick_drop_streaks.get(akey) == 1
    assert catcher._rotation.index(ADDRESS) == before + 1


def test_a_stable_links_drop_clears_the_record(env):
    """The other half of the verdict: a link that survived the window
    proves the path, whoever ends it - its drop clears ledger and streak."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    akey = catcher._address_key(ADDRESS)
    key = (catcher.claims.adapter_key("hci5"), akey)
    catcher._connect_failures[key] = 2
    catcher._quick_drop_streaks[akey] = 2
    before = catcher._rotation.index(ADDRESS)

    client._catcher_established_at -= catcher.QUICK_REDROP_WINDOW + 50
    _drop_link(client)

    assert key not in catcher._connect_failures
    assert akey not in catcher._quick_drop_streaks
    assert catcher._rotation.index(ADDRESS) == before


def test_a_requested_disconnect_is_never_a_quick_redrop(env):
    """shyion's whole duty cycle is 3-8s connect/read/disconnect polls.
    Charging a short link WE chose to end would poison a healthy card's
    placement on every poll - the discriminator is who ended it, and
    disconnect() already marks that (settled)."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)

    async def poll():
        await client.connect()
        await client.disconnect()          # 0s link, ended by us

    asyncio.run(poll())
    # a late disconnect event after our own teardown changes nothing
    RECORDED_INITS[-1]["disconnected_callback"](client)

    akey = catcher._address_key(ADDRESS)
    assert catcher._connect_failures == {}
    assert akey not in catcher._quick_drop_streaks


def test_quick_drops_price_the_next_reconnect(env, monkeypatch):
    """The monitor's RF analysis: 267/284 disconnects were supervision
    timeouts in correlated storms - rotation almost never helps, so the
    PRIMARY effect must be backoff, or storms just ping-pong across both
    cards. Escalating, capped, and priced BEFORE claim acquisition."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    delays = []
    real_sleep = asyncio.sleep

    async def recording_sleep(d, *a, **k):
        delays.append(d)
        await real_sleep(0)

    monkeypatch.setattr(catcher.asyncio, "sleep", recording_sleep)

    async def storm():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        for _ in range(6):
            await client.connect()
            _drop_link(client)

    asyncio.run(storm())
    assert delays == [2.0, 4.0, 8.0, 16.0, 30.0]      # first connect free, then 2^n capped


def test_unknownobject_start_notify_retries_in_place(env, monkeypatch):
    """~8x/day on prod: StartNotify hits a stale BlueZ GATT object path
    after a reconnect and the pack migrated adapters over a local cache
    artifact. The remedy is re-resolve and retry once on the SAME adapter -
    if the retry lands, no failure ever surfaces to the driver."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    refreshed = []

    async def fake_refresh(client):
        refreshed.append(client)
        return True

    monkeypatch.setattr(catcher.validators, "refresh_services", fake_refresh)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        seen = []
        fails = [catcher.BleakDBusError("org.freedesktop.DBus.Error.UnknownObject", ["stale path"])]

        async def flaky(char_specifier, callback, **kwargs):
            seen.append(char_specifier)
            if fails:
                raise fails.pop(0)

        client._backend.start_notify = flaky

        class FakeChar:
            handle = 17

        await client.start_notify(FakeChar(), lambda *_: None)
        return client, seen

    client, seen = asyncio.run(scenario())
    assert len(refreshed) == 1, "service table was not re-read"
    assert len(seen) == 2 and seen[1] == 17, "retry must go by handle, not the stale object"
    assert client._catcher_cache_fault is False


def test_a_failed_cache_retry_is_charged_to_the_cache_not_the_card(env, monkeypatch):
    """If the retry fails too it is a real failure - but a cache-shaped
    one. The teardown that follows must not read as a quick redrop, or
    fix 2's retry gets eaten by fix 1's damping (the interaction the
    integration chat flagged)."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def fake_refresh(client):
        return True

    monkeypatch.setattr(catcher.validators, "refresh_services", fake_refresh)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()

        async def always_stale(char_specifier, callback, **kwargs):
            raise catcher.BleakDBusError("org.freedesktop.DBus.Error.UnknownObject", ["stale path"])

        client._backend.start_notify = always_stale
        with pytest.raises(catcher.BleakDBusError):
            await client.start_notify("fff4", lambda *_: None)
        return client

    client = asyncio.run(scenario())
    assert client._catcher_cache_fault is True

    _drop_link(client)          # driver tears down moments later
    assert catcher._connect_failures == {}, "cache fault charged to the adapter"
    assert catcher._quick_drop_streaks == {}


def test_an_int_handle_specifier_passes_through_the_cache_retry_unchanged(env, monkeypatch):
    """jkbms_brn ships CHAR_HANDLE_FAILOVER = 4 - a bare int handle as
    char_specifier. An int has no .handle attribute, so the retry-by-handle
    branch must leave it alone: pinned here at the integration chat's ask,
    because a specifier-type surprise in the retry path would only surface
    on the one driver that uses ints, on hardware nobody runs day-to-day."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def fake_refresh(client):
        return True

    monkeypatch.setattr(catcher.validators, "refresh_services", fake_refresh)

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        seen = []
        fails = [catcher.BleakDBusError("org.freedesktop.DBus.Error.UnknownObject", ["stale path"])]

        async def flaky(char_specifier, callback, **kwargs):
            seen.append(char_specifier)
            if fails:
                raise fails.pop(0)

        client._backend.start_notify = flaky
        await client.start_notify(4, lambda *_: None)
        return seen

    seen = asyncio.run(scenario())
    assert seen == [4, 4], "an int handle must be retried as itself"



def _live_claim_file(claim_dir, name):
    """A claim file that reads as LIVE to every liveness test: this pid,
    fresh mtime. Stands in for some other process's link on the card."""
    with open(os.path.join(claim_dir, name), "w") as f:
        f.write(f"{os.getpid()} some-other-service {int(time.time())}\n")


def test_a_card_carrying_links_is_not_drained_or_cycled(env, monkeypatch):
    """The other half of "only when empty": a card that has earned its
    three strikes but carries a live link is not even drained - draining
    would steer every new placement off a card somebody is using for 60s
    and then be vetoed anyway. Placement already scores it last."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    attempts = []

    async def fake_reset(adapter, claims_manager=None, force=False, gone_silent=False, **kw):
        attempts.append(adapter)
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    _live_claim_file(env.dir, f"{catcher.claims.adapter_key('hci5')}.link.0")

    async def scenario():
        for i in range(catcher.SCAN_FAILURES_BEFORE_RESET):
            if i == catcher.SCAN_FAILURES_BEFORE_RESET - 1:
                key = catcher.claims.adapter_key("hci5")
                catcher._scan_failure_since[key] -= catcher.RECOVERY_STRIKE_SPAN + 1
            SCANNER_START_RESULTS.append(RuntimeError("Set scan parameters failed"))
            scanner = sys.modules["bleak"].BleakScanner()
            with pytest.raises(RuntimeError):
                await scanner.start()
        if catcher._recovery_tasks:
            await asyncio.gather(*list(catcher._recovery_tasks), return_exceptions=True)

    asyncio.run(scenario())
    assert attempts == [], "cycled a card that was carrying a live link"


def test_an_acquire_notify_opt_out_is_overridden_to_start_notify(env):
    """Fleet policy: nobody opts out of StartNotify. bleak defaults to it;
    AcquireNotify is reachable only via bluez={"use_start_notify": False},
    and it is the only path that creates the notify_io BlueZ 5.72
    double-frees - the fleet root cause."""
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        seen = {}

        async def capture(char_specifier, callback, **kwargs):
            seen.update(kwargs)

        client._backend.start_notify = capture
        theirs = {"use_start_notify": False, "other": 1}
        await client.start_notify("fff4", lambda *_: None, bluez=theirs)
        return seen, theirs

    seen, theirs = asyncio.run(scenario())
    assert seen["bluez"]["use_start_notify"] is True
    assert seen["bluez"]["other"] == 1                 # the rest of their args survive
    assert theirs["use_start_notify"] is False         # their dict was copied, not mutated


def test_no_bluez_args_means_none_are_invented(env):
    env.install(adapters=("hci5",), link_caps={"hci5": 2})

    async def scenario():
        client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
        await client.connect()
        seen = {}

        async def capture(char_specifier, callback, **kwargs):
            seen.update(kwargs)

        client._backend.start_notify = capture
        await client.start_notify("fff4", lambda *_: None)
        return seen

    assert "bluez" not in asyncio.run(scenario())


def test_a_device_outside_the_allow_set_is_refused_before_anything_is_spent(env):
    """THE RULE from the fleet root cause: a BLE peer set is bounded by
    stored configuration, never by radio range. sensors-py connected to
    ~63 uninvited devices; the volume detonated a dormant BlueZ UAF. With
    an allow-set in force, an unlisted device is refused before a claim is
    taken or a backend built - and with a NON-BleakError, so
    bleak-retry-connector does not burn four attempts on a policy."""
    catcher.install_bleak_catcher(OWNER, adapters=("hci5",), claim_dir=env.dir, tune_conn_params=False,
                                  allowed_devices=("11:22:33:44:55:66",))
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    with pytest.raises(catcher.DeviceNotPermitted):
        asyncio.run(client.connect())
    assert os.listdir(env.dir) == []                           # nothing spent on the card
    assert not issubclass(catcher.DeviceNotPermitted, catcher.BleakError)


def test_the_allow_set_matches_any_spelling_and_cannot_be_clobbered(env):
    """The two silent gate bugs a consumer already shipped: a set
    overwritten after load, and a MAC matched as a trailing segment. The
    catcher's set is a frozenset of canonical keys."""
    spelled = ADDRESS.lower().replace(":", "-")
    catcher.install_bleak_catcher(OWNER, adapters=("hci5",), claim_dir=env.dir, tune_conn_params=False,
                                  allowed_devices=f" {spelled}, not-a-mac ")
    assert isinstance(catcher._config.allowed_devices, frozenset)
    assert catcher._config.allowed_devices == frozenset({catcher.claims.mac_key(ADDRESS)})
    client = sys.modules["bleak"].BleakClient(ADDRESS, _is_retry_client=True)
    asyncio.run(client.connect())
    assert client.is_connected is True


def test_no_allow_set_means_no_gate(env):
    env.install(adapters=("hci5",), link_caps={"hci5": 2})
    assert catcher._config.allowed_devices is None
    catcher.install_bleak_catcher(OWNER, adapters=("hci5",), claim_dir=env.dir, tune_conn_params=False,
                                  allowed_devices="")
    assert catcher._config.allowed_devices is None       # empty means not configured, not deny-all



def test_recovery_probes_the_drained_card_before_cycling(env, monkeypatch):
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    monkeypatch.setattr(catcher.recovery, "is_bluetoothd_alive", lambda: True)
    seen = {}

    async def fake_reset(adapter, **kw):
        seen.update(kw)
        return False

    monkeypatch.setattr(catcher.recovery, "reset_adapter", fake_reset)
    asyncio.run(catcher._recover_adapter("hci5"))
    assert seen["probe"] is catcher._probe_adapter


def test_the_probe_passes_when_the_drained_card_advertises(env, monkeypatch):
    """Judged the way every scan here is judged: by traffic. One real
    advertisement and the card is back."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    monkeypatch.setattr(catcher, "PROBE_SECONDS", 2.0)

    async def scenario():
        task = asyncio.create_task(catcher._probe_adapter("hci5"))
        await asyncio.sleep(0.05)
        RECORDED_SCANNER_INITS[-1]["detection_callback"]("device", _adv(local_name="SmartShunt"))
        return await task

    assert asyncio.run(scenario()) is True
    assert os.listdir(env.dir) == []          # the probe's scan claim released


def test_the_probe_fails_when_the_drained_card_stays_silent(env, monkeypatch):
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    monkeypatch.setattr(catcher, "PROBE_SECONDS", 0.3)
    assert asyncio.run(catcher._probe_adapter("hci5")) is False
    assert os.listdir(env.dir) == []


def test_the_probe_fails_when_the_card_refuses_to_scan(env, monkeypatch):
    env.install(adapters=("hci5",), wrap_scanner=True)
    monkeypatch.setattr(catcher, "present_adapters", lambda: {"hci5"})
    SCANNER_START_RESULTS.append(RuntimeError("Set scan parameters failed"))
    assert asyncio.run(catcher._probe_adapter("hci5")) is False


def test_our_own_drain_does_not_block_our_probe_scan(env):
    """The probe scans the card we are draining, on purpose. A drain held
    by THIS process must not steer the explicit probe scan off it; a drain
    held by anyone else still does."""
    env.install(adapters=("hci5",), wrap_scanner=True)
    key = catcher.claims.adapter_key("hci5")
    _live_claim_file(env.dir, f"{key}.drain")                     # our pid
    snapshot = catcher._config.claims.claims()
    assert catcher._scan_placement(catcher._config, snapshot, "hci5")[1] == ["hci5"]
    with open(os.path.join(env.dir, f"{key}.drain"), "w") as f:
        f.write(f"1 other-svc {int(time.time())}\n")                # pid 1: alive, foreign
    snapshot = catcher._config.claims.claims()
    assert catcher._scan_placement(catcher._config, snapshot, "hci5")[1] == []
