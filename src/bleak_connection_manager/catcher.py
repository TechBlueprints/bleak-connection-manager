# -*- coding: utf-8 -*-
"""Process-wide bleak client routing: the bleak catcher.

The same mechanism Home Assistant uses to put habluetooth underneath every
BLE integration: rebind bleak.BleakClient, process wide, to a wrapper that
routes connections through claim-aware adapter selection, per-adapter link
slots and failure-driven adapter rotation. Libraries that build their own
clients pick up the wrapper through their own `from bleak import
BleakClient`, provided install_bleak_catcher() runs before they are
imported. bleak_retry_connector is invariably imported before the catcher
installs, so its internal binding is permanently stale; its module
attributes (BleakClient, BleakClientWithServiceCache) are rebound too, which
covers the dominant establish_connection(BleakClientWithServiceCache, ...)
pattern.

The wrapper ROUTES; it does not retry. Retry semantics belong to whoever
drives the client (bleak-retry-connector above it), exactly as in Home
Assistant. Nesting retry inside the wrapper would multiply every caller's
retry budget.

This module's own name bindings were taken at import time, before any
install, so everything constructed internally uses the original classes and
can never recurse into the wrapper.
"""

import logging
import os
import re
import subprocess

from bleak import BleakClient as _ORIGINAL_BLEAK_CLIENT
from bleak import BleakScanner as _ORIGINAL_BLEAK_SCANNER
from bleak.exc import BleakError

from .claims import CLAIM_DIR, ClaimManager

try:
    import bleak_retry_connector as _brc_module

    _ORIGINAL_BRC_CLIENT = _brc_module.BleakClient
    _ORIGINAL_BRC_CACHE_CLIENT = _brc_module.BleakClientWithServiceCache
except (ImportError, AttributeError):
    _brc_module = None
    _ORIGINAL_BRC_CLIENT = None
    _ORIGINAL_BRC_CACHE_CLIENT = None

logger = logging.getLogger(__name__)


class OutOfConnectionSlotsError(BleakError):
    """Every eligible adapter's configured link capacity is fully claimed.

    The message starts with the literal substring "connection slot", which
    bleak-retry-connector string-matches (OUT_OF_SLOTS_ERRORS) to apply its
    4s out-of-slots backoff on any version, regardless of exception
    identity. It is raised before any backend is constructed and costs file
    stats only: brc draws it from the transient budget, so one
    establish_connection call may absorb it up to ~9 times.
    """


def parse_adapter_entries(entries):
    """
    Split configured adapter entries into pinned devices and the shared pool.

    Entries of the form MAC@hciX pin that device to that adapter, with no
    fallback to the shared pool. Repeating a MAC pins it to several adapters
    in order: the first is used for every connection attempt, the rest are
    tried only when it cannot be resolved there. That keeps a battery on a
    known good radio while leaving it somewhere to go if that radio fails,
    without returning it to a pool shared with other devices.

    Plain hciX entries form the pool used by every device that is not pinned.
    Returns (pins, pool), with pins keyed by upper case MAC address and each
    value a list of adapters in priority order.
    """
    pins = {}
    pool = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if "@" in entry:
            mac, _, adapter = entry.rpartition("@")
            mac = mac.strip().upper()
            adapter = adapter.strip()
            if mac and adapter:
                adapters = pins.setdefault(mac, [])
                if adapter not in adapters:
                    adapters.append(adapter)
            else:
                logger.warning(f"Ignoring malformed adapter entry '{entry}'")
        else:
            pool.append(entry)
    return pins, pool


def present_adapters():
    """Adapter names the kernel currently exposes, or an empty set for "no
    answer".

    hciN names are not stable identities: a USB reset or reboot renumbers
    them, and an adapter a device is configured for can stop existing while
    its number lives on pointing at different hardware. /sys/class/bluetooth
    is the kernel's view - an adapter can be present there while bluetoothd
    is not serving it; such an adapter fails its connect and consumes a walk
    step, acceptable because the walk is failure-driven. hciconfig is the
    fallback, for parity with the dbus-serialbattery fork's production code.
    """
    try:
        names = {name for name in os.listdir("/sys/class/bluetooth") if name.startswith("hci")}
    except OSError:
        names = set()
    if names:
        return names
    try:
        result = subprocess.run(["hciconfig"], capture_output=True, text=True, timeout=5)
        return set(re.findall(r"^(hci\d+):", result.stdout, re.MULTILINE))
    except Exception:
        return set()


def _address_key(address):
    return str(address).strip().upper()


def _mac_qualifier(address):
    return _address_key(address).replace(":", "")


class BleAdapterRotation:
    """Failure-driven adapter walk state for clients the catcher builds.

    The per-address index advances only when a connect attempt fails - a
    disconnect is not a failure, so a dropped link reconnects on the adapter
    it was using - and never resets on success: once a device is talking
    over an adapter there is no reason to re-probe a preferred one that may
    be gone, and the modulo brings the list round to it again if this one
    later fails. State is per address and process wide, because one device's
    client may be rebuilt many times per session (bleak's context managers,
    bleak-retry-connector's attempts) and each rebuild must continue the
    walk, not restart it.
    """

    def __init__(self):
        self._attempt_index = {}

    def index(self, address):
        return self._attempt_index.get(_address_key(address), 0)

    def connect_failed(self, address):
        key = _address_key(address)
        self._attempt_index[key] = self._attempt_index.get(key, 0) + 1


_rotation = BleAdapterRotation()

# addresses already warned about bare connect() calls, once per process -
# a reconnect loop hitting this every few seconds would flood the log
_warned_bare_connect_addresses = set()


class _CatcherConfig:
    def __init__(self, owner, pins, pool, link_caps, claims):
        self.owner = owner
        self.pins = pins
        self.pool = pool
        self.link_caps = link_caps
        self.claims = claims


_config = None


def _out_of_slots_error(address, exhausted, config):
    """The typed exhaustion error, with per-adapter occupancy: when this
    fires on a GX device the occupancy detail is the whole diagnosis."""
    snapshot = config.claims.claims()
    detail = ", ".join(f"{adapter} ({(snapshot.get(adapter) or {}).get('links', cap)}/{cap} links held)" for adapter, cap in exhausted)
    return OutOfConnectionSlotsError(f"connection slot exhausted for {address}: {detail}")


def _acquire_adapter(address):
    """Select an adapter for the next attempt and take its claims.

    Returns (adapter-or-None, claims-held). Selection: the device's pins or
    the shared pool, filtered by kernel presence, then by foreign live scan
    claims - each filter falling back to the unfiltered list rather than
    refusing to attempt - walked from the failure-driven index. An adapter
    whose configured link capacity is fully claimed is skipped WITHOUT
    advancing the index: exhaustion says nothing about the radio, and
    advancing would let a busy adapter push a device off its pin. Raises
    OutOfConnectionSlotsError only when every eligible adapter is capped and
    full.
    """
    config = _config
    if config is None:
        return None, []
    adapters = config.pins.get(_address_key(address)) or config.pool
    if not adapters:
        return None, []
    adapters = list(adapters)
    present = present_adapters()
    usable = [a for a in adapters if a in present] if present else adapters
    if not usable:
        # refusing to attempt is worse than trying an adapter that may be gone
        usable = adapters
    snapshot = config.claims.claims()
    own_pid = os.getpid()

    def foreign_scan(adapter):
        entry = snapshot.get(adapter)
        return bool(entry and entry["hard"] and entry["hard_pid"] != own_pid)

    eligible = [a for a in usable if not foreign_scan(a)]
    if not eligible:
        logger.info(f"BLE [{address}]: every usable adapter is scan-claimed by another process, using them anyway")
        eligible = usable
    start = _rotation.index(address) % len(eligible)
    exhausted = []
    for adapter in eligible[start:] + eligible[:start]:
        cap = config.link_caps.get(adapter)
        if cap:
            slot = config.claims.claim_slot(adapter, cap)
            if slot is None:
                exhausted.append((adapter, cap))
                continue
        else:
            slot = None
        soft = config.claims.claim_soft(adapter, qualifier=_mac_qualifier(address))
        return adapter, [c for c in (slot, soft) if c is not None]
    raise _out_of_slots_error(address, exhausted, config)


def _claim_explicit(adapter, address):
    """Claims for an adapter the caller chose explicitly.

    The choice is never overridden, but a configured link cap still gates
    it: the cap is physics, not coordination - the connect is doomed when
    the card is full, and the typed error buys correct pacing from
    bleak-retry-connector. Defensible precisely because caps are opt-in:
    with no cap configured this is byte-for-byte plain bleak behavior.
    """
    config = _config
    if config is None:
        return []
    cap = config.link_caps.get(adapter)
    if not cap:
        return []
    slot = config.claims.claim_slot(adapter, cap)
    if slot is None:
        raise _out_of_slots_error(address, [(adapter, cap)], config)
    soft = config.claims.claim_soft(adapter, qualifier=_mac_qualifier(address))
    return [c for c in (slot, soft) if c is not None]


class BLEConnection(_ORIGINAL_BLEAK_CLIENT):
    """Drop-in BleakClient that picks its adapter and claims at connect time.

    Construction stores the arguments and builds nothing, because the
    adapter can only be chosen when the connection is actually attempted -
    bleak wires the adapter into its platform backend inside __init__, and
    callers like aiobmsble construct placeholder clients long before they
    connect. connect() runs the real __init__ with the routed adapter merged
    into the bluez args, then delegates; every reconnect on the same
    instance re-runs the selection, which is what lets a retrying caller
    walk on to the next radio after a failure. Every other bleak method
    works via inheritance - they all delegate to self._backend.
    """

    def __init__(self, address_or_ble_device, disconnected_callback=None, services=None, **kwargs):
        # bleak-retry-connector marks every client its establish_connection
        # constructs; habluetooth's wrapper checks the same marker. Without
        # it, connect() below is a single unretried attempt. Popped, never
        # passed down: the real __init__ does not know it.
        self._catcher_is_retry_client = kwargs.pop("_is_retry_client", False)
        self._catcher_args = (address_or_ble_device, disconnected_callback, services)
        self._catcher_kwargs = kwargs
        self._catcher_address = getattr(address_or_ble_device, "address", address_or_ble_device)
        self._catcher_claims = []
        self._catcher_manager = None
        self._backend = None
        self._pair_before_connect = False

    def _release_claims(self):
        held, self._catcher_claims = self._catcher_claims, []
        manager = self._catcher_manager
        for claim in held:
            if manager is not None:
                manager.release(claim)
            else:
                claim.release()

    def _warn_bare_connect(self):
        key = _address_key(self._catcher_address)
        if key in _warned_bare_connect_addresses:
            return
        _warned_bare_connect_addresses.add(key)
        logger.warning(
            f"BLE [{self._catcher_address}]: BleakClient.connect() called without bleak-retry-connector. "
            "A bare connect() is a single attempt with no recovery; connect through "
            "bleak_retry_connector.establish_connection() instead."
        )

    def _make_disconnected_callback(self, raw_callback):
        # bleak wraps the raw callable in functools.partial(cb, self) inside
        # the real __init__, so this wrapping must happen on the raw
        # callable: claims are released before the caller hears about the
        # drop, and an unexpected drop frees its link slot.
        def _disconnected(client):
            self._release_claims()
            if raw_callback is not None:
                raw_callback(client)

        return _disconnected

    async def connect(self, **kwargs):
        if not self._catcher_is_retry_client:
            self._warn_bare_connect()
        address_or_ble_device, raw_callback, services = self._catcher_args
        init_kwargs = dict(self._catcher_kwargs)
        # a reconnect on this instance must not leak the previous claims
        self._release_claims()
        config = _config
        explicit = init_kwargs.get("adapter") or (init_kwargs.get("bluez") or {}).get("adapter")
        if explicit:
            held = _claim_explicit(explicit, self._catcher_address)
        else:
            adapter, held = _acquire_adapter(self._catcher_address)
            if adapter:
                bluez = dict(init_kwargs.get("bluez") or {})
                bluez.setdefault("adapter", adapter)
                init_kwargs["bluez"] = bluez
        self._catcher_claims = held
        self._catcher_manager = config.claims if config is not None else None
        callback = raw_callback
        if held or raw_callback is not None:
            callback = self._make_disconnected_callback(raw_callback)
        _ORIGINAL_BLEAK_CLIENT.__init__(
            self,
            address_or_ble_device,
            disconnected_callback=callback,
            services=services,
            **init_kwargs,
        )
        try:
            return await _ORIGINAL_BLEAK_CLIENT.connect(self, **kwargs)
        except Exception:
            # a failed attempt: free the claims and walk to the next adapter
            self._release_claims()
            _rotation.connect_failed(self._catcher_address)
            raise

    @property
    def is_connected(self):
        # queried by callers on never-connected placeholders (aiobmsble does)
        if self._backend is None:
            return False
        return self._backend.is_connected

    @property
    def address(self):
        if self._backend is None:
            return self._catcher_address
        return self._backend.address

    async def disconnect(self):
        if self._backend is None:
            return
        try:
            return await _ORIGINAL_BLEAK_CLIENT.disconnect(self)
        finally:
            self._release_claims()


class BLEConnectionWithServiceCache(BLEConnection):
    """bleak-retry-connector's BleakClientWithServiceCache surface on top of
    BLEConnection.

    establish_connection's BlueZ GattService1-KeyError path does
    isinstance(client, BleakClientWithServiceCache) -> await
    client.clear_cache(); the module global resolves at call time, so after
    install that check finds this class. The underlying bleak may lack
    clear_cache entirely (vendored 3.x does), hence the hasattr guard -
    a bare inherit would AttributeError on exactly that path.
    """

    def set_cached_services(self, services):
        """No-op back-compat shim, matching bleak-retry-connector's own."""

    async def clear_cache(self, *args, **kwargs):
        """Clear the device's service cache, when the underlying bleak can."""
        if hasattr(super(), "clear_cache"):
            return await super().clear_cache(*args, **kwargs)
        logger.warning("clear_cache not implemented in this version of bleak")
        return False


def _scan_candidates(config):
    """Adapters a scan may use: the shared pool, or with no pool the union
    of pinned adapters in configuration order (a scan serves discovery for
    every device, so pins only narrow it when they are all there is)."""
    if config.pool:
        return list(config.pool)
    seen = []
    for adapters in config.pins.values():
        for adapter in adapters:
            if adapter not in seen:
                seen.append(adapter)
    return seen


def _acquire_scan_adapter():
    """Choose an adapter for a scan and take its hard claim.

    Returns (adapter-or-None, Claim-or-None). Candidates are filtered by
    kernel presence (falling back to the unfiltered list), ranked by live
    occupancy - fewest soft claims plus held link slots first, configuration
    order breaking ties - and the best claimable card wins; claim_hard
    itself skips cards another live process is scanning on. When every card
    is claimed, the best-ranked one is scanned anyway, unclaimed:
    coordination is an optimization, never a gate.
    """
    config = _config
    if config is None:
        return None, None
    candidates = _scan_candidates(config)
    if not candidates:
        return None, None
    present = present_adapters()
    usable = [a for a in candidates if a in present] if present else candidates
    if not usable:
        # refusing to scan is worse than scanning an adapter that may be gone
        usable = candidates
    snapshot = config.claims.claims()

    def occupancy(adapter):
        entry = snapshot.get(adapter) or {}
        return entry.get("soft", 0) + entry.get("links", 0)

    ranked = sorted(usable, key=lambda a: (occupancy(a), usable.index(a)))
    for adapter in ranked:
        claim = config.claims.claim_hard(adapter)
        if claim is not None:
            return adapter, claim
    logger.info(f"bt-claims: every adapter is scan-claimed, scanning on {ranked[0]} unclaimed")
    return ranked[0], None


class BLEScanner(_ORIGINAL_BLEAK_SCANNER):
    """Drop-in BleakScanner that picks its adapter and takes the hard scan
    claim at start().

    Deferred init, like BLEConnection: bleak wires the adapter into its
    platform scanner backend inside __init__, so construction stores the
    arguments and builds nothing, and start() runs the real __init__ with
    the routed adapter merged in (via the backwards-compat adapter kwarg,
    which is how this bleak's BlueZ scanner backend receives it). The hard
    claim (hciN.scan) is held per scan activity: taken at start(), released
    at stop() or on a failed start - so other processes' placement steers
    connections and scans away from a card while it is actually scanning,
    and no longer. An adapter the caller chose explicitly is never
    overridden, though its claim is still taken, best effort.

    Rebinding over bleak.BleakScanner is opt-in
    (install_bleak_catcher(..., wrap_scanner=True)).
    """

    def __init__(self, detection_callback=None, service_uuids=None, scanning_mode="active", **kwargs):
        self._catcher_args = (detection_callback, service_uuids, scanning_mode)
        self._catcher_kwargs = kwargs
        self._catcher_claim = None
        self._catcher_manager = None
        self._backend = None
        self._backend_id = None

    def _release_scan_claim(self):
        claim, self._catcher_claim = self._catcher_claim, None
        if claim is None:
            return
        if self._catcher_manager is not None:
            self._catcher_manager.release(claim)
        else:
            claim.release()

    async def start(self):
        detection_callback, service_uuids, scanning_mode = self._catcher_args
        init_kwargs = dict(self._catcher_kwargs)
        # a restart on this instance must not leak the previous claim
        self._release_scan_claim()
        config = _config
        self._catcher_manager = config.claims if config is not None else None
        explicit = init_kwargs.get("adapter") or (init_kwargs.get("bluez") or {}).get("adapter")
        if explicit:
            if config is not None:
                self._catcher_claim = config.claims.claim_hard(explicit)
        else:
            adapter, claim = _acquire_scan_adapter()
            self._catcher_claim = claim
            if adapter:
                init_kwargs.setdefault("adapter", adapter)
        _ORIGINAL_BLEAK_SCANNER.__init__(
            self,
            detection_callback,
            service_uuids,
            scanning_mode,
            **init_kwargs,
        )
        try:
            return await _ORIGINAL_BLEAK_SCANNER.start(self)
        except Exception:
            self._release_scan_claim()
            raise

    async def stop(self):
        if self._backend is None:
            return
        try:
            return await _ORIGINAL_BLEAK_SCANNER.stop(self)
        finally:
            self._release_scan_claim()

    @property
    def discovered_devices(self):
        # queried on never-started placeholders; delegate once real
        if self._backend is None:
            return []
        prop = getattr(_ORIGINAL_BLEAK_SCANNER, "discovered_devices", None)
        if isinstance(prop, property):
            return prop.fget(self)
        return self._backend.discovered_devices


def install_bleak_catcher(owner, adapters=(), link_caps=None, claim_dir=CLAIM_DIR, wrap_scanner=False):
    """Route every bleak client in this process through the catcher.

    Must run before consumer libraries are imported: they capture `from
    bleak import BleakClient` at import time. A consumer that imported
    BleakClientWithServiceCache before install and passes the stale original
    as client_class silently skips cache-clear - degradation, not breakage.

    owner names this process's claims; the pid is appended to disambiguate
    restart races (the old process's claims awaiting reap while the new one
    starts). adapters are raw config strings, verbatim ("MAC@hciX" pins,
    plain "hciX" pools - see parse_adapter_entries). link_caps maps adapter
    name to its established-link capacity; caps are opt-in, an uncapped
    adapter is never slot-gated. wrap_scanner additionally rebinds
    bleak.BleakScanner to the adapter-bound, hard-claiming BLEScanner -
    opt-in because it changes which adapter unrelated code scans on.
    Idempotent.
    """
    global _config
    import bleak

    pins, pool = parse_adapter_entries(adapters)
    caps = {}
    for adapter, cap in (link_caps or {}).items():
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            cap = 0
        if cap > 0:
            caps[str(adapter).strip()] = cap
        else:
            logger.warning(f"Ignoring link cap for '{adapter}': not a positive integer")
    if _config is not None:
        _config.claims.release_all()
    _config = _CatcherConfig(
        owner=owner,
        pins=pins,
        pool=pool,
        link_caps=caps,
        claims=ClaimManager(owner=f"{owner}-{os.getpid()}", claim_dir=claim_dir),
    )
    bleak.BleakClient = BLEConnection
    bleak.BleakScanner = BLEScanner if wrap_scanner else _ORIGINAL_BLEAK_SCANNER
    if _brc_module is not None:
        _brc_module.BleakClient = BLEConnection
        _brc_module.BleakClientWithServiceCache = BLEConnectionWithServiceCache
    logger.info("bleak catcher installed: BLE clients are routed through claim-aware adapter selection")


def uninstall_bleak_catcher():
    """Restore the original classes and release every held claim."""
    global _config
    import bleak

    bleak.BleakClient = _ORIGINAL_BLEAK_CLIENT
    bleak.BleakScanner = _ORIGINAL_BLEAK_SCANNER
    if _brc_module is not None:
        _brc_module.BleakClient = _ORIGINAL_BRC_CLIENT
        _brc_module.BleakClientWithServiceCache = _ORIGINAL_BRC_CACHE_CLIENT
    if _config is not None:
        _config.claims.release_all()
        _config = None
