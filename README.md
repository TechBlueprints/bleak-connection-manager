# bleak-connection-manager (v2)

An embeddable library that does for any [bleak](https://github.com/hbldh/bleak)
consumer what habluetooth does for Home Assistant: a per-process connection
layer injected underneath every `from bleak import BleakClient` via module
rebinding, so consumers (dbus-serialbattery drivers, vendored aiobmsble,
anything) transparently connect through claim-aware adapter selection,
per-adapter link slots, and failure-driven adapter rotation.

The v1 codebase (the BlueZ connection-lifecycle manager) lives on the
[`v1-main`](https://github.com/TechBlueprints/bleak-connection-manager/tree/v1-main)
branch with its full history; target v1 hotfixes there.

## Quick start

```python
from bleak_connection_manager import install_bleak_catcher

# BEFORE importing any library that does `from bleak import BleakClient` -
# they capture the binding at import time.
install_bleak_catcher(
    "dbus-blebattery.0",
    adapters=["C8:47:8C:11:22:33@hci0", "C8:47:8C:11:22:33@hci1", "hci2"],
    link_caps={"hci0": 5},          # opt-in; uncapped adapters are never gated
    wrap_scanner=True,              # opt-in; also route BleakScanner (see below)
)

import my_bms_library  # its BleakClient is now the routed wrapper
```

Every client the process builds afterwards routes through the catcher. The
wrapper **routes, it never retries** — retry semantics belong to whoever
drives the client (bleak-retry-connector, typically). A bare
`BleakClient.connect()` without bleak-retry-connector logs a warning, once
per address per process.

## Adapter configuration

Entries are raw strings, passed verbatim:

- `MAC@hciX` pins that device to that adapter; repeating the MAC gives an
  ordered preference list. A pinned device **never** falls back to the pool.
- Plain `hciX` entries form the shared pool for unpinned devices.

With **no adapters configured at all**, every adapter the kernel exposes is
a candidate (numeric hci order) — the pool config acts as an allowlist, and
an unconfigured install still spreads load by default.

Selection re-runs on every `connect()` call. **Pinned devices** walk their
pin list failure-driven: the per-address index advances only after a failed
connect attempt — a disconnect is not a failure (a dropped link reconnects
on the adapter it was using), and the index never resets on success; the
modulo wrap brings a recovered preferred radio back naturally. **Unpinned
devices** are placed by habluetooth-parity connect scoring: penalties for
live occupancy (soft claims from *any* process — the cross-process
generalization of habluetooth's in-progress term), per-adapter failure
counts for this address (cleared by a success there), and a last-slot
penalty; capped-full adapters rank last. Ties break by configuration order,
and all scoring state is process-wide, so fresh-client-per-attempt callers
continue where they left off.

**The configuring driver picks its placement mode.** By default
(`scan_to_score=False`) nothing scans, so the score has no RSSI base —
routing is least-used. With `scan_to_score=True` the catcher runs periodic
short active sweeps per candidate adapter (habluetooth's active-window
cadence: 10s every 300s, each sweep holding that adapter's `hciN.scan`
claim and skipping cards another live process is scanning on), and the
score gains its RSSI base with penalties charged in units of the spread
between the two best paths, exactly as habluetooth scores connection paths.

Configured adapters are filtered against what the kernel currently exposes
(`/sys/class/bluetooth`, `hciconfig` fallback), against adapters whose sysfs
MAC is all-zeros (a dead or unserved controller stays listed in sysfs
forever), and against adapters another live process is actively scanning on
(a foreign `hciN.scan` claim). Every filter falls back to the unfiltered
list rather than refusing to attempt — coordination is an optimization,
never a gate. An adapter the caller chose explicitly (`bluez={"adapter":
...}`) is never overridden.

**Resolved BLEDevices route themselves.** bleak's BlueZ backend connects
via `device.details["path"]` whenever the device carries one — the adapter
argument is only honored when it has to scan. So adapter *selection* applies
to plain-address connects; a cache-resolved `BLEDevice` (bleak-retry-
connector's `get_device`, scanner-discovered devices) already names its
adapter in its D-Bus path, and the catcher treats that as caller-explicit:
claims, cap gating, and connection-parameter tuning land on the adapter the
link will actually use. Explicit connects (both kinds) always write the soft
claim, so they still count in every process's occupancy score.

## Link slots

`link_caps` bounds **established-link capacity** per adapter (dongle limits
like the CSR8510's are undocumented, so caps are deployment config, not
discovery — bleak-retry-connector's field experience suggests ~5 for CSR
adapters and ~7 for Broadcom as starting points). A capped adapter's connections each hold a numbered exclusive
`hciN.link.<k>` claim file; when all slots are held live, selection moves to
the next eligible adapter **without** advancing the failure index. When every
eligible adapter is full, `connect()` raises `OutOfConnectionSlotsError`
(a `BleakError` whose message starts with the literal `"connection slot"`,
which bleak-retry-connector string-matches into its 4-second out-of-slots
backoff on any version), with per-adapter occupancy in the message, e.g.
`hci0 (5/5 links held)`. Slots release on `disconnect()`, on an unexpected
drop (via the disconnected callback), and on a failed connect.

## Scanning (opt-in)

With `wrap_scanner=True`, `bleak.BleakScanner` is also rebound, to an
adapter-bound, hard-claiming scanner. At `start()` it ranks the shared pool
(or, with no pool, the union of pinned adapters) by live occupancy — fewest
soft claims plus held link slots first — skips cards another live process is
scanning on, and takes the winner's exclusive `hciN.scan` claim, held per
scan activity: released at `stop()` or on a failed start. When every card is
claimed it scans on the best-ranked one anyway, unclaimed. A caller-chosen
adapter is never overridden (its claim is still taken, best effort). Opt-in
because it changes which adapter unrelated code scans on.

A wrapped scanner also carries habluetooth's watchdog: checked every 30s,
quiet past 90s → restart (which re-runs selection, so a dead card is walked
away from), quiet past 120s or never-saw-anything → a **drain-coordinated
hardware reset**. Empty advertisements don't count as signs of life.

A reset kills every link on the card, so it is never sprung on the card's
other users. The resetter takes the adapter's exclusive `hciN.drain` claim
(convention 0.3): placement in every participating process steers new work
elsewhere, and each process's drain watcher — run on its claim heartbeat —
disconnects its own clients on the card *when they have somewhere else to
go*, letting their retry loops reconnect on another adapter. A holder that
cannot move (its only working card, an operator pin, a caller-chosen
explicit adapter) stays put, and its live claims veto the reset: the
resetter waits up to `drain_timeout` (default 60s) for the card to empty
and gives up rather than pulling it out from under a holder. `force` skips
every gate, for an operator who knows the card is dead.

The reset primitive itself is native and stdlib-only — rfkill unblock,
`HCIDEVDOWN`/`HCIDEVUP` bounce over a raw `AF_BLUETOOTH` socket, and
`USBDEVFS_RESET` for a USB card that has gone silent — so no extra install
is needed. When `bluetooth-auto-recovery` (the optional `recovery` extra)
is importable it is preferred for its mgmt-socket handling; it is not
vendored because its rfkill path hard-imports GPLv3 PyRIC. After a
successful reset, `bluetoothd` is restarted if the reset killed it and
bleak's cached D-Bus manager state is invalidated so the next connect
rebuilds from `GetManagedObjects`. `reset_adapter(adapter,
claims_manager=None, force=False, gone_silent=False,
drain_timeout=DRAIN_TIMEOUT)` is exported for consumers' own recovery
paths; `drain_timeout=0` is the old immediate foreign-claims gate.

## Connection parameters

By default (`tune_conn_params=True`) each routed connect pre-seeds the
kernel with habluetooth's FAST parameters (7.5ms interval, 10s supervision
timeout) over the BlueZ management socket so they apply to the connection
being established, then relaxes to MEDIUM (8.75–11.25ms, 8s) once it's up.

**Venus OS caveat** (field, 2026-08-22, both Cerbos): the platform Python
is built without bluetooth socket support — no `socket.AF_BLUETOOTH` — so
the mgmt channel is unavailable and conn-param tuning silently no-ops
there, by design degradation. Every other AF_BLUETOOTH user in this
package carries a subprocess fallback (`adapter_mac` → hciconfig, the
reset's interface bounce → `hciconfig down/up`); the tuning path has none
because there is no subprocess equivalent for `Load Connection
Parameters`. If tuning ever matters on Venus it needs a btmgmt-binary
fallback — until then, assume it is inert on the flagship platform.
Degrades silently to a no-op wherever the mgmt channel is unavailable
(non-Linux, Python without `AF_BLUETOOTH`, no NET_ADMIN).

## Post-connect validation (optional)

A connect that returns success is not always a usable link: GATT discovery
can come back empty, the characteristic the caller needs can be missing, a
phantom connection can read as connected until the first real read fails.
v1 answered this with a `validate_connection` callback, and it is here
unchanged in meaning — `async (client) -> bool`, run once the link is up,
and **a rejection is a connection failure**: the link is disconnected, its
claims are released, the adapter takes the failure in the placement score
and the pin walk, and `ConnectionValidationError` (a `BleakError`) is
raised so the retry loop above — bleak-retry-connector — attempts again,
on the next radio. A validator that raises counts as a rejection.

Two ways in, per client and process-wide:

```python
from bleak_connection_manager import validate_char_exists

# per client: establish_connection passes surplus kwargs to the client class
client = await establish_connection(
    BleakClientWithServiceCache, device, name,
    validate_connection=validate_char_exists("6e400003-b5a3-f393-e0a9-e50e24dcca9e"),
)

# process-wide fallback: validates connections made deep inside a library
# the driver never calls directly (a vendored BMS library, say)
install_bleak_catcher("dbus-blebattery.0", validate_connection=validate_gatt_services)
```

The client kwarg wins over the installed default; `client.connect(
validate_connection=...)` wins over both. With neither, nothing is
validated and connects behave exactly as before.

The built-ins live in `bleak_connection_manager.validators` (stdlib only,
duck-typed on the client — the module imports without bleak), weakest to
strongest: `validate_gatt_services` (service table non-empty),
`validate_char_exists(uuid)`, `validate_read_char(uuid, timeout=5.0)`
(reads the characteristic, so a phantom link or dead HCI handle fails
here). Any `async (client) -> bool` of your own works just as well.

**Chips that register GATT late** (Telink-based ones notably) announce
ServicesResolved with only the Generic Attribute service present and add
the vendor services seconds later. v1 waited 2s/4s/8s for them, re-reading
the service table from BlueZ between tries, around *every* validator. The
catcher itself never retries, so here that wait is explicit — wrap the
validator in it where you want v1's behaviour:

```python
from bleak_connection_manager import tolerate_late_gatt, validate_char_exists

validate_connection=tolerate_late_gatt(validate_char_exists(UUID))   # 2s, 4s, 8s
```

It gives up early if the link drops while waiting, and `refresh_services(
client)` — the cache-bypassing GATT re-read it uses — is exported for
validators that want to do their own waiting. Validation runs inside
`connect()`, so whatever it spends comes out of bleak-retry-connector's
60-second per-attempt safety timeout — budget slow validators accordingly.

## The claims layer

Coordination uses the bt-claims file convention under `/run/bt-claims`
(tmpfs): heartbeated `<pid> <service> <since>` files, live = pid alive AND
mtime fresh, anyone may reap a file failing both. Kinds: `hciN.scan` (hard,
exclusive), `hciN.use.<owner>[.<qualifier>]` (soft, ranks placement),
`hciN.link.<k>` (numbered exclusive slots), `hciN.drain` (exclusive,
convention 0.3: "this card is about to be reset — place elsewhere, and
move your links off it if you can").

A connection's claims are tied to the truth of its link, not to the
wrapper object: once connected they are re-checked on every heartbeat and
released if the link is gone, so a torn-down D-Bus watch or an abandoned
client frees its slot within a TTL instead of at process exit. Link truth
is *observed traffic* — a notification arriving, or a `read_gatt_char` /
`write_gatt_char` returning — which outvotes a `is_connected` that reads
False on a broken D-Bus view. The same signal re-acquires claims that were
lost while the link lived, so a polling consumer that never subscribes to
notifications recovers on its next poll.

`bleak_connection_manager.claims` is deliberately standalone — stdlib only,
no bleak, vendorable verbatim — so services that only want adapter
coordination can use it (or copy it) without any of the bleak machinery:

```python
from bleak_connection_manager.claims import ClaimManager

manager = ClaimManager(owner="my-scanner")
claim = manager.claim_hard("hci0")     # exclusive scan claim, or None
adapter, soft = manager.choose(["hci0", "hci1"])  # ranked placement
```

The upstream bt-claims reference library (convention 0.1) is also vendored
verbatim in [`ext/bt_claims.py`](ext/bt_claims.py) for anyone who wants the
plain claims coordination without adopting any of this package — see
[`ext/README.md`](ext/README.md) for provenance.

## Public API

| Name | Purpose |
| --- | --- |
| `install_bleak_catcher(owner, adapters=(), link_caps=None, claim_dir="/run/bt-claims", wrap_scanner=False, tune_conn_params=True, scan_to_score=False, validate_connection=None)` | Rebind `bleak.BleakClient` and, when importable, `bleak_retry_connector.BleakClient` / `.BleakClientWithServiceCache`; with `wrap_scanner=True`, also `bleak.BleakScanner`. Idempotent. The pid is appended to `owner` in claim files. |
| `reset_adapter(adapter, claims_manager=None, force=False, gone_silent=False, drain_timeout=DRAIN_TIMEOUT)` | Drain-coordinated hardware reset, stdlib-native (the `recovery` extra is optional and preferred when installed). Takes the `hciN.drain` claim, waits for holders to migrate, refuses if foreign claims remain at the deadline; `drain_timeout=0` is an immediate foreign-claims gate. |
| `uninstall_bleak_catcher()` | Restore the originals and release all held claims. |
| `BLEConnection` | The wrapper client (subclass of `BleakClient`, deferred init, routes at `connect()`). |
| `BLEConnectionWithServiceCache` | Adds bleak-retry-connector's service-cache surface (`set_cached_services` no-op, guarded `clear_cache`). |
| `BLEScanner` | The adapter-bound, hard-claiming scanner (deferred init, routes and claims at `start()`). |
| `OutOfConnectionSlotsError` | `BleakError` subclass raised when every eligible adapter's cap is fully claimed. |
| `ConnectionValidationError` | `BleakError` subclass raised when `validate_connection` rejects a link (after it has been torn down). |
| `validate_gatt_services`, `validate_char_exists(uuid)`, `validate_read_char(uuid, timeout=5.0)` | Built-in post-connect validators (`bleak_connection_manager.validators`). |
| `tolerate_late_gatt(validator, waits=(2.0, 4.0, 8.0))`, `refresh_services(client)` | Wrap a validator to wait out late-registering GATT services; the cache-bypassing service re-read it uses. |

## Tests

```bash
python -m pytest tests/
```

No bleak install and no hardware needed: the suites run against an in-memory
bleak 3.x-shaped stub and a tmpdir claim directory.
