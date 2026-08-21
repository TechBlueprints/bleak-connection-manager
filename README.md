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

Selection re-runs on every `connect()` call. The walk is failure-driven: the
per-address index advances only after a **failed connect attempt** — a
disconnect is not a failure (a dropped link reconnects on the adapter it was
using), and the index never resets on success; the modulo wrap brings a
recovered preferred radio back naturally. Rotation state is per-address and
process-wide, so fresh-client-per-attempt callers continue the walk.

Configured adapters are filtered against what the kernel currently exposes
(`/sys/class/bluetooth`, `hciconfig` fallback) and against adapters another
live process is actively scanning on (a foreign `hciN.scan` claim). Every
filter falls back to the unfiltered list rather than refusing to attempt —
coordination is an optimization, never a gate. An adapter the caller chose
explicitly (`bluez={"adapter": ...}`) is never overridden.

## Link slots

`link_caps` bounds **established-link capacity** per adapter (dongle limits
like the CSR8510's are undocumented, so caps are deployment config, not
discovery). A capped adapter's connections each hold a numbered exclusive
`hciN.link.<k>` claim file; when all slots are held live, selection moves to
the next eligible adapter **without** advancing the failure index. When every
eligible adapter is full, `connect()` raises `OutOfConnectionSlotsError`
(a `BleakError` whose message starts with the literal `"connection slot"`,
which bleak-retry-connector string-matches into its 4-second out-of-slots
backoff on any version), with per-adapter occupancy in the message, e.g.
`hci0 (5/5 links held)`. Slots release on `disconnect()`, on an unexpected
drop (via the disconnected callback), and on a failed connect.

## The claims layer

Coordination uses the bt-claims file convention under `/run/bt-claims`
(tmpfs): heartbeated `<pid> <service> <since>` files, live = pid alive AND
mtime fresh, anyone may reap a file failing both. Kinds: `hciN.scan` (hard,
exclusive), `hciN.use.<owner>[.<qualifier>]` (soft, ranks placement),
`hciN.link.<k>` (numbered exclusive slots).

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
| `install_bleak_catcher(owner, adapters=(), link_caps=None, claim_dir="/run/bt-claims")` | Rebind `bleak.BleakClient` and, when importable, `bleak_retry_connector.BleakClient` / `.BleakClientWithServiceCache`. Idempotent. The pid is appended to `owner` in claim files. |
| `uninstall_bleak_catcher()` | Restore the originals and release all held claims. |
| `BLEConnection` | The wrapper client (subclass of `BleakClient`, deferred init, routes at `connect()`). |
| `BLEConnectionWithServiceCache` | Adds bleak-retry-connector's service-cache surface (`set_cached_services` no-op, guarded `clear_cache`). |
| `OutOfConnectionSlotsError` | `BleakError` subclass raised when every eligible adapter's cap is fully claimed. |

`BleakScanner` wrapping (an adapter-bound, hard-claiming scanner) is
deliberately deferred to a later phase.

## Tests

```bash
python -m pytest tests/
```

No bleak install and no hardware needed: the suites run against an in-memory
bleak 3.x-shaped stub and a tmpdir claim directory.
