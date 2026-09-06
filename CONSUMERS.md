# How a service uses BCM

The standing contract between bleak-connection-manager and every service
that uses it. BCM and all of its consumers are ours (dbus-shyion-switch,
venus-os-dbus-ble-sensors-py, dbus-power-watchdog, dbus-easytouchrv,
venus-os_dbus-serialbattery), so there is one way to do this and no
compatibility matrix: every service sources the shared install on the box
itself, from a folder it is configured with, and never carries a copy of
BCM. (Clint, 2026-09-06: "all things should source the bcm in the same way,
no shim"; "each user of bcm should have a simple way to configure where the
bcm is, and source it from there".)

Reference implementation of the sourcing step: `ble_stack.py` in
venus-os_dbus-serialbattery. It imports only `os` and `sys` and nothing from
its driver; every consumer lifts it as-is.

## 1. Install: the shared checkout on the box

Every consumer's installer converges the one checkout (idempotent; ff-only,
so a stale installer can never move the box backwards):

```sh
BCM_DIR=/data/bcm
if [ -d "$BCM_DIR/.git" ]; then
    git -C "$BCM_DIR" fetch -q origin && git -C "$BCM_DIR" merge -q --ff-only origin/main
else
    git clone -q https://github.com/TechBlueprints/bleak-connection-manager "$BCM_DIR"
fi
"$BCM_DIR/install.sh"
```

`install.sh` fetches the pinned submodules (bleak, bleak-retry-connector)
shallowly and by exact sha, with retries — sized for RV uplinks, ~2 MB —
then smoke-imports the whole stack and appends a line to
`/data/bcm/deploy.log` (`<ISO-UTC> install <commit> by user@ip`). If the
smoke import fails it prints the rollback command: surface that as an
installer failure. Mind the shell: piping install.sh through `tail`/`tee`
eats its exit code unless you `set -o pipefail` (BusyBox ash supports it)
or check `$PIPESTATUS`. A run killed by the link is safe to re-run.

Do NOT pass `--autowire` from a consumer installer. Autowire (the sitewide
`.pth` hook that catches processes which know nothing about BCM) is a
per-box decision made by Clint; `install.sh --autowire` handles the
read-only Venus rootfs itself and registers a boot-time replant in
`/data/rc.local` logging to `/data/bcm/autowire-replant.log`.

## 2. In process: `ensure_ble_stack()`

Signature, from the reference module:
`ensure_ble_stack(shared_dir="/data/bcm", vendored_dir=<dir or None>)
-> "provided" | "shared" | "vendored"`, with `shared_lib_paths(root)` and
`shared_install_present(root)` exposed. The root layout is pinned by a
test against `bcm_autowire._lib_paths()` in this repo, so drift between the
two fails loudly. The rules it implements:

1. **The folder is the consumer's configuration.** One key,
   `BLUETOOTH_CONNECTION_MANAGER_DIR`, in whatever config the service
   already has, default `/data/bcm`; an EMPTY value means never look.
   Nothing about how the process was launched decides where the stack
   comes from: plain `python3`, unchanged command line, no interpreter
   shim, no `PYTHONPATH`, no environment contract.

2. **Presence is the package, not the folder.**
   `<dir>/src/bleak_connection_manager` is a directory → present. An empty
   or half-cloned folder is absent.

3. **Import roots, in this order, at the FRONT of `sys.path`** (the layout
   `install.sh` writes):
   ```
   <dir>/src
   <dir>/ext
   <dir>/ext/upstream/bleak
   <dir>/ext/upstream/bleak-retry-connector/src
   ```
   Front, because a bleak or bleak-retry-connector anywhere else on the
   path (the script directory, an `ext/` insert, site-packages) would
   otherwise win. The shared install carries its own bleak, brc, dbus_fast,
   bluetooth_adapters and aiooui; a consumer never shadows them. BCM does
   not ship or import habluetooth; a service that needs it keeps its own.

4. **Import `bleak_connection_manager` BEFORE anything imports `bleak`.**
   Two reasons. Consumer libraries capture `from bleak import BleakClient`
   at module scope (aiobmsble does; so does every driver that names the
   class at import), so the catcher must be installed before they load.
   And the sitewide autowire hook, where planted, stands down for a
   process that has `bleak_connection_manager` in `sys.modules` — this
   import is what keeps autowire from installing a generic catcher with a
   cmdline-derived owner and the box-wide `autowire.conf`. It holds even
   though BCM's own catcher module imports bleak during that import:
   Python registers the package in `sys.modules` before running its body
   (pinned by `tests/test_autowire.py`).

5. **Call it unconditionally and early**, from the entry point before the
   BLE module import chain, and independently of any "manager enabled"
   flag — importability is not a feature flag. A tiny setup module the
   entry points import first (easytouch's `catcher_setup.py` shape) is the
   right place.

6. **Then install the catcher:**
   `install_bleak_catcher(owner, adapters=..., link_caps=...,
   force_start_notify=...)`. Per-process policy is passed here, from the
   consumer's own config; there is no shared or central BCM config file.
   The StartNotify policy is one key next to the location key,
   `BLUETOOTH_CONNECTION_MANAGER_FORCE_START_NOTIFY`, default `true`.
   (`install_bleak_catcher` still reads `BCM_FORCE_START_NOTIFY` from the
   environment when the argument is None, only as the legacy path. The
   `/data/bcm/python3` shim retires box by box: `install.sh` writes it only
   while some run script on that box still execs it and removes it the
   first time none does, printing which scripts still need it; the
   environment read is removed from the code once no box needs the shim.)

7. **Migrate the launcher `/service/<name>` actually resolves to, read
   on the box.** The run that `/service/<name>` resolves to is the
   launcher, whatever kind of entry it is: usually a symlink into the
   repo's root-level `service/` directory, but on prod the packs'
   `/service/dbus-blebattery.N` entries are real directories whose `run`
   lives under `/service` itself, outside `/data/apps`, regenerated only by
   the boot hook (`enable.sh --boot`), so a deploy of the app directory
   never reaches them. An in-tree `service/run` under `src/opt/...`, a
   `start-<name>.sh` beside it, or the generator template alone may never
   be what runs. sensors-py (PR #9, 2026-09-06) migrated its start
   script and left `exec /data/bcm/python3` in the run the symlink
   resolves to; only the pre-restart check caught it, and deploying as-is
   would have left prod on the shim while `ensure_ble_stack()` returned
   `provided` and looked migrated. Follow the symlink, migrate that file,
   and pin the box-side pre-restart check to the resolved file (print its
   exec line, stop on shim residue); a repo test may pin the template as
   well, never instead.

8. **On the `vendored` path, touch nothing of BCM's.** No import from
   `bleak_connection_manager` (not `DeviceNotPermitted`, not `claims`),
   and no BCM-only adapter syntax handed to plain bleak/brc: degrade
   `MAC@hciN` and MAC-keyed pins to a bare `hciN` adapter kwarg, or drop
   them. Absent means plain bleak; there is no vendored connection manager
   anywhere.

### Startup lines (part of the contract)

Anchor `BLE coordination: `. The monitor greps for these across the fleet
and the serialbattery overlay pins them verbatim, with levels, by test (an
AST test there pins every logger call in the install path to the anchor).
Raise class for monitors: no-shared-install, DIR-empty,
predates-force_start_notify, unusable, would-not-install. INFO loaded-from
and catcher-installed mean active (presence tier); manager off means
silence.

- `shared`, manager on — INFO once:
  `BLE coordination: bleak_connection_manager loaded from <dir-of-package>`
  — the imported package's own directory (`bleak_connection_manager.__file__`),
  NOT the configured key, so the line proves which tree served.
- `shared`, manager on, `install_bleak_catcher` returned — INFO once,
  immediately after the line above:
  `BLE coordination: catcher installed (force_start_notify=<True|False>,
  adapters=<n> configured, <m> pinned)`. Emitted by the CONSUMER, because
  BCM's own install-time INFO lines ("bleak catcher installed",
  "StartNotify is forced") never reach a consumer whose root logger sits
  at WARNING, and the ruling is not to open BCM's logger in consumer logs;
  this is how the install and the StartNotify policy are visible per
  consumer. Presence-tier for monitors.
- `vendored`, no shared install — WARNING once:
  `BLE coordination: no shared install at <DIR>; running uncoordinated, no
  claims, no adapter routing, no card recovery`.
- coordination requested but nowhere to look (manager on, location key
  empty) — WARNING once:
  `BLE coordination: BLUETOOTH_CONNECTION_MANAGER is on but
  BLUETOOTH_CONNECTION_MANAGER_DIR is empty; running uncoordinated, no
  claims, no adapter routing, no card recovery`. (A consumer whose enable
  and location are one key collapses this into the line above.)
- present but broken (folder there, import raised) — ERROR once:
  `BLE coordination: shared install at <DIR> is present but unusable,
  running uncoordinated: <repr(exc)>`. The module has already withdrawn
  every path and module the folder contributed before returning. "No
  shared install" and "shared install unusable" are different operator
  actions; never log them alike.
- shared install older than 159536a (2026-09-02), i.e.
  `install_bleak_catcher` lacks `force_start_notify=` — WARNING once:
  `BLE coordination: shared install at <DIR> predates the
  force_start_notify parameter; StartNotify policy passed through the
  legacy BCM_FORCE_START_NOTIFY environment`. The consumer checks the
  signature (`inspect.signature`) and sets the environment variable
  instead. Operator action: update the install. The monitor treats this
  line as a raise.
- import succeeded but `install_bleak_catcher` raised (bad kwarg, a
  raising validator, a catcher bug) — ERROR once:
  `BLE coordination: catcher would not install from <DIR>, running
  uncoordinated: <repr(exc)>`. Operator action: fix the driver or the
  catcher, NOT the install; deliberately distinct from "present but
  unusable", which is the install.
- manager deliberately off — no line at all. The consumer still calls
  `ensure_ble_stack()` and imports `bleak_connection_manager` (rule 4) but
  installs no catcher, and emits no INFO line, because on a box with
  autowire planted that line would read as coordination-active for a
  process that has none.

## 3. Nothing of BCM is vendored, ever

No copy of bleak-connection-manager lives in any consumer's tree, in any
form (submodule, overlay, copied `src/`), and no `sys.path` insertion
points at one. A service that must stay runnable standalone for third
parties may keep vendored `bleak`, `bleak_retry_connector`, `dbus_fast`,
`bluetooth_adapters`, `aiooui` for the `vendored` state only, in a
directory that ONLY `ensure_ble_stack()` inserts, after the shared install
has been found absent — never before, never alongside.

**If you keep a vendored fallback, keep it CURRENT.** Found the hard way
(2026-08-25): a consumer whose installer treats convergence as non-fatal
has built a path that is taken precisely when something has already gone
wrong, and if the vendored copy is stale the box quietly runs the old
stack as a consequence of the failure. Either keep both paths current with
a mechanism (the installer compares the two shas after converging and
reports when the vendored copy is *behind*, with the distance; ahead or
diverged is a deliberate pin) or make convergence fatal. Build that check's
fixture as a **submodule, not a `git clone`**: in a submodule `.git` is a
file, so a `[ -d "$dir/.git" ]` guard is false on every real deployment and
a clone-based fixture agrees with it for the wrong reason (found dead in
production while its test passed, 2026-08-25). Ask git
(`git -C "$dir" rev-parse --git-dir`), and have the fixture assert it
produced a `.git` file. Run every such test against the broken version
first.

**Check your dependency-update script.** A script that refreshes `ext/`
from upstream will happily repopulate the flat `ext/` you emptied, putting
bleak/brc back where a bare insert makes them win over the shared install.
serialbattery's `update.py` did; fixed by giving the BLE set its own
subdirectory (`"subdir": "ble"`) that only `ensure_ble_stack()` inserts,
pinned by test. Make your updater write only to the directory the
`vendored` state inserts, and test that the flat `ext/` stays empty of the
BLE set.

Why the earlier shape (a `/data/bcm/python3` interpreter shim exporting
`PYTHONPATH`, `BCM_AUTOWIRE=0` and `BCM_FORCE_START_NOTIFY`, with an
`_ensure_ble_stack` that preferred "whatever the interpreter already
provides" and fell back to `ext/`) is retired: it made the launcher decide
where the stack came from, so a launcher that did not exec the shim on a
box with autowire planted was autowired at `import bleak` with the wrong
owner and no pins (prod, 2026-08-22, `owner=autowire--c bleak=own`), and a
consumer's own path insert could shadow the shim without anyone noticing.
Rules 3 and 4 close both, and the folder test lives in the process.

## 4. Verify after a deploy

- `install.sh` printed `bcm-install: smoke import ok ... (<hash>)` and
  `/data/bcm/deploy.log` gained the line.
- The service's first log lines carry
  `BLE coordination: bleak_connection_manager loaded from
  /data/bcm/src/bleak_connection_manager`.
- `/proc/<pid>/environ` has no `PYTHONPATH`, `BCM_AUTOWIRE` or
  `BCM_FORCE_START_NOTIFY`; the command line is plain `python3 ... <main>.py`.
- Claims appear in `/run/bt-claims` under your owner; none named
  `autowire-*`.
- One INFO per process: `conn-param tuning active: mgmt channel open...`.
- `/data/bcm/autowire-events.log` gains NO line for your process: a record
  with your script's name means rule 4 was not met.
- A disconnect paired with `hciN is draining, migrating` is drain
  cooperation, not a failure. A warning naming "another live instance of
  this service" means an orphaned pid is fighting you — check for leftovers.
- On prod a restart of dbus-ble-sensors-py after anyone's USB card reset is
  the stock `bt-config` hotplug rule, not your change (open item with Clint).

## 5. Operator knobs

- Rollback: `git -C /data/bcm checkout <hash> && /data/bcm/install.sh`,
  then restart the consumers (`packaging/restart-consumers.sh`, with notice
  to the running monitor first).
- Canary: clone a second checkout at a pinned hash and point ONE service's
  `BLUETOOTH_CONNECTION_MANAGER_DIR` at it; everyone else rides `/data/bcm`.
- Card cycling kill switch: `touch /data/bcm/no-card-cycle`.
