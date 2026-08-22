# Migrating a consumer to the shared BCM install

Audience: the session/maintainer of one consumer service (dbus-shyion-switch,
venus-os-dbus-ble-sensors-py, dbus-power-watchdog, venus-os_dbus-serialbattery).
One pass per consumer; after it, the service never re-vendors this stack again.
Deploy point: BCM main @ 25a16d4 or later.

## 1. Installer: converge the shared checkout

Add to the service's installer (idempotent; safe if another consumer already
installed or updated it — ff-only means a stale installer can never move the
fleet backwards):

```sh
BCM_DIR=/data/bcm
if [ -d "$BCM_DIR/.git" ]; then
    git -C "$BCM_DIR" fetch -q origin && git -C "$BCM_DIR" merge -q --ff-only origin/main
else
    git clone -q https://github.com/TechBlueprints/bleak-connection-manager "$BCM_DIR"
fi
"$BCM_DIR/install.sh"
```

`install.sh` initializes the pinned submodules (bleak v3.0.2, brc v4.6.0),
smoke-imports the whole stack, and writes the interpreter shim
`/data/bcm/python3`. If the smoke import fails it does NOT update the shim and
prints the rollback command — surface that as an installer failure.

Do NOT pass `--autowire`. That is a separate, fleet-level decision Clint makes
once the shared checkout has soaked.

## 2. Run script: exec through the shim, with standalone fallback

```sh
BCM_PY=/data/bcm/python3
[ -x "$BCM_PY" ] || BCM_PY=python3
exec "$BCM_PY" <your-main>.py
```

The fallback keeps the public repo working from a bare clone (using whatever
the repo still vendors for standalone use). The shim also exports
BCM_AUTOWIRE=0 — shim-launched processes are deliberate consumers and are
never autowired.

## 3. Delete the vendored copies

Remove from the service's `ext/` (or equivalent): `bleak`,
`bleak_retry_connector`, `bleak-connection-manager`, `dbus_fast`,
`bluetooth_adapters`, `aiooui` — everything the shared checkout now serves.
KEEP service-specific deps (aiobmsble, velib_python, etc.). Remove the
`sys.path` insertions that pointed at the deleted trees; the shim provides the
path. Keep the `install_bleak_catcher(...)` call exactly as it is — owner,
adapters, link_caps, validators all unchanged.

If the repo must stay runnable standalone for third parties, either keep a
vendored set for that purpose (the run-script fallback uses it automatically)
or document the shared install as a prerequisite — maintainer's choice.

## 4. Verify after deploy

- `install.sh` printed `bcm-install: shim ready ... (<hash>)`.
- Service starts, connects; claims appear in `/run/bt-claims` with your owner.
- Expect one INFO line per process: `conn-param tuning active: mgmt channel
  open...` — tuning genuinely works on Venus as of cfbace5; this line is the
  positive observable. If timing-sensitive behavior changes, say so — the
  activation is new fleet-wide.
- A disconnect paired with `hciN is draining, migrating` is drain cooperation,
  not a failure (convention 0.3). A warning naming "another live instance of
  this service" means an orphaned pid is fighting you — check for leftovers.

## 5. Canary / rollback (operator knobs, for reference)

- Rollback: `git -C /data/bcm checkout <hash> && /data/bcm/install.sh`
- Canary: clone a second checkout at a pinned hash, set `BCM_ROOT=<that path>`
  in ONE service's run script; everyone else rides `/data/bcm`.
