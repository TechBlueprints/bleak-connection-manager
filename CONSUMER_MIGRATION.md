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

`install.sh` fetches the pinned submodules (bleak v3.0.2, brc v4.6.0)
shallowly and by exact sha, with retries — sized for RV uplinks
(Starlink/LTE), ~2 MB instead of full clone history — then smoke-imports
the whole stack and writes the interpreter shim `/data/bcm/python3`. If
the smoke import fails it does NOT update the shim and prints the
rollback command — surface that as an installer failure. Mind the shell:
piping install.sh through `tail`/`tee` eats its exit code unless you
`set -o pipefail` (BusyBox ash supports it) or check `$PIPESTATUS`; when
in doubt, run it unpiped and capture `$?` directly. A run killed by the
link is safe to re-run — the installer is idempotent and resumes from
whatever partial state the death left.

Do NOT pass `--autowire` from a consumer installer. Autowire is a
fleet-level, per-box decision (already enabled on the prod Cerbo by Clint);
when it is enabled, `install.sh --autowire` handles the read-only Venus
rootfs itself (remount-rw, plant, remount-ro) and registers a boot-time
replant in `/data/rc.local` that logs to `/data/bcm/autowire-replant.log` —
check that log after any firmware update.

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

**If you keep a vendored fallback, keep it CURRENT.** This is the trap
worth spelling out, found the hard way (2026-08-25): a consumer whose
installer treats convergence as non-fatal — fetch fails, smoke import
fails, log it and carry on with the vendored copy — has built a path that
is taken *precisely when something has already gone wrong*. If that
vendored copy is pinned at some older commit, the box quietly runs the old
stack **as a consequence of the failure**, and the bug you shipped a fix
for is live again on the one box that just told you it was unhealthy. The
fallback is not a safety net if it is stale; it is a silent downgrade.

Two acceptable resolutions, and you should pick one deliberately:

- **Keep both paths current** — bump the vendored copy whenever you bump
  the shared checkout, so either path carries the same fixes. Preferred
  when the repo must stay runnable standalone.
- **Make convergence fatal** — treat a failed fetch or smoke import as an
  installer failure and refuse to start, so a broken convergence is loud
  rather than a downgrade.

What is not acceptable is a fail-soft fallback nobody updates. If you take
the first option, prefer a *mechanism* over a promise: have the installer
compare the two shas after converging and report when the vendored copy is
an ancestor, with the distance and what it costs. Report only *behind* —
ahead or diverged is a deliberate pin, and warning on it trains people to
ignore the line.

One thing to expect if you build that check: **you will not be able to
provoke it by hand on a box.** A `git submodule update --init` earlier in
the same installer undoes any staling you do to the working tree before
the comparison runs. That is correct — the question is whether the repo's
*pin* lags the shared checkout, not whether someone poked the tree — but
it means the warning cannot be demonstrated live, and an implementer may
wrongly conclude it does not work. Cover it with a test that builds two
clones of one history at different points and asserts all four cases:
behind warns with the distance, in-step and ahead stay silent, and a
missing vendored tree is tolerated rather than an error.

If the repo must stay runnable standalone for third parties, keep a
vendored set and defer to whatever the interpreter already provides. The
reference pattern (contributed from dbus-easytouchrv's migration, the first
through this checklist — adapt the subpath list to your vendored set):

```python
def _ensure_ble_stack() -> None:
    """Put the vendored ext/ BLE stack on sys.path unless already provided
    (by the shared /data/bcm shim, or by test stubs)."""
    if "bleak_connection_manager" in sys.modules:
        return
    try:
        import importlib.util
        if importlib.util.find_spec("bleak_connection_manager") is not None:
            return
    except (ImportError, ValueError):
        pass
    _ext = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext")
    for _sub in [
        os.path.join(_ext, "bleak-connection-manager", "src"),
        os.path.join(_ext, "bleak-connection-manager", "ext"),
        os.path.join(_ext, "bleak-retry-connector", "src"),
        os.path.join(_ext, "bluetooth-adapters", "src"),
        os.path.join(_ext, "aiooui", "src"),
        os.path.join(_ext, "bleak"),
    ]:
        if os.path.isdir(_sub) and _sub not in sys.path:
            sys.path.insert(0, _sub)
```

Design notes that make this shape correct:
1. The `sys.modules` check comes first — cheap, and it is what makes test
   stubs work (ModuleType entries with `__spec__ = None` make `find_spec`
   raise `ValueError`, which is also why that except clause catches it).
2. Call it lazily from the function that installs the catcher, not at
   module import — entry points then need zero path knowledge.
3. The ImportError/ValueError swallow means a weird interpreter state
   degrades to "insert ext paths" — the safe direction: worst case you
   shadow the shim with your vendored copy, never running with no stack.

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
