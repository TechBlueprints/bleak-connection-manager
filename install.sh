#!/bin/sh
# BCM shared-install setup - runs ON the Cerbo, from inside this checkout.
#
# The shared install IS a git checkout (normally /data/bcm - the one
# directory that survives Venus firmware updates). Consumer installers
# converge it:
#
#   BCM_DIR=/data/bcm
#   if [ -d "$BCM_DIR/.git" ]; then
#       git -C "$BCM_DIR" fetch -q origin && git -C "$BCM_DIR" merge -q --ff-only origin/main
#   else
#       git clone -q https://github.com/TechBlueprints/bleak-connection-manager "$BCM_DIR"
#   fi
#   "$BCM_DIR/install.sh"
#
# --ff-only means a stale consumer's install can never move the fleet
# backwards; rollback is `git checkout <hash> && ./install.sh`; canary is
# a second clone at a pinned hash plus BCM_ROOT in one service's run
# script. This script finishes the job: submodules, smoke import, shim.
#
# --autowire additionally plants the sitewide import hook so EVERY python
# process that imports bleak gets the catcher (see packaging/bcm_autowire.py).
# Off by default; enable only once the shared lib has soaked on the fleet.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
LIBPATH="$ROOT/src:$ROOT/ext:$ROOT/ext/upstream/bleak:$ROOT/ext/upstream/bleak-retry-connector/src"

# submodules (bleak, bleak-retry-connector) - pinned by the checkout.
# Shallow, sha-targeted, retried: consumer installers run this over RV
# uplinks (Starlink/LTE - field 2026-08-22: a full bleak history clone
# died twice mid-transfer), so fetch exactly the pinned commit at depth 1
# and keep the window small. Handles every partial state a killed run can
# leave, including a created-but-unborn submodule (no HEAD yet).
_submodule_at_pin() {
    sub="$1"
    sha="$(git -C "$ROOT" ls-tree HEAD -- "$sub" | awk '{print $3}')"
    [ -n "$sha" ] || return 1
    cur="$(git -C "$ROOT/$sub" rev-parse HEAD 2>/dev/null || true)"
    [ "$cur" = "$sha" ] && return 0
    # registers config and creates the submodule repo; often lands the pin
    # outright. Failures are fine - the sha fetch below is the real step.
    git -C "$ROOT" submodule update --init --depth 1 --quiet -- "$sub" 2>/dev/null || true
    cur="$(git -C "$ROOT/$sub" rev-parse HEAD 2>/dev/null || true)"
    [ "$cur" = "$sha" ] && return 0
    [ -e "$ROOT/$sub/.git" ] || return 1
    n=0
    while [ "$n" -lt 3 ]; do
        if git -C "$ROOT/$sub" fetch --quiet --depth 1 origin "$sha" 2>/dev/null \
            && git -C "$ROOT/$sub" checkout --quiet --detach "$sha" 2>/dev/null; then
            return 0
        fi
        n=$((n + 1))
        echo "bcm-install: fetch of $sub@$sha failed (attempt $n/3), retrying" >&2
        sleep 5
    done
    return 1
}

# ask git whether this is a checkout rather than inspecting its layout: in
# a SUBMODULE .git is a FILE pointing into the superproject, not a
# directory, so `[ -d .git ]` is false for every vendored-as-submodule
# deployment while being true in any fixture built with `git clone`
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    for sub in ext/upstream/bleak ext/upstream/bleak-retry-connector; do
        _submodule_at_pin "$sub" || { echo "bcm-install: could not fetch $sub at its pin - flaky link? re-run install.sh" >&2; exit 1; }
    done
fi
[ -f "$ROOT/ext/upstream/bleak/bleak/__init__.py" ] || { echo "bcm-install: bleak submodule missing (git submodule update --init)" >&2; exit 1; }

# smoke: every library a consumer will import, under the exact path the
# shim serves. A failure leaves the previous shim untouched and names the
# rollback.
if ! PYTHONPATH="$LIBPATH" python3 -c "import bleak_connection_manager, bleak, bleak_retry_connector, dbus_fast, bluetooth_adapters, aiooui" ; then
    PREV="$(git -C "$ROOT" rev-parse --short '@{1}' 2>/dev/null || true)"
    echo "bcm-install: smoke import FAILED - shim not updated." >&2
    [ -n "$PREV" ] && echo "bcm-install: roll back with: git -C $ROOT checkout $PREV && $ROOT/install.sh" >&2
    exit 1
fi

# the interpreter shim: the ONE place the shared path is written down.
# Consumers exec this instead of python3 (falling back to python3 when it
# is absent). BCM_ROOT=<other-checkout> in a run script is the canary knob.
cat > "$ROOT/python3.tmp" <<SHIM_EOF
#!/bin/sh
# BCM interpreter shim - written by install.sh; do not edit.
R="\${BCM_ROOT:-$ROOT}"
export PYTHONPATH="\$R/src:\$R/ext:\$R/ext/upstream/bleak:\$R/ext/upstream/bleak-retry-connector/src\${PYTHONPATH:+:\$PYTHONPATH}"
# shim-launched processes are deliberate BCM consumers: they install the
# catcher explicitly, so the sitewide autowire must stand down for them
export BCM_AUTOWIRE=0
exec python3 "\$@"
SHIM_EOF
chmod 755 "$ROOT/python3.tmp"
mv "$ROOT/python3.tmp" "$ROOT/python3"

VERSION="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "bcm-install: shim ready at $ROOT/python3 ($VERSION)"

# --autowire: sitewide hook. A .pth runs in EVERY python process on the
# box, so this is opt-in and the module it loads is built to never raise.
if [ "$1" = "--autowire" ]; then
    SITE="${BCM_SITE:-$(python3 -c 'import site; print(site.getsitepackages()[0])')}"
    [ -d "$SITE" ] || { echo "bcm-install: no site-packages at $SITE" >&2; exit 1; }
    # idempotence first: on a normal boot the plant is intact and nothing
    # below (including the remount dance) needs to run
    if cmp -s "$ROOT/packaging/bcm_autowire.py" "$SITE/bcm_autowire.py" 2>/dev/null \
        && [ "$(cat "$SITE/bcm_autowire.pth" 2>/dev/null)" = "import bcm_autowire" ]; then
        echo "bcm-install: autowire already planted in $SITE"
    else
        # Venus mounts the rootfs read-only (field 2026-08-22, prod): the
        # plant - and every post-firmware-update REPLANT, which is the
        # whole point - must handle the remount itself, and must never
        # exit early while the rootfs is left rw
        REMOUNTED=0
        if touch "$SITE/.bcm-write-test" 2>/dev/null; then
            rm -f "$SITE/.bcm-write-test"
        else
            echo "bcm-install: rootfs is read-only, remounting rw for the autowire plant"
            if mount -o remount,rw /; then
                REMOUNTED=1
            else
                echo "bcm-install: autowire plant FAILED: cannot remount rootfs rw" >&2
                exit 1
            fi
        fi
        PLANTED=1
        cp "$ROOT/packaging/bcm_autowire.py" "$SITE/bcm_autowire.py" || PLANTED=0
        if [ "$PLANTED" = 1 ]; then
            printf 'import bcm_autowire\n' > "$SITE/bcm_autowire.pth" || PLANTED=0
        fi
        if [ "$REMOUNTED" = 1 ]; then
            mount -o remount,ro / || echo "bcm-install: WARNING: could not remount rootfs read-only" >&2
        fi
        [ "$PLANTED" = 1 ] || { echo "bcm-install: autowire plant FAILED writing into $SITE" >&2; exit 1; }
        echo "bcm-install: autowire planted in $SITE"
    fi
    # the rootfs (and the .pth with it) is erased by firmware updates;
    # /data/rc.local replants on every boot. The replant logs to a file
    # instead of /dev/null: a failed replant after a firmware update is
    # exactly the event that must not disappear silently.
    RC=/data/rc.local
    LOG="$ROOT/autowire-replant.log"
    LINE="[ -x $ROOT/install.sh ] && $ROOT/install.sh --autowire > $LOG 2>&1 # bcm-autowire"
    if [ -w /data ]; then
        touch "$RC"; chmod 755 "$RC"
        if grep -q "# bcm-autowire" "$RC" 2>/dev/null; then
            # refresh the registered line (older registrations logged to /dev/null)
            grep -v "# bcm-autowire" "$RC" > "$RC.tmp" && echo "$LINE" >> "$RC.tmp" && mv "$RC.tmp" "$RC" && chmod 755 "$RC"
        else
            echo "$LINE" >> "$RC"
        fi
        echo "bcm-install: replant registered in $RC (logs to $LOG)"
    fi
fi
