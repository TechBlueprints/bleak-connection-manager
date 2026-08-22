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

# submodules (bleak, bleak-retry-connector) - pinned by the checkout
if [ -d "$ROOT/.git" ]; then
    git -C "$ROOT" submodule update --init --quiet
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
    SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
    [ -d "$SITE" ] || { echo "bcm-install: no site-packages at $SITE" >&2; exit 1; }
    cp "$ROOT/packaging/bcm_autowire.py" "$SITE/bcm_autowire.py"
    printf 'import bcm_autowire\n' > "$SITE/bcm_autowire.pth"
    echo "bcm-install: autowire planted in $SITE"
    # the rootfs (and the .pth with it) is erased by firmware updates;
    # /data/rc.local replants it on every boot
    RC=/data/rc.local
    LINE="[ -x $ROOT/install.sh ] && $ROOT/install.sh --autowire >/dev/null 2>&1 # bcm-autowire"
    if [ -w /data ]; then
        touch "$RC"; chmod 755 "$RC"
        grep -q "# bcm-autowire" "$RC" 2>/dev/null || echo "$LINE" >> "$RC"
        echo "bcm-install: replant registered in $RC"
    fi
fi
