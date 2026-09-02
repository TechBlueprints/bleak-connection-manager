#!/bin/sh
# Restart BCM's shim-launched consumers so they pick up a new /data/bcm, and
# RECORD it: one line per service in /data/bcm/deploy.log, so anyone
# attributing pid changes on this box can see them (the night watch could
# not, and filed BCM's restarts as unexplained, 2026-09-02).
#
#   packaging/restart-consumers.sh            # the default set
#   packaging/restart-consumers.sh svc1 svc2  # an explicit set
#
# blebattery.* are deliberately not in the default set: serialbattery runs
# its own vendored BCM, so a restart there gains nothing and costs a pack
# telemetry gap. Staggered 3s so the claims directory sees one departure
# at a time.
ROOT="${BCM_ROOT:-/data/bcm}"
LOG="$ROOT/deploy.log"
VERSION="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DEFAULT="dbus-shyion-switch dbus-easytouchrv dbus-power-watchdog dbus-ble-sensors-py"
SVCS="${*:-$DEFAULT}"
for s in $SVCS; do
    [ -d "/service/$s" ] || continue
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') restart $s $VERSION by ${SUDO_USER:-${USER:-?}}@${SSH_CLIENT%% *}" >> "$LOG" 2>/dev/null
    svc -t "/service/$s"
    sleep 3
done
sleep 8
for s in $SVCS; do
    [ -d "/service/$s" ] && svstat "/service/$s"
done
