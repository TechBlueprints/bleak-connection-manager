#!/bin/sh
# Arm the crash-hunt core catcher, optionally surviving reboots.
#
#   /data/bcm/packaging/arm-core-catcher.sh            arm until next reboot
#   /data/bcm/packaging/arm-core-catcher.sh --persist  arm now AND register in
#                                                      /data/rc.local so every
#                                                      boot re-arms
#   /data/bcm/packaging/arm-core-catcher.sh --disarm   restore Venus's stock
#                                                      handler and remove the
#                                                      rc.local registration
#
# Why this exists: Venus re-arms its own core handler (which DISCARDS cores)
# at every boot via core-handler --init. During the 2026-08 crash hunt the
# catcher was silently disarmed by reboots three times, each producing a
# blind window that read as "no crashes". Persistence trades Venus's
# one-line crash-logger (which stops recording while the catcher is armed)
# for actually keeping cores - a deliberate, documented, reversible choice.
set -e

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
CATCHER="$(dirname "$SELF")/core-catcher.sh"
RC=/data/rc.local
TAG="# bcm-core-catcher"

case "$1" in
    --disarm)
        /sbin/core-handler --init 2>/dev/null || sysctl -w kernel.core_pattern="core" >/dev/null
        if [ -f "$RC" ] && grep -q "$TAG" "$RC"; then
            grep -v "$TAG" "$RC" > "$RC.tmp" && mv "$RC.tmp" "$RC" && chmod 755 "$RC"
        fi
        echo "core catcher disarmed; Venus stock handler restored; rc.local registration removed"
        exit 0
        ;;
    --persist)
        touch "$RC"; chmod 755 "$RC"
        grep -q "$TAG" "$RC" || echo "[ -x $SELF ] && $SELF > /data/cores/arm.log 2>&1 $TAG" >> "$RC"
        echo "registered in $RC (re-arms at every boot; disarm with: $SELF --disarm)"
        ;;
esac

[ -x "$CATCHER" ] || { echo "catcher missing at $CATCHER" >&2; exit 1; }
mkdir -p /data/cores
sysctl -w kernel.core_pattern="|$CATCHER %e %p %s" kernel.core_pipe_limit=1 >/dev/null
echo "armed: $(cat /proc/sys/kernel/core_pattern) ($(date -u 2>/dev/null || date))"
