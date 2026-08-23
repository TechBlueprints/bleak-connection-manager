#!/bin/sh
# Crash-hunt core catcher for Venus OS. Arm with:
#   sysctl -w kernel.core_pattern="|/data/bcm/packaging/core-catcher.sh %e %p %s" \
#              kernel.core_pipe_limit=1
# Disarm (restore Venus's own handler):
#   /sbin/core-handler --init
#
# Venus's stock handler logs one line per crash and DISCARDS the core - its
# gdb branch is dead code, since the image ships no gdb. Weeks of crashes
# left nothing to examine. This keeps the core instead.
#
# NOTE it REPLACES the stock handler rather than supplementing it, so
# /var/log/crash-logger stops recording while this is armed. Disarm when the
# investigation is done. A reboot also restores the stock handler, since
# core-handler --init runs at boot - so the experiment self-ends.
#
# cat, NOT dd: the kernel feeds the core through a PIPE, and dd counts every
# short read as a whole block, so `dd bs=1M count=400` silently truncated
# the first two cores captured (25MB of 47MB) - losing the faulting thread's
# stack, which was the whole point. Size is bounded by keeping only 3.
COMM="$1"; PID="$2"; SIG="$3"
CORES=/data/cores
KEEP=3          # per process name, not total
MIN_FREE_K=300000

case "$COMM" in
    python*|bluetoothd) : ;;   # the two that actually crash on these boxes
    *) exec cat > /dev/null ;;
esac

mkdir -p "$CORES" 2>/dev/null
FREE=$(df -k /data | tail -n 1 | awk '{print $4}')
if [ "${FREE:-0}" -lt "$MIN_FREE_K" ]; then
    echo "$(date) SKIPPED $COMM.$PID sig$SIG: only ${FREE}k free" >> "$CORES/capture.log"
    exec cat > /dev/null
fi

OUT="$CORES/core.$COMM.$PID.sig$SIG.$(date +%s)"
cat > "$OUT"
# Prune PER PROCESS NAME, not globally. A global newest-N evicts by age
# alone, so a rare small core is displaced by common large ones - field
# 2026-08-23: the first bluetoothd core (3.7MB, the C-only one that is by
# far the easiest to analyse) was pruned by three python cores of 48MB
# each within an hour of being captured.
ls -t "$CORES"/core."$COMM".* 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r f; do rm -f "$f"; done
echo "$(date) captured $OUT (sig$SIG) $(wc -c < "$OUT") bytes" >> "$CORES/capture.log"
