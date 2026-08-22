# BCM Operations Guide — Everything You Need to Continue

This document captures the full operational context for bleak-connection-manager (BCM) integration with Venus OS services on a Cerbo GX. Written to enable a new chat agent to pick up where the previous one left off.

## Target System

- **Device**: Cerbo GX running Venus OS v3.67
- **SSH**: `ssh root@cerbo` (hostname alias configured in `~/.ssh/config`)
- **Shell**: BusyBox — `head` and `tail` require `-n N` not `-N` (this bites you every time)
- **Service manager**: daemontools (`svc`, `svstat`, `svscan`). Services live in `/service/`, managed via symlinks from `/data/apps/*/service/`.
- **Logging**: `multilog` writes to `/var/log/<service>/current` in TAI64 format. Pipe through `tai64nlocal` to get human-readable timestamps.
- **Python**: `/usr/bin/python` (3.12, BusyBox Linux)

## BLE Devices on the System

| MAC Address | Device | Service | Adapter | Critical |
|---|---|---|---|---|
| `53:20:B7:D7:F9:E7` | HumsiENK BMS (bat0) | `dbus-blebattery.0` | hci0 | **YES** - battery management |
| `AB:80:72:54:E0:B4` | HumsiENK BMS (bat1) | `dbus-blebattery.1` | hci0 | **YES** - battery management |
| `24:EC:4A:E4:69:A5` | Power Watchdog | `dbus-power-watchdog` | hci0 | No |
| 4x Shyion relays | Relay switches | `dbus-shyion-switch` | varies | No |

## Project Locations (Mac Development Machine)

| Project | Path | Branch | Purpose |
|---|---|---|---|
| BCM library | `/Users/clint/techblueprints/bleak-connection-manager` | `main` | The connection manager library |
| dbus-serialbattery | `/Users/clint/techblueprints/venus-os_dbus-serialbattery` | `bcm-integration` | BMS battery driver |
| dbus-power-watchdog | `/Users/clint/techblueprints/dbus-power-watchdog` | `feature/bcm-integration` | Power monitor |
| dbus-shyion-switch | `/Users/clint/techblueprints/dbus-shyion-switch` | `feature/bcm-integration` | Relay controller |

## Deployment Paths on Cerbo

| Service | Cerbo Path |
|---|---|
| serialbattery | `/data/apps/dbus-serialbattery/` |
| power-watchdog | `/data/apps/dbus-power-watchdog/` |
| shyion-switch | `/data/apps/dbus-shyion-switch/` |

## How BCM is Vendored

Each project includes BCM and all BLE dependencies as git submodules in `ext/`:

```
ext/
  bleak-connection-manager/   -> TechBlueprints/bleak-connection-manager (main)
  bleak-retry-connector/      -> Bluetooth-Devices/bleak-retry-connector (main)
  bleak/                      -> hbldh/bleak (develop)
  bluetooth-adapters/         -> Bluetooth-Devices/bluetooth-adapters (main)
  aiooui/                     -> Bluetooth-Devices/aiooui (main)
  velib_python/               -> victronenergy/velib_python (master)
```

Each project's entry point adds these to `sys.path`. The serialbattery project does this in `dbus-serialbattery.py`:

```python
_ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext")
for _sub in [
    os.path.join(_ext_dir, "bleak-connection-manager", "src"),
    os.path.join(_ext_dir, "bleak-retry-connector", "src"),
    os.path.join(_ext_dir, "bluetooth-adapters", "src"),
    os.path.join(_ext_dir, "aiooui", "src"),
    os.path.join(_ext_dir, "bleak"),
]:
    if os.path.isdir(_sub) and _sub not in sys.path:
        sys.path.insert(0, _sub)
```

Power-watchdog and shyion-switch have the same pattern in their entry points.

## How to Deploy All Services

### Deploy BCM library to all three projects at once:

```bash
# From development Mac:
rsync -avz --exclude='.git' --exclude='__pycache__' \
  /Users/clint/techblueprints/bleak-connection-manager/src/bleak_connection_manager/ \
  root@cerbo:/data/apps/dbus-serialbattery/ext/bleak-connection-manager/src/bleak_connection_manager/

rsync -avz --exclude='.git' --exclude='__pycache__' \
  /Users/clint/techblueprints/bleak-connection-manager/src/bleak_connection_manager/ \
  root@cerbo:/data/apps/dbus-power-watchdog/ext/bleak-connection-manager/src/bleak_connection_manager/

rsync -avz --exclude='.git' --exclude='__pycache__' \
  /Users/clint/techblueprints/bleak-connection-manager/src/bleak_connection_manager/ \
  root@cerbo:/data/apps/dbus-shyion-switch/ext/bleak-connection-manager/src/bleak_connection_manager/
```

### Deploy serialbattery driver files:

```bash
rsync -avz /Users/clint/techblueprints/venus-os_dbus-serialbattery/dbus-serialbattery/utils_ble.py \
  root@cerbo:/data/apps/dbus-serialbattery/utils_ble.py

rsync -avz /Users/clint/techblueprints/venus-os_dbus-serialbattery/dbus-serialbattery/bms/humsienk_ble.py \
  root@cerbo:/data/apps/dbus-serialbattery/bms/humsienk_ble.py

rsync -avz /Users/clint/techblueprints/venus-os_dbus-serialbattery/dbus-serialbattery/dbus-serialbattery.py \
  root@cerbo:/data/apps/dbus-serialbattery/dbus-serialbattery.py
```

### Deploy power-watchdog:

```bash
rsync -avz /Users/clint/techblueprints/dbus-power-watchdog/power_watchdog_device.py \
  root@cerbo:/data/apps/dbus-power-watchdog/power_watchdog_device.py

rsync -avz /Users/clint/techblueprints/dbus-power-watchdog/dbus-power-watchdog.py \
  root@cerbo:/data/apps/dbus-power-watchdog/dbus-power-watchdog.py
```

### Deploy shyion-switch:

```bash
rsync -avz /Users/clint/techblueprints/dbus-shyion-switch/shyion_ble.py \
  root@cerbo:/data/apps/dbus-shyion-switch/shyion_ble.py

rsync -avz /Users/clint/techblueprints/dbus-shyion-switch/dbus-shyion-switch.py \
  root@cerbo:/data/apps/dbus-shyion-switch/dbus-shyion-switch.py
```

### Restart services after deployment:

```bash
ssh root@cerbo 'svc -d /service/dbus-blebattery.0 /service/dbus-blebattery.1; sleep 3; svc -u /service/dbus-blebattery.0 /service/dbus-blebattery.1'
ssh root@cerbo 'svc -d /service/dbus-power-watchdog; sleep 2; svc -u /service/dbus-power-watchdog'
ssh root@cerbo 'svc -d /service/dbus-shyion-switch; sleep 2; svc -u /service/dbus-shyion-switch'
```

**CRITICAL WARNING**: Never use `rsync --delete` on `/data/apps/` directories. This deletes the `supervise/` directories that daemontools needs. If this happens, you must recreate them and restart `svscanboot` (which will reboot the Cerbo).

## Diagnosing Stuck Connections

### Quick Health Check (run this first):

```bash
ssh root@cerbo '
  echo "=== time ==="; date
  echo "=== hci connections ==="; hcitool -i hci0 con
  echo "=== service status ==="; svstat /service/dbus-blebattery.0 /service/dbus-blebattery.1 /service/dbus-power-watchdog /service/dbus-shyion-switch
  echo "=== bat0 latest ==="; tai64nlocal < /var/log/dbus-blebattery.0/current | tail -n 5
  echo "=== bat1 latest ==="; tai64nlocal < /var/log/dbus-blebattery.1/current | tail -n 5
  echo "=== dbus voltage ==="; dbus -y com.victronenergy.battery.ble_5320b7d7f9e7 /Dc/0/Voltage GetValue 2>/dev/null || echo "bat0 MISSING"
  dbus -y com.victronenergy.battery.ble_ab807254e0b4 /Dc/0/Voltage GetValue 2>/dev/null || echo "bat1 MISSING"
'
```

### What to look for in the log output:

| Log Pattern | Meaning | Action |
|---|---|---|
| `heartbeat: BLE=UP, last_data=Xs ago` | Normal operation, polling loop alive | Good if X < 60 |
| `heartbeat: BLE=DOWN` | BLE connection lost, daemon thread reconnecting | Wait for reconnect |
| `Data resumed — RX battery_info (26B)` | Fresh BMS data received | Good |
| `Data resumed — RX handshake (0B)` | Handshake only, no battery_info yet | Expect battery_info in ~60s |
| `Stale data (Xs since last)` | No BMS data for X seconds | Problem if X > 120 |
| `WATCHDOG FIRED` | 90s no-notification timeout hit | BCM will disconnect + reconnect |
| `InProgress on hciX, rotating` | BlueZ scan contention | BCM tries next adapter |
| `Found in BlueZ cache (bypassed InProgress)` | BCM cache fallback worked | Good |
| `scan failed, trying direct connect` | Scan failed, falling back to cached BlueZ device | Good |
| `connection attempt failed (N consecutive)` | Repeated reconnect failures | Investigate |
| No log output for > 2 minutes | **Silent hang** — see below | Send SIGUSR1 |

### Silent Hang Diagnosis

If a battery service has no log output for 2+ minutes but the process is alive:

1. **Find the Python PID** (not the `run` script PID):
   ```bash
   ps | grep "python /data/apps/dbus-serialbattery" | grep -v grep
   ```

2. **Send SIGUSR1 to dump Python thread tracebacks** (faulthandler is enabled):
   ```bash
   kill -USR1 <python_pid>
   ```

3. **Read the traceback from the log**:
   ```bash
   tai64nlocal < /var/log/dbus-blebattery.0/current | tail -n 30
   ```

4. **Interpret common stuck locations**:
   - `_pop_next_notification` at `time.sleep(0.02)` — **Normal**. The main thread is just in the 0.3s poll loop. If heartbeats are flowing, everything is fine. If logs are silent but heartbeats show `last_data` increasing, data IS flowing but at DEBUG level.
   - `selectors.py:select` in BLE thread — **Normal**. Asyncio event loop waiting.
   - `future.result(timeout=2.0)` in send_data — **Possible hang**. BLE thread not processing commands.

### When Batteries Are "Stuck" (Not Emitting D-Bus Signals)

The most common failure mode observed:

1. **Connection drops silently** — HCI handles disappear from `hcitool -i hci0 con`
2. **Reconnection fails** — `org.bluez.Error.InProgress` blocks all scan attempts
3. **Root cause**: BlueZ stuck in stale internal scan state (from previous failed BCM scan or `vesmart_server`), returning `InProgress` on all scan attempts

**Recovery steps**:
```bash
# Check if BlueZ is stuck in Discovering mode:
bluetoothctl show 2>&1 | grep Discovering

# If "Discovering: yes", try stopping it:
dbus-send --system --dest=org.bluez --print-reply /org/bluez/hci0 org.bluez.Adapter1.StopDiscovery

# If that returns "No discovery started" but Discovering is still yes,
# another process owns the scan. Power cycle the adapter:
hciconfig hci0 down && sleep 1 && hciconfig hci0 up

# Remove stale BlueZ cache for the specific devices:
bluetoothctl remove 53:20:B7:D7:F9:E7
bluetoothctl remove AB:80:72:54:E0:B4

# Restart battery services:
svc -d /service/dbus-blebattery.0 /service/dbus-blebattery.1
sleep 3
svc -u /service/dbus-blebattery.0 /service/dbus-blebattery.1
```

**WARNING**: `hciconfig hci0 down` will disconnect ALL BLE devices on hci0, including power-watchdog. Only do this if the batteries are stuck and not recovering.

## Key Configuration Values

### In `utils_ble.py` (serialbattery):
- `_notification_watchdog_timeout = 90` — If no BLE notifications for 90s, force reconnect
- `overall_timeout=240.0` — Max time for entire `establish_connection` call
- `timeout=15.0` — Per-attempt connection timeout
- `max_attempts=5` — Connection retry count
- `try_direct_first=True` (when scan fails) — Bypass scanning, use cached BlueZ device info
- **Direct connect fallback**: If `managed_find_device()` returns None, constructs a `BLEDevice` from the address and tries `establish_connection` with `try_direct_first=True`

### In `humsienk_ble.py`:
- `poll_interval = 1000` (1 second) — How often the GLib main loop calls `refresh_data()`
- Heartbeat log every 60s at INFO level showing BLE status, voltage, SoC
- Polls BMS commands (0x21, 0x20, 0x22) every 3 seconds when connected
- Re-sends handshake after 10s of no data
- Escalating stale-data warnings: <15s DEBUG, 15-60s INFO, 60-300s WARNING, 300-900s WARNING+D-Bus alarm, >900s ERROR+D-Bus alarm

### In `power_watchdog_device.py`:
- `overall_timeout=300.0` — 5 minute connection timeout
- `reset_adapter=False` — Does NOT reset BT adapter on failure

### In `shyion_ble.py`:
- `overall_timeout=300.0` — 5 minute connection timeout
- `reset_adapter=False` — Does NOT reset BT adapter on failure
- Polls relay state once per hour

### BCM Escalation Profiles:
- `PROFILE_BATTERY`: `reset_adapter=True`, most aggressive recovery (for critical BMS)
- `PROFILE_SENSOR` / `PROFILE_ON_DEMAND`: `reset_adapter=False`, lighter recovery

## Known Issues and Open Problems

### 1. Why Do HCI Connections Drop?
**Not fully diagnosed.** The HCI handles disappear silently — no disconnect callback fires, no error in logs. The BMS just stops sending data. Possible causes:
- BLE radio interference
- BlueZ internal state corruption
- HCI supervision timeout on the peripheral side
- The power-watchdog or shyion-switch competing for the adapter

### 2. `dbus-ble-advertisements` Does NOT Cause Scan Contention
This service uses `btmon` (raw HCI monitoring), NOT bleak or BlueZ scanning APIs. It does NOT hold a BlueZ discovery session and does NOT cause `InProgress` errors. If `InProgress` errors occur, the cause is elsewhere — likely stale BlueZ internal state from a previous failed scan by one of the BCM-using services themselves, or from `vesmart_server`. Do NOT blame `dbus-ble-advertisements` for scan contention.

**Mitigations for InProgress regardless of cause**:
- BCM scanner rotates through adapters
- BCM falls back to BlueZ cache lookup (`_find_in_bluez_cache()`)
- `utils_ble.py` falls back to direct connect when scan fails

### 3. BlueZ Stuck Discovering State
After stopping a process that held a scan, BlueZ sometimes stays in `Discovering: yes` with no way to stop it from another D-Bus client (`StopDiscovery` returns "No discovery started"). The only fix is power-cycling the adapter.

### 4. Log Silence Does Not Mean Stuck
After the handshake, if data flows continuously, ALL logging is at DEBUG level (not visible in multilog). The 60-second INFO heartbeat was added to confirm the process is alive. If heartbeats show `last_data=2s ago`, the process is working fine.

### 5. `faulthandler` Is Enabled
`dbus-serialbattery.py` imports `faulthandler` and registers SIGUSR1. Send `kill -USR1 <python_pid>` to dump all thread tracebacks to the log.

## BCM Architecture

```
bleak-connection-manager/src/bleak_connection_manager/
  __init__.py          - Public API exports
  connection.py        - establish_connection() — the main entry point
  scanner.py           - managed_find_device() / discover() with scan lock + rotation + InProgress retry + BlueZ cache fallback
  scan_lock.py         - Cross-process file-based scan locking (/run/bleak-cm-hciX-scan.lock)
  lock.py              - Cross-process file-based connection slot locking (/run/bleak-cm-hciX-slot-N.lock)
  watchdog.py          - ConnectionWatchdog — fires callback after no-activity timeout
  diagnostics.py       - diagnose_stuck_state() / clear_stuck_state() — phantom/stale detection
  recovery.py          - EscalationPolicy — progressive failure recovery (cache clear -> disconnect -> remove -> adapter reset)
  hci.py               - HCI interaction via hcitool subprocess (get_connections, find_connection_by_address, etc.)
  bluez.py             - BlueZ D-Bus helpers (disconnect_device, remove_device, is_inactive_connection, etc.)
  adapters.py          - Adapter discovery via bluetooth-adapters library, pick_adapter rotation
  validators.py        - Built-in connection validators (validate_gatt_services, validate_char_exists, validate_read_char)
  const.py             - Constants and dataclasses (LockConfig, ScanLockConfig, IS_LINUX, etc.)
```

## Updating BCM Submodules After Changes

When you make changes to BCM on the `main` branch:

```bash
# 1. Commit and push BCM
cd /Users/clint/techblueprints/bleak-connection-manager
git add -A && git commit -m "your message" && git push origin main

# 2. Update submodule in each consuming project
cd /Users/clint/techblueprints/venus-os_dbus-serialbattery/dbus-serialbattery/ext/bleak-connection-manager
git pull origin main
cd ../..
git add ext/bleak-connection-manager && git commit -m "Update BCM submodule" && git push origin bcm-integration

# Repeat for power-watchdog and shyion-switch:
cd /Users/clint/techblueprints/dbus-power-watchdog/ext/bleak-connection-manager
git pull origin main
cd ../..
git add ext/bleak-connection-manager && git commit -m "Update BCM submodule" && git push origin feature/bcm-integration

cd /Users/clint/techblueprints/dbus-shyion-switch/ext/bleak-connection-manager
git pull origin main
cd ../..
git add ext/bleak-connection-manager && git commit -m "Update BCM submodule" && git push origin feature/bcm-integration
```

## Monitoring Command (Run Periodically)

```bash
ssh root@cerbo '
  echo "=== $(date) ==="
  echo "--- hci ---"; hcitool -i hci0 con
  echo "--- bat0 ---"; tai64nlocal < /var/log/dbus-blebattery.0/current | grep -E "heartbeat|WATCHDOG|disconnect|Data resumed|Stale|stale|scanning|connection failed" | tail -n 5
  echo "--- bat1 ---"; tai64nlocal < /var/log/dbus-blebattery.1/current | grep -E "heartbeat|WATCHDOG|disconnect|Data resumed|Stale|stale|scanning|connection failed" | tail -n 5
  echo "--- dbus ---"
  V0=$(dbus -y com.victronenergy.battery.ble_5320b7d7f9e7 /Dc/0/Voltage GetValue 2>/dev/null || echo "MISSING")
  V1=$(dbus -y com.victronenergy.battery.ble_ab807254e0b4 /Dc/0/Voltage GetValue 2>/dev/null || echo "MISSING")
  echo "bat0=${V0}V  bat1=${V1}V"
'
```

### What "healthy" looks like:
- Both batteries show heartbeats with `last_data` < 60s
- HCI connections show handles for both BMS MACs + power-watchdog
- D-Bus voltage values are present and reasonable (13-14V range for 4S LiFePO4)

### What "stuck" looks like:
- No heartbeat for > 2 minutes
- HCI handles missing for BMS devices
- D-Bus values missing or unchanged
- Log shows repeated `connection attempt failed` or `All scans failed with InProgress`

## Relationship to Upstream bleak-retry-connector

BCM wraps `bleak-retry-connector` (upstream by Bluetooth-Devices/bdraco). We filed issues on the upstream repo proposing improvements. The maintainer asked us to discuss before PRing. Our issues are filed at: https://github.com/Bluetooth-Devices/bleak-retry-connector/issues

The key design decision: BCM is a **separate wrapper project**, not a fork. If upstream accepts our proposals, we remove the corresponding workaround from BCM. This keeps us independent of upstream acceptance timelines.

The `bleak-retry-connector` submodule points to the **upstream main branch** (Bluetooth-Devices), not our fork.

## Strategic Decision: We Moved Away From Upstream Changes

**We are NOT modifying bleak-retry-connector anymore.** This is the most important architectural decision in the project.

We originally planned 7 PRs to upstream `bleak-retry-connector`. We wrote code on feature branches, filed PRs, and the maintainer (bdraco) asked us to close them and file issues to discuss first instead. Rather than continue down that path, we decided to **build all the resilience we need into BCM (bleak-connection-manager) as a wrapper** that sits on top of unmodified upstream `bleak-retry-connector`.

**What this means in practice:**
- `bleak-retry-connector` submodule stays pointed at **upstream main**. Do not modify it.
- `bleak` submodule stays pointed at **upstream develop**. Do not modify it.
- ALL BLE resilience logic (phantom detection, scan locking, adapter rotation, escalation, watchdog, etc.) lives in BCM.
- The feature branches on our `bleak-retry-connector` fork are **historical reference only**. Do not merge them, do not reopen PRs from them.
- If upstream ever adopts similar features, we can remove the corresponding BCM code, but don't hold your breath.

**Why we moved away:**
- The maintainer's process (file issue -> discuss -> small PR) is slow and we need solutions now.
- Our use case (embedded Linux, BlueZ, multi-process, multiple adapters) is niche. The upstream library targets Home Assistant primarily.
- BCM gives us full control without waiting for upstream approval.

**Reference documents** (historical, for context on what we learned):
- `/Users/clint/techblueprints/bleak-retry-improvements/01-bleak-retry-connector-improvements.md` — The 7 proposed PRs (historical reference)
- `/Users/clint/techblueprints/bleak-retry-improvements/12-observed-stuck-states.md` — 22 observed stuck states (STILL ACTIVE — this is the field guide for diagnosing problems)
- `/Users/clint/techblueprints/bleak-retry-improvements/13-upstream-contribution-approach.md` — Strategy doc (historical)
- `/Users/clint/techblueprints/bleak-retry-improvements/14-23-issue-*.md` — Draft issues (historical)

## Files Modified in Each Project

### `venus-os_dbus-serialbattery` (branch: `bcm-integration`)

| File | What was changed |
|---|---|
| `dbus-serialbattery/dbus-serialbattery.py` | Added `faulthandler.enable()` and `faulthandler.register(signal.SIGUSR1)` for diagnostic tracebacks; added `sys.path` entries for vendored BLE deps |
| `dbus-serialbattery/utils_ble.py` | Complete BLE overhaul: integrated BCM's `establish_connection`, `managed_find_device`, `ConnectionWatchdog`, `EscalationPolicy`; added direct-connect fallback when scan fails; 90s watchdog timeout; enhanced disconnect/reconnect logging |
| `dbus-serialbattery/bms/humsienk_ble.py` | Added 60s INFO-level heartbeat log; escalating stale-data warnings; handshake resend after 10s silence |
| `dbus-serialbattery/ext/bleak-connection-manager` | Submodule at commit 42536e7 |

### `dbus-power-watchdog` (branch: `feature/bcm-integration`)

| File | What was changed |
|---|---|
| `dbus-power-watchdog.py` | Added `sys.path` entries for vendored BLE deps |
| `power_watchdog_device.py` | Rewrote BLE connectivity to use BCM; `reset_adapter=False`; `overall_timeout=300s` |
| `.gitmodules` | All BLE deps as submodules (BCM, bleak, bleak-retry-connector, bluetooth-adapters, aiooui) |

### `dbus-shyion-switch` (branch: `feature/bcm-integration`)

| File | What was changed |
|---|---|
| `dbus-shyion-switch.py` | Added `sys.path` entries for vendored BLE deps |
| `shyion_ble.py` | Rewrote BLE connectivity to use BCM; `reset_adapter=False`; `overall_timeout=300s`; polls relay state once per hour |
| `.gitmodules` | All BLE deps as submodules (same as power-watchdog) |

## Critical Things That Will Bite You

### 1. BusyBox Quirks on Venus OS
- `head -5` does NOT work. Use `head -n 5`.
- `tail -10` does NOT work. Use `tail -n 10`.
- `grep -P` (Perl regex) is not available. Use `grep -E` (extended regex).
- `ps aux` doesn't exist. Use `ps` (no flags) or `ps -o pid,args`.

### 2. daemontools Supervise Directories
- `/service/<name>/supervise/` directories are managed by daemontools.
- NEVER delete them. If deleted, the service cannot be managed by `svc`.
- If you accidentally delete them (e.g., via `rsync --delete`), you must either:
  - `mkdir -p /service/<name>/supervise` then restart `svscanboot`
  - Or reboot the Cerbo (which recreates everything)

### 3. vesmart_server Force-Disconnects All BLE Devices
- Victron's `vesmart_server.py` disconnects ALL BLE devices every 60 seconds.
- This amplifies phantom connection creation.
- A patch was deployed to `/data/vesmart-protected-devices.conf` listing all our device MACs.
- If the Cerbo firmware updates, this patch may be overwritten.
- The suppression list on the Cerbo should contain ALL our BLE MACs.

### 4. Log Output Levels
- Normal operation after BLE handshake logs at DEBUG level only.
- `multilog` does not show DEBUG by default.
- If logs look "silent" but heartbeats show `last_data=2s ago`, the process IS working.
- The 60-second heartbeat in `humsienk_ble.py` exists specifically to prove the process is alive.

### 5. Two BLE Adapters
- `hci0` = internal adapter on the Cerbo GX
- `hci1` = USB BLE adapter (may go DOWN spontaneously — Stuck State 22)
- BCM auto-discovers both and rotates between them on failure.

## The Core Unsolved Problem

**Why do HCI connections silently drop, and why doesn't BCM always reconnect?**

This was the user's primary frustration. The known facts:

1. HCI handles disappear silently — no disconnect callback fires.
2. The BMS just stops sending notification data.
3. The 90-second ConnectionWatchdog should fire and trigger reconnection.
4. Sometimes it does, and reconnection works.
5. Sometimes the reconnection gets stuck on `InProgress` errors from a stale BlueZ state.
6. Sometimes the service appears alive (heartbeats flowing, D-Bus values present) but the user perceives it as "stuck" — this was sometimes a false alarm caused by confusing DEBUG-level silence with a hang.

**Areas needing investigation:**

1. **Why doesn't the disconnect callback fire?** BCM registers one, but it may not be invoked if BlueZ itself doesn't realize the connection dropped (phantom creation path).
2. **Does the ConnectionWatchdog actually fire on time?** The timeout was reduced from 240s to 90s, but it needs monitoring to confirm it works in the field.
3. **When the watchdog fires, does `client.disconnect()` hang?** If the connection is phantom, `disconnect()` hangs (Stuck State 8). BCM has a disconnect timeout but this path needs verification.
4. **After disconnect, does the scan succeed?** If `dbus-ble-advertisements` holds a persistent scan, BCM's scan can fail with `InProgress`. The BlueZ cache fallback and direct-connect fallback were added to handle this, but their effectiveness in the field is unconfirmed.

**Next steps for a new chat agent:**
- Deploy the current code to Cerbo.
- Monitor every 20 minutes for at least 12 hours.
- When a battery goes "stuck", capture detailed diagnostics immediately:
  - `hcitool -i hci0 con` and `hcitool -i hci1 con`
  - `bluetoothctl info <MAC>` for each battery
  - Full log from the affected service: `tai64nlocal < /var/log/dbus-blebattery.N/current`
  - Send `kill -USR1 <python_pid>` and capture the thread traceback
- Correlate the stuck state with the 22-state taxonomy in `12-observed-stuck-states.md`.
- Determine which specific BCM path failed and fix it.

## User Rules and Preferences

These are hard-won rules from this conversation. Violating them will frustrate the user.

1. **Always SSH as `root@cerbo`** — not `cerbo`, not `user@cerbo`. The hostname alias is configured in `~/.ssh/config`.
2. **Always use `head -n N` and `tail -n N`** on Venus OS. BusyBox does not support `head -5` or `tail -10`. This was corrected repeatedly.
3. **Never reset HCI adapters unless absolutely necessary.** Other services (power-watchdog, shyion-switch, ble-advertisements) share the adapters. Only the BMS battery driver (`dbus-serialbattery`) is critical enough to justify `reset_adapter=True`.
4. **Never say "everything is working" based on one snapshot.** The user was repeatedly frustrated by premature "all good" conclusions. Monitor for at least 12 hours with checks every 20 minutes before declaring stability.
5. **Dependencies must come from upstream repos, not TechBlueprints forks.** `bleak` comes from `hbldh/bleak`, `bleak-retry-connector` comes from `Bluetooth-Devices/bleak-retry-connector`. We deliberately do NOT depend on any of our fork's feature branches. All custom logic lives in BCM.
6. **Do not modify bleak-retry-connector or bleak.** We moved away from upstream contributions. All resilience logic goes in BCM. The upstream submodules stay on upstream main/develop. See "Strategic Decision" section.
7. **Don't kill processes that get supervised back up.** Using `svc -d` and then `svc -u` is the correct pattern. Killing PIDs directly causes contention during the restart race.
8. **When polling multiple BLE devices, poll one at a time.** Don't try to connect to all devices simultaneously — this saturates the adapter.
9. **The user checks periodically and expects honest status.** If things are broken, say so. If you don't know, say that too.

## Objectives and Priorities

### Primary Goal
Get the two HumsiENK BLE batteries (`53:20:B7:D7:F9:E7` and `AB:80:72:54:E0:B4`) reliably connected and reporting voltage/SoC data 24/7 through D-Bus. Any outage longer than 5 minutes is unacceptable.

### Secondary Goals
1. Keep `dbus-power-watchdog` connected to the Hughes Power Watchdog 50A.
2. Keep `dbus-shyion-switch` able to poll/control Shyion relay switches (hourly poll is fine).
3. Iterate on BCM to handle all 22 observed stuck states automatically.

### Outstanding Investigation
The core question: **Why do connections silently die and why doesn't BCM always recover?** This requires overnight monitoring, capturing logs at the moment of failure, and correlating with the stuck state taxonomy.

### Outstanding Tasks (from conversation)
- Monitor Cerbo every 20 minutes, fix issues as found
- Diagnose why HCI connections drop silently (no disconnect callback)
- Verify the 90-second ConnectionWatchdog fires correctly in the field
- Verify the direct-connect fallback works during real disconnects
- File remaining issues on upstream `bleak-retry-connector` (one at a time, discuss first)
- Create `downstream/all-features` branch in `bleak-retry-connector` fork merging all feature branches

## Upstream Feature Branches (HISTORICAL — Do Not Use)

The user's fork at `/Users/clint/techblueprints/bleak-retry-connector` has feature branches from our earlier attempt to contribute upstream. **These are abandoned.** All the functionality from these branches has been reimplemented in BCM instead. They exist only as historical reference if you need to understand how a particular feature was originally designed.

| Branch | What it was | Now lives in BCM as |
|---|---|---|
| `feat/phantom-detection` | Detect and clear phantom connections | `diagnostics.py`, `bluez.py` |
| `feat/thread-safety-timer` | Thread-level safety timer | Built into `connection.py` |
| `feat/inprogress-classification` | Classify InProgress errors | `scanner.py`, `connection.py` |
| `feat/notification-watchdog` | ConnectionWatchdog for zombie detection | `watchdog.py` |
| `feat/enhanced-disconnect` | HCI-validated disconnect | `bluez.py` |
| `feat/multi-adapter-rotation` | Rotate between adapters on failure | `adapters.py`, `connection.py` |
| `feat/escalation-chain` | Progressive recovery escalation | `recovery.py` |
| `feat/validate-connection` | Post-connect validation callback | `validators.py`, `connection.py` |
| `feat/clear-stale-connections` | Clear stale connections on failure | `diagnostics.py` |
| `feat/cross-process-lock` | File-based per-adapter scan lock | `scan_lock.py`, `lock.py` |
| `downstream/all-features` | Was going to merge all features | Not needed — BCM is the downstream |

**Do not merge, rebase, or reopen PRs from these branches.** The `main` branch on this fork tracks upstream and should stay that way.

## vesmart Patch

Located at `/Users/clint/techblueprints/venus-os_dbus-serialbattery/vesmart-patch/`:

| File | Purpose |
|---|---|
| `vesmart_server.py` | Patched version that reads a suppression list |
| `gattserver.py` | Patched GATT server |
| `vesmart-protected-devices.conf` | List of MAC addresses to NOT disconnect |
| `vesmart-protect-install.sh` | Installer script |

The suppression list (`/data/vesmart-protected-devices.conf` on the Cerbo) must contain ALL BLE device MACs managed by our services. If a firmware update overwrites the patch, it must be reapplied.

## bleak-retry-improvements Directory

Located at `/Users/clint/techblueprints/bleak-retry-improvements/` (local only, not a git repo). Contains all planning documents:

| File | Purpose |
|---|---|
| `01-bleak-retry-connector-improvements.md` | Master plan: 7 proposed PRs with full implementation details |
| `02-07-migration-*.md` | Migration plans for each consuming project |
| `08-11-inventory-*.md` | BLE code inventory for each project |
| `12-observed-stuck-states.md` | **Critical**: 22 observed stuck states with diagnostics and recovery |
| `13-upstream-contribution-approach.md` | Strategy for engaging with the maintainer |
| `14-23-issue-*.md` | Draft issue text for each upstream issue to file |
| `24-wrapper-components.md` | BCM wrapper component design |

## Related GitHub Repositories

| Repo | URL | Purpose |
|---|---|---|
| BCM | https://github.com/TechBlueprints/bleak-connection-manager | Our connection manager wrapper |
| bleak-retry-connector | https://github.com/Bluetooth-Devices/bleak-retry-connector | Upstream retry library |
| bleak | https://github.com/hbldh/bleak | Core BLE library |
| bluetooth-adapters | https://github.com/Bluetooth-Devices/bluetooth-adapters | Adapter enumeration |
| aiooui | https://github.com/Bluetooth-Devices/aiooui | MAC vendor lookup |
| serialbattery | https://github.com/TechBlueprints/venus-os_dbus-serialbattery | Fork of dbus-serialbattery |
| power-watchdog | https://github.com/TechBlueprints/dbus-power-watchdog | Power Watchdog BLE driver |
| shyion-switch | https://github.com/TechBlueprints/dbus-shyion-switch | Shyion relay BLE driver |
| bleak-retry-improvements | Local only: `/Users/clint/techblueprints/bleak-retry-improvements/` | Issue/PR planning docs |

## Field validation — 2026-08-21, dev Cerbo (Venus 3.72)

First real v2 deployment: dbus-serialbattery `feat/bcm-v2` (vendored at
`bc31d9a`), CHINS/JBD battery via aiobmsble `jbd_bms`, `wrap_scanner=True`,
`link_caps={"hci1": 5}`, no adapters configured. **No BCM defects found.**

- The dead-card filter only worked because of `bc31d9a`'s hciconfig
  fallback: this Cerbo's dead onboard UART controller (hci0) reports an
  all-zeros MAC and the kernel exposes **no sysfs address attribute at
  all**. Every scan and connect routed to hci1; zero attempts on hci0.
- Device-path claim attribution (`b6a5496`) verified live: the driver's
  cache-resolved BLEDevice connected on hci1 and `/run/bt-claims` held
  `hci1.link.0` plus the MAC-qualified soft claim — on the adapter the
  link actually uses.
- Install-order contract held: catcher installed before BMS imports,
  aiobmsble connected through `BLEConnectionWithServiceCache`, no
  bare-connect warnings.
- Ecosystem note (not a BCM issue): sparse-advertising JBD modules defeat
  scan-based discovery outright; consumers should resolve devices
  cache-first (the serialbattery driver now does), which is also the path
  where device-path claim attribution applies.

### Resolved field issue (2026-08-21, dev Cerbo): claims vanish while the link persists

After a service restart and reconnect (aiobmsble jbd_bms via the catcher),
the LE link to A4:C1:38:33:41:24 stayed up on hci1 with live data flowing,
but `/run/bt-claims` was empty - both the link slot and the soft claim were
gone. First connect of the day held its claims for the whole session, so
this is reconnect-path specific. Effect was under-counting only
(placement/caps degrade to uncoordinated), but it defeated cross-process
occupancy scoring.

**Root cause** (reproduced in `tests/test_catcher.py`, reconnect-path claim
survival section): every connect generation's disconnected callback closes
over the same wrapper, and released claims unconditionally. When
bleak-retry-connector retries `connect()` on one instance, a late disconnect
event from the torn-down previous backend released the claims the newer
connect had acquired; no new `connect()` runs afterwards, so nothing ever
re-claimed. Two adjacent holes shared the mechanism: validity tied claim
life to the *wrapper object* rather than the link, and there was no path
back from "claims lost, link alive".

**Fix** (catcher.py): claim validity now tracks link truth, and the
accounting can heal itself.

- The disconnected callback releases only when it belongs to the current
  connect generation AND the wrapper's own view agrees the link is down.
  Stale and spurious events release nothing; if the link really died, the
  heartbeat sweeps within a beat.
- `_arm_claim_validity` counts recent notification traffic
  (`LINK_EVIDENCE_SECONDS` = `CLAIM_TTL`) as link truth even when
  `is_connected` reads False, and falls back to the backend's own connected
  state when the wrapper has been collected while the BlueZ link survives.
- `start_notify` taps the consumer's callback: every notification stamps
  the evidence clock, and if data arrives while the wrapper holds no live
  claims (and no intentional `disconnect()` ran), the slot and soft claim
  are re-acquired on the adapter the connect used and validity re-armed.
  Losing the slot race to another process degrades to soft-claim-only.

### Field validation — 2026-08-22, dev Cerbo: post-connect validators (482ff7f)

Re-vendored into the driver (16d11f9, inert - no validator configured) and
exercised live against the IP22 with a bare routed client:

- Positive: `validate_char_exists(306b0002-...)` - connect succeeded,
  slot + soft claim held, released on disconnect.
- Negative: bogus UUID - validator logged the device's full GATT table in
  the rejection, the link was torn down (is_connected False), claims
  released, and `ConnectionValidationError` raised as specified.
- Incidental: dead-pid claims from an earlier killed test run were reaped
  by the live manager within minutes - the reap convention works in anger.
- Regression: battery service reconnected at 482ff7f, claims held through
  heartbeats, no behavior change with no validator configured.

### Field issue and validation — 2026-08-22, prod Cerbo: claims lost for a polling client (9a2fd16)

First prod deployment (dbus-easytouchrv, two EasyTouch thermostats,
persistent connections, coexisting with two dbus-serialbattery claim
holders). After a messy restart — stale BlueZ links from the previous
process, manually dropped, then reconnects — one thermostat's claims were
gone from `/run/bt-claims` while its LE link was demonstrably up on hci0
(`hcitool -i hci0 con`) and status polling succeeded every 10s.

**Root cause**: not the stale-callback path — 371ce1b's generation guard
already covered that. `_recent_link_evidence` was fed only by the
`start_notify` tap, so for a consumer that only calls `read_gatt_char`,
validity collapsed to bare `is_connected`. bleak's BlueZ `is_connected`
reads a cached D-Bus properties dict which is stranded False when a device
object is removed and re-added (exactly what the stale-link cleanup does)
with no property signal to correct it. One transient false negative
released the claims on the next heartbeat, and the only re-arm path was one
a polling driver can never take. GATT reads kept working throughout because
they do not go through that cached view.

The consumer had already learned this independently: `easytouch_ble.py`'s
`_exchange()` has refused to gate on `is_connected` since the v1 field
debugging era, for the same reason. **BCM was trusting the exact property
that deployment had stopped trusting.**

**Fix** (9a2fd16): link truth is *observed traffic*, not a cached property.

- `read_gatt_char` / `write_gatt_char` / `read_gatt_descriptor` /
  `write_gatt_descriptor` stamp the evidence clock after the await. This
  both holds claims against a false negative and re-arms claims already
  lost, on the next poll. An operation that raised proves nothing.
- Auditing every other `is_connected` gate found a second, worse instance:
  `tolerate_late_gatt` abandoned re-validation on the property alone, and a
  validator rejection disconnects the link, releases its claims and rotates
  the radio. It now takes a failing `refresh_services()` — a real GATT
  re-read — corroborated by the property, before abandoning.
- The disconnected callback still releases on `is_connected` alone, by
  agreement with the consumer: on a capped shared card a delayed release
  costs a neighbour an out-of-slots error, while the current shape costs a
  <=10s occupancy undercount that self-heals on the next poll.

**Validation**: 9a2fd16 deployed to prod through the identical sequence that
produced the loss that morning (messy restart, bluetoothctl churn,
reconnects). Old code: claim swept in under 3 minutes. New code: both
claims present and stable past 4 minutes, both units polling normally. The
outcome was the *hold* path — no `traffic flowing ... re-claimed` line —
which by design logs nothing, so a silently-absorbed false negative and a
run with no false negative are indistinguishable from outside. Per rule 4
below, this is one window, not a stability verdict; the `re-claimed` line
is the tell if a spurious release ever does happen.

**Still open**: the stale-link-on-restart quirk itself. It has now hit 4 of
4 restarts-with-held-connections on this deployment, and the adapter the
stale link sits on varies between restarts — on one, the link was on hci1
where `bluetoothctl` could not see it at all (it only talks to the default
controller) and only `hcitool -i hci1 con` found it. Any detection must
enumerate per adapter. Tracked as its own work.

### Field notes — 2026-08-22, prod Airstream Cerbo (via the BCMv2 implementation session)

- **`tune_conn_params` is a silent no-op on Venus OS.** The platform
  Python (3.12.13 as shipped) has no `socket.AF_BLUETOOTH`, so the mgmt
  control socket cannot open and `mgmt.available()` is False. Every
  connect on both Cerbos to date ran without the fast-then-medium
  parameter loads — by-design degradation, now documented in the README.
  The same gap silently broke the native reset's ioctl interface bounce;
  fixed with an `hciconfig down`/`up` subprocess fallback (the runbook
  remedy, automated).
- **Orphaned driver process signature.** A TERM-immune leftover driver
  (survived `svc -t`) fought the supervised instance for the same battery:
  ~8s connect/disconnect flap for 45 minutes, indistinguishable from radio
  failure. BCMv2 exonerated (flap survived config rollback; killing the
  orphan ended it). The catcher now warns on connect when a second live
  instance of the same service (same owner base, different pid) holds a
  claim for the same device MAC: "another live instance of this service
  ... check for a leftover pid." If the drain watcher ever logs
  disconnects while such a claim exists, suspect the orphan, not the
  radio. Remedy per rule 7: `svc -d`, verify the pid actually exited,
  `kill -9` the survivor if not, `svc -u`.
