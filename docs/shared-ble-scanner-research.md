# Shared BLE Scanner Daemon: IPC Options Research

**Target:** Cerbo GX running Venus OS with BusyBox  
**Goal:** ONE process owns BLE scanning; multiple independent Python processes consume scan results  
**Python:** 3.12

---

## Critical Finding: BlueZ Already Shares Discovery

**Yes — BlueZ shares scan results across D-Bus clients.** This fundamentally changes the architecture.

### How BlueZ Discovery Works

From the [BlueZ Adapter API](https://manpages.debian.org/unstable/bluez/org.bluez.Adapter.5.en.html):

> "A discovery procedure is **shared between all discovery sessions** thus calling StopDiscovery will only release a single session and discovery will stop when **all sessions from all clients have finished**."

When Process A calls `StartDiscovery()` on an adapter:

1. **Process B** (and C, D, …) subscribing to:
   - `org.freedesktop.DBus.ObjectManager.InterfacesAdded` — receives new Device objects as they appear
   - `org.freedesktop.DBus.Properties.PropertiesChanged` with `arg0="org.bluez.Device1"` — receives property updates (RSSI, ManufacturerData, etc.)

2. **Will receive** all devices that A discovers. No custom daemon needed to "relay" data.

### The Real Problem

The issue is **multiple processes each calling `StartDiscovery()`**:

- Each call creates a new "session" from BlueZ's perspective
- Discovery only stops when *all* sessions call `StopDiscovery()`
- If processes start/stop discovery independently, you get `org.bluez.Error.InProgress` when another process is already scanning
- Your `bleak-connection-manager` already mitigates this with `fcntl.flock` scan locks

### Minimal Solution: Single StartDiscovery Owner

**Option A — No custom daemon:** Designate ONE long-lived process as the "discovery owner":

- That process calls `StartDiscovery()` once and keeps its D-Bus connection alive
- All other processes **never** call `StartDiscovery()`; they only subscribe to `InterfacesAdded` and `PropertiesChanged`
- Clients get real-time device discovery via D-Bus signals — no IPC layer

**Requirement:** The owner must keep its D-Bus proxy connection to the adapter alive. If it exits or disconnects, discovery stops.

---

## IPC Mechanism Comparison

Assuming you still want a **custom scanner daemon** (e.g. for adapter rotation, filtering, or caching), here is the evaluation.

### 1. D-Bus Service

**Model:** Daemon registers `com.victronenergy.ble.scanner` (or similar), exposes methods like `GetDiscoveredDevices()` and signals like `DeviceDiscovered`.

| Criterion | Assessment |
|-----------|------------|
| **Latency** | Low. D-Bus signals are push-based; clients get notified immediately. Method calls for snapshots add ~1–5 ms. |
| **Complexity** | Medium. Need to define D-Bus interface, implement service, handle `dbus-fast` or `pydbus`. Venus OS already uses D-Bus heavily (`com.victronenergy.system`, etc.). |
| **Reliability** | Good. If daemon crashes, clients see `NameOwnerChanged`; D-Bus itself is supervised. No stale file handles. |
| **Resource usage** | Low. D-Bus is already running; one extra service adds minimal overhead. |
| **Venus OS / BusyBox** | Excellent. D-Bus is core to Venus; `dbus-fast` works on Python 3.12. |
| **Prevents multiple StartDiscovery** | Yes, if the daemon is the only process calling it. |

**Pros:** Fits Venus ecosystem, observability via `dbus-spy`, matches existing patterns.  
**Cons:** More code than "just listen to BlueZ"; need service lifecycle (start/stop, restart).

---

### 2. Unix Domain Socket

**Model:** Daemon listens on `/run/ble-scanner.sock`; clients connect; daemon pushes scan results (e.g. JSON lines) to each connected client.

| Criterion | Assessment |
|-----------|------------|
| **Latency** | Low. Push model; data arrives as soon as daemon writes. |
| **Complexity** | Medium. Need server loop, client connection handling, reconnection logic. Protocol design (framing, JSON vs binary). |
| **Reliability** | Good. Kernel-managed; if daemon dies, clients get EOF and can reconnect. |
| **Resource usage** | Low. One socket per client; small buffers. |
| **Venus OS / BusyBox** | Good. `socket.AF_UNIX` is standard; Python `asyncio` supports it. BusyBox doesn't affect Python sockets. |
| **Prevents multiple StartDiscovery** | Yes, if daemon is sole scanner. |

**Pros:** Flexible, no D-Bus dependency for this path, supports multiple concurrent clients.  
**Cons:** Custom protocol, no built-in discovery (clients must know socket path).

---

### 3. Shared Memory / mmap

**Model:** Daemon mmaps a file (e.g. `/run/ble-scan.bin`); writes device structs; clients mmap same file and read.

| Criterion | Assessment |
|-----------|------------|
| **Latency** | Very low for reads. |
| **Complexity** | High. Fixed-size buffer, synchronization (version counter, locks, or atomic updates), serialization format. |
| **Reliability** | Fragile. Daemon crash leaves stale data; no notification when new data arrives (clients must poll). |
| **Resource usage** | Low if buffer is small (e.g. 32–64 KB). `multiprocessing.shared_memory` uses `/dev/shm`; on some embedded systems `/dev/shm` can be small and cause `SIGBUS`. |
| **Venus OS / BusyBox** | Risky. Python 3.8+ has `multiprocessing.shared_memory`; file-backed mmap works, but sync and crash handling add complexity. |
| **Prevents multiple StartDiscovery** | Yes, if daemon is sole scanner. |

**Pros:** Fastest reads.  
**Cons:** Complex, poor fit for variable-length scan data, no push semantics, embedded `/dev/shm` concerns.

---

### 4. Named Pipe (FIFO)

**Model:** Daemon writes to a FIFO; clients read.

| Criterion | Assessment |
|-----------|------------|
| **Latency** | Low for single reader. |
| **Complexity** | Low for 1 writer, 1 reader. |
| **Reliability** | Poor for multiple readers. |
| **Resource usage** | Minimal. |
| **Venus OS / BusyBox** | BusyBox has `mkfifo`; Python supports FIFOs. Non-blocking FIFO handling in BusyBox is limited. |
| **Prevents multiple StartDiscovery** | Yes, if daemon is sole scanner. |

**Critical limitation:** **FIFOs cannot broadcast.** Data read by one process is consumed; other readers get nothing. To support multiple clients you'd need one FIFO per client and the daemon would have to write to each — equivalent complexity to a Unix socket server.

**Verdict:** Not suitable for multiple consumers.

---

### 5. BlueZ D-Bus Signals Directly (No Custom Daemon)

**Model:** One process owns `StartDiscovery()`; all others subscribe to BlueZ `InterfacesAdded` and `PropertiesChanged` only.

| Criterion | Assessment |
|-----------|------------|
| **Latency** | Lowest. Direct D-Bus signals from BlueZ. |
| **Complexity** | Lowest. No custom daemon, no custom IPC. Clients use `dbus-fast` to add signal receivers. |
| **Reliability** | Good. BlueZ is system service; if owner crashes, discovery stops but clients can detect and potentially become the new owner (with coordination). |
| **Resource usage** | Minimal. Reuses existing D-Bus. |
| **Venus OS / BusyBox** | Excellent. |
| **Prevents multiple StartDiscovery** | Yes — by design. Only the designated owner calls it. |

**Pros:** Simplest, no extra process, uses BlueZ as intended.  
**Cons:** Need clear process ownership (who calls `StartDiscovery` and when). Coordination if owner crashes (e.g. leader election or systemd-managed single instance).

---

### 6. File-Based (JSON/msgpack)

**Model:** Daemon writes `/run/ble-scan.json` (or `.msgpack`) periodically; clients poll and read.

| Criterion | Assessment |
|-----------|------------|
| **Latency** | High. Depends on poll interval (e.g. 1–5 s). |
| **Complexity** | Low. Write file, read file. |
| **Reliability** | Moderate. Daemon crash leaves last written file; clients get stale data until next write. |
| **Resource usage** | Low. Small JSON/msgpack files. |
| **Venus OS / BusyBox** | Good. Standard file I/O. |
| **Prevents multiple StartDiscovery** | Yes, if daemon is sole scanner. |

**Pros:** Very simple, easy to debug (inspect file).  
**Cons:** Polling latency, write amplification, no push.

---

## Summary Table

| Mechanism | Latency | Complexity | Reliability | RAM | Venus/BusyBox | Multi-Client | Prevents Multi-StartDiscovery |
|-----------|---------|------------|-------------|-----|---------------|--------------|------------------------------|
| D-Bus service | Low | Medium | Good | Low | Excellent | Yes | Yes |
| Unix socket | Low | Medium | Good | Low | Good | Yes | Yes |
| mmap | Very low | High | Fragile | Low | Risky | Yes | Yes |
| FIFO | Low | Low (1:1 only) | N/A | Minimal | OK | **No** | Yes |
| **BlueZ signals only** | **Lowest** | **Lowest** | **Good** | **Minimal** | **Excellent** | **Yes** | **Yes** |
| File-based | High | Low | Moderate | Low | Good | Yes | Yes |

---

## Recommendation

### Primary: BlueZ Signals Only (Option 5)

**Implement a "discovery owner" pattern without a custom scanner daemon:**

1. **Single owner process** (e.g. a lightweight `ble-discovery-owner` or the first BLE-dependent service like `dbus-serialbattery`) calls `StartDiscovery()` and keeps its D-Bus connection alive.
2. **All other processes** never call `StartDiscovery()`. They subscribe to:
   - `InterfacesAdded` on `org.freedesktop.DBus.ObjectManager` (service `org.bluez`, path `/`)
   - `PropertiesChanged` on `org.freedesktop.DBus.Properties` with `arg0="org.bluez.Device1"`
3. Use your existing **scan lock** (`fcntl.flock`) so that only the owner can acquire the lock; others skip scanning and rely on signals + BlueZ cache (as in `_find_in_bluez_cache`).

**Benefits:**

- No new daemon
- No custom IPC
- Uses BlueZ as designed
- Aligns with your current `bleak-connection-manager` architecture
- Venus OS already runs D-Bus; `dbus-fast` is used by bleak

**Implementation sketch:**

- Add a small module that either (a) holds `StartDiscovery` in a long-lived loop, or (b) is started by systemd and stays up
- Other services check "is discovery active?" (e.g. `Discovering` property on adapter) and, if so, only subscribe to signals
- If owner exits, discovery stops; a new owner can acquire the scan lock and call `StartDiscovery` again

### Fallback: D-Bus Service (Option 1)

If you need a dedicated daemon (e.g. for adapter rotation, filtering, or a stable API), implement a **D-Bus service**:

- Register `com.victronenergy.ble.scanner` (or similar)
- Expose `StartDiscovery` / `StopDiscovery` and `DeviceDiscovered` signal
- Clients call methods or subscribe to signals
- Fits Venus OS patterns and tooling (`dbus-spy`, `dbus -y`)

### Avoid

- **FIFO** — does not support multiple readers
- **mmap** — high complexity and sync issues for little gain
- **File polling** — unless latency of several seconds is acceptable

---

## References

- [BlueZ Adapter API (Debian manpage)](https://manpages.debian.org/unstable/bluez/org.bluez.Adapter.5.en.html)
- [BlueZ: Shared discovery across clients](https://blog.linumiz.com/archives/201)
- [Venus OS D-Bus wiki](https://github.com/victronenergy/venus/wiki/dbus)
- [FIFO multiple readers limitation](https://stackoverflow.com/questions/1634580/named-pipes-fifos-on-unix-with-multiple-readers)
- [Python multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
- `bleak-connection-manager` scan lock and `_find_in_bluez_cache` implementation
