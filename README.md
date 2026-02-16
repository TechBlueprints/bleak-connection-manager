# bleak-connection-manager

Robust BLE connection lifecycle manager for Linux (BlueZ).

Wraps [bleak-retry-connector](https://github.com/Bluetooth-Devices/bleak-retry-connector) with production-hardened workarounds for real-world BlueZ failure modes: phantom connections, InProgress errors, zombie connections, adapter saturation, and more.

## Where it fits

```
Your application
    └── bleak-connection-manager      ← this project
            ├── bleak-retry-connector  (single-attempt connection + error classification)
            ├── bluetooth-adapters     (adapter enumeration)
            ├── bluetooth-auto-recovery (adapter reset via MGMT socket / USB)
            ├── dbus-fast              (D-Bus queries for phantom detection)
            └── bleak                  (raw BLE API)
                └── BlueZ
```

## What it does

Each feature is independently removable — when upstream fixes the root cause, the corresponding workaround is deleted:

| Feature | Problem solved |
|---------|---------------|
| **Phantom detection** | `Connected=True` but `ServicesResolved!=True` — dead connection occupying a slot |
| **InProgress classification** | `org.bluez.Error.InProgress` from stale BlueZ state, not a real conflict |
| **Post-connect validation** | Connection succeeds but GATT is empty or device is unresponsive |
| **Adapter rotation** | Round-robin across adapters when one is saturated |
| **Notification watchdog** | Detect zombie connections via data silence |
| **Cross-process lock** | Serialize BLE operations when multiple services share an adapter |
| **Thread safety timer** | Unblock frozen asyncio event loops via out-of-band thread |
| **Escalation chain** | Track failures and escalate: retry → clear → rotate → reset |
| **Clear stale on retry** | Remove BlueZ cached state between attempts |
| **Enhanced disconnect** | Timeout-guarded disconnect with phantom cleanup |

## Installation

```bash
pip install bleak-connection-manager
```

## Quick start

```python
from bleak_connection_manager import establish_connection
from bleak import BleakClient

# Drop-in replacement with all workarounds enabled
client = await establish_connection(
    BleakClient,
    device,
    device.name,
    max_attempts=4,
)

# Or opt in to specific features
from bleak_connection_manager import ConnectionWatchdog, LockConfig

client = await establish_connection(
    BleakClient,
    device,
    device.name,
    max_attempts=4,
    close_inactive_connections=True,  # phantom detection
    validate_connection=my_validator,  # post-connect check
    lock_config=LockConfig(enabled=True),  # cross-process lock
)

# Monitor for zombie connections
watchdog = ConnectionWatchdog(
    timeout=30.0,
    on_timeout=my_reconnect,
    client=client,
    device=device,
)
watchdog.start()
```

## License

Apache License 2.0
