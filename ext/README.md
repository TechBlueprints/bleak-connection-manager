# Vendored libraries

## dbus_fast/

dbus-fast 5.0.22 (MIT, see DBUS_FAST_LICENSE), vendored verbatim from the
PyPI sdist's `src/dbus_fast` — pure Python; the `.pxd` files are inert
Cython acceleration headers and unused at runtime.

Why it's here: Venus OS ships `python3-dbus-fast` 2.21.1, while current
bleak (from f0b106e, "use dbus-fast typed annotations") requires
dbus-fast >= 4.0.0 — a fresh deploy of a vendored modern bleak dies on the
OS package with `No module named 'dbus_fast.annotations'`. Any Venus
consumer vendoring bleak needs to vendor a modern dbus-fast alongside it
and put it ahead of the system package on `sys.path`; this copy is the one
to take.

- Do not edit this copy; re-vendor from PyPI to update.

## bluetooth_adapters/ and aiooui/

bluetooth-adapters 2.4.0 (Apache-2.0, see BLUETOOTH_ADAPTERS_LICENSE) and
aiooui 0.1.9 (MIT, see AIOOUI_LICENSE), vendored verbatim from the PyPI
wheels (bluetooth-adapters re-vendored 2026-09-06 to meet bleak-retry-connector
4.7.0's declared floor of >= 2.3.0; same third-party imports as 2.1.1, and
still pure Python). Here because bleak-retry-connector hard-imports
bluetooth_adapters on Linux (and it in turn needs aiooui); part of the
shared-install stack served by /data/bcm.

- Do not edit these copies; re-vendor from PyPI to update.

## upstream/ (git submodules)

- upstream/bleak: github.com/hbldh/bleak, pinned at v3.0.2 (bb49377) -
  the exact tree the fleet field-validated (located by git tree-hash
  match against the serialbattery deployment's vendored copy).
- upstream/bleak-retry-connector: github.com/Bluetooth-Devices/
  bleak-retry-connector, pinned at v4.7.0 (84bda39); bumped from v4.6.0
  2026-09-06 on Clint's ruling - 4.6.1-4.6.3 are fixes (clear_cache guard,
  decorator retries EOFError/BrokenPipeError), 4.7.0 adds
  restore_discoveries_sync; establish_connection's behaviour is unchanged.

Per the standing rule, both pin UPSTREAM commits - never TechBlueprints
fork branches. Bump pins deliberately, one at a time, with a fleet soak.
