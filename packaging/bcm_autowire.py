# -*- coding: utf-8 -*-
"""Sitewide bleak catcher autowire - planted by BCM's install.sh --autowire.

Loaded by bcm_autowire.pth at the startup of EVERY Python interpreter on
the machine, which imposes the one absolute rule of this file: it must
never raise and never slow anything down. At import it does exactly one
cheap thing - register a meta-path finder - and goes dormant. Only when a
process imports bleak does anything else happen: the finder lets the real
import complete (serving the shared checkout's bleak to processes that
have none of their own; NEVER overriding one the process brought), then
installs the bleak catcher over it before the importing code can capture
bleak.BleakClient - so every Python bleak consumer on the box gets
claim-aware slotting and latching without knowing this package exists.

Scope honesty: this covers Python+bleak only. C programs talking to BlueZ
(Victron's dbus-ble-sensors) and bluetoothctl remain outside the claims
convention, autowired or not.

Config: /data/bcm/autowire.conf (JSON), keys matching
install_bleak_catcher: adapters, link_caps, wrap_scanner, scan_to_score,
tune_conn_params. Owner defaults to the process name. Kill switch:
BCM_AUTOWIRE=0 in the environment.

Never from inside BCM: a process that imports bleak_connection_manager
itself is a deliberate consumer - it installs the catcher explicitly,
and the finder stands down the moment the package appears in
sys.modules. The /data/bcm/python3 shim also exports BCM_AUTOWIRE=0,
because every shim-launched process is by definition such a consumer.
"""

import os
import sys

_ROOT = os.environ.get("BCM_AUTOWIRE_ROOT", "/data/bcm")
_CONF = os.path.join(_ROOT, "autowire.conf")
_EVENTS = os.path.join(_ROOT, "autowire-events.log")
_EVENTS_MAX = 1_000_000  # a crash-looping script must not grow it forever
_served_shared = False


def _lib_paths():
    return [
        os.path.join(_ROOT, "src"),
        os.path.join(_ROOT, "ext"),
        os.path.join(_ROOT, "ext", "upstream", "bleak"),
        os.path.join(_ROOT, "ext", "upstream", "bleak-retry-connector", "src"),
    ]


def _record_wire(owner):
    """One durable line per wired process: /data/bcm answers "what BLE
    software has ever run on this box" after the fact - the wired
    process's own logging is unconfigured in exactly the community
    scripts autowire exists for, and claim files are tmpfs. Same
    never-raise rule as everything here; skips silently when /data/bcm
    is unwritable or the log has hit its size guard."""
    try:
        import time

        try:
            if os.path.getsize(_EVENTS) > _EVENTS_MAX:
                return
        except OSError:
            pass
        bleak_mod = sys.modules.get("bleak")
        source = "shared" if _served_shared else "own"
        origin = getattr(bleak_mod, "__file__", None) or "?"
        version = getattr(bleak_mod, "__version__", "") or ""
        argv = " ".join(sys.argv).replace("\n", " ") or "?"
        try:
            cwd = os.getcwd()
        except OSError:
            cwd = "?"
        line = (
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()} owner={owner} "
            f"bleak={source}{f' {version}' if version else ''} ({origin}) "
            f"argv=\"{argv}\" cwd={cwd}\n"
        )
        fd = os.open(_EVENTS, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8", "replace"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _install_catcher():
    try:
        for p in _lib_paths():
            # append, never prepend: the process's own choices always win
            if p not in sys.path and os.path.isdir(p):
                sys.path.append(p)
        from bleak_connection_manager import install_bleak_catcher

        config = {"wrap_scanner": True}
        try:
            import json

            with open(_CONF) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception:
            pass  # no conf, or a broken one: fleet defaults
        owner = os.path.basename(sys.argv[0] or "") or "python"
        if owner.startswith("python"):
            owner = "autowired-python"
        config.setdefault("adapters", ())
        install_bleak_catcher(f"autowire-{owner}", **config)
        _record_wire(f"autowire-{owner}")
    except Exception:
        # a broken autowire must never break the process it rode into
        try:
            import logging

            logging.getLogger("bcm_autowire").debug("autowire failed", exc_info=True)
        except Exception:
            pass


class _Finder:
    """Meta-path finder: dormant until 'bleak' is imported."""

    def __init__(self):
        self._busy = False
        self._done = False

    def find_spec(self, name, path=None, target=None):
        if self._done or self._busy or name != "bleak":
            return None
        if "bleak_connection_manager" in sys.modules:
            # the bleak import came from (or after) BCM itself: this
            # process is BCM-aware and will call install_bleak_catcher
            # explicitly with its own owner and config. Autowiring here
            # would install a generic catcher from INSIDE the package's
            # own import - stand down permanently instead.
            self._done = True
            return None
        self._busy = True
        try:
            import importlib.util

            spec = importlib.util.find_spec("bleak")
            if spec is None:
                # no bleak of its own: serve the shared checkout's
                global _served_shared
                _served_shared = True
                shared = os.path.join(_ROOT, "ext", "upstream", "bleak")
                dbus_fast_home = os.path.join(_ROOT, "ext")
                for p in (dbus_fast_home, shared):
                    if p not in sys.path and os.path.isdir(p):
                        sys.path.insert(0, p)
                spec = importlib.util.find_spec("bleak")
            if spec is None or spec.loader is None:
                return None
            self._done = True
            real_exec = spec.loader.exec_module

            def exec_module(module):
                real_exec(module)
                # bleak is fully imported; rebind before the importer's
                # own `from bleak import BleakClient` can run
                _install_catcher()

            try:
                spec.loader.exec_module = exec_module
            except AttributeError:
                # a loader we cannot wrap: import proceeds uncaught
                return None
            return spec
        except Exception:
            return None
        finally:
            self._busy = False


try:
    if os.environ.get("BCM_AUTOWIRE", "1") != "0" and os.path.isdir(_ROOT):
        sys.meta_path.insert(0, _Finder())
except Exception:
    pass
