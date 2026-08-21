# -*- coding: utf-8 -*-
"""bleak-connection-manager v2: the embeddable bleak catcher.

Two layers, deliberately separable:

- ``bleak_connection_manager.claims``: the bt-claims file convention
  (stdlib only, no bleak, vendorable verbatim). Usable standalone by any
  service that wants adapter coordination without the bleak machinery.
- ``bleak_connection_manager.catcher``: the process-wide bleak client
  rebinding layer (requires bleak), exposed lazily here so importing the
  package - or the claims module - never drags bleak in.
"""

__version__ = "2.0.0.dev0"

_CATCHER_EXPORTS = (
    "install_bleak_catcher",
    "uninstall_bleak_catcher",
    "BLEConnection",
    "BLEConnectionWithServiceCache",
    "OutOfConnectionSlotsError",
)

__all__ = list(_CATCHER_EXPORTS) + ["claims"]


def __getattr__(name):
    if name in _CATCHER_EXPORTS:
        from bleak_connection_manager import catcher

        return getattr(catcher, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
