# Vendored libraries

## bt_claims.py

The bt-claims reference implementation, vendored for anyone who wants the
plain adapter-claims coordination (`/run/bt-claims` file convention) without
adopting this package's bleak catcher stack — copy the one file, or import
it from here; it is stdlib-only with no dependencies on this repository.

- Source: https://github.com/TechBlueprints/bt-claims
  (local: /Users/clint/techblueprints/bt-claims), commit 78eb45b,
  version 0.2.0, license MIT (see BT_CLAIMS_LICENSE).
- Do not edit this copy; fix things upstream and re-vendor.

As of 0.2 the upstream `bt_claims.py`, this vendored copy, and
`bleak_connection_manager.claims` in this package are **byte-identical** —
one implementation of the full convention (scan claims, qualified soft
claims, `hciN.link.<k>` slots, `claims()` snapshot, `foreign_use`, heartbeat
validity checks), published under two module names. Import whichever name
suits your tree; keep all three in lockstep when re-vendoring.
