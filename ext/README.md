# Vendored libraries

## bt_claims.py

The bt-claims reference implementation, vendored verbatim for anyone who
wants the plain adapter-claims coordination (`/run/bt-claims` file
convention) without adopting this package's bleak catcher stack — copy the
one file, or import it from here; it is stdlib-only with no dependencies on
this repository.

- Source: https://github.com/TechBlueprints/bt-claims
  (local: /Users/clint/techblueprints/bt-claims), commit c7ff3f2,
  version 0.1.0, license MIT (see BT_CLAIMS_LICENSE).
- Do not edit this copy; fix things upstream and re-vendor.

Note: `bleak_connection_manager.claims` in this package is a strict superset
of this file implementing the 0.2 convention (adds numbered exclusive
`hciN.link.<k>` link slots, qualified soft claims, `release_all()`, and a
public `claims()` snapshot; `Claim.hard` is renamed `Claim.exclusive`). It
is equally standalone and vendorable. This 0.1 copy is kept as the canonical
upstream convention reference until the 0.2 convention bump lands in
bt-claims itself.
