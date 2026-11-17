# Scan-claim detail: declaring what a passive scan listens for

**Status: UNIMPLEMENTED proposal.** Nothing described here exists in the code.
Tracking issue: see the link at the bottom (filled in when filed).

## Problem

A passive scanner that restricts the controller to an accept list (for
example a raw-HCI scan with the accept-list-only filter policy) must hold the
card's hard `.scan` claim. It reprograms the controller's filter policy, so
every advertisement from a device not on its list is discarded for every user
of that radio. Another service that wants to scan or connect on that card
cannot tell from the claim what is being listened for, so it must treat the
card as unavailable.

## Constraints

- Hardness stays decided by the writer, from its mechanism. A claim whose
  strength depends on the reader's version lets two readers disagree about
  whether a card is free, which is the double-booking the 0.5 hardlink work
  closed.
- Unknown claim content must read as an ordinary hard claim, so 0.3-0.5
  readers and shell participants are unaffected.
- The two exclusive names (`.scan`, `.drain`) are a hardlink to the holder
  file. Its inode is the ownership proof, so it must never be replaced.

## Proposal

Keep line 1 of the holder/lock file exactly as it is
(`<pid> <service> <since> [purpose]`). Existing readers only look at the first
whitespace-separated token (`_read_pid`), so lines after it are invisible to
them. Add `key value` lines after it, one per line, unknown keys ignored:

```
12345 some-service-12345 1757100000 scan
mode passive
mechanism hci-acceptlist
hears AA:BB:CC:DD:EE:FF
hears 11:22:33:44:55:66
```

One `hears` line per MAC, so there is no delimiter to sanitize. Accept lists
are hardware-capped (typically 8-32 entries), so the file stays small.
BCM's own BlueZ scans would write `mode active` / `mechanism bluez-discovery`.

### Writing

The list changes when the writer reslices its accept list, while other
processes snapshot the file. Rules:

- Never temp-and-rename: that changes the inode and breaks ownership.
- Never truncate line 1: an empty file reads as no pid, so a snapshot would
  report the card free for a tick. `pwrite` the new body at the offset just
  past line 1, then `ftruncate` to the new length.
- Readers treat a malformed line as absent and a file with no body as an
  opaque hard claim.

### Code touched

- `ClaimManager`: a `describe(claim, lines)` that does the in-place body write.
- `claims()` snapshot: a parsed `hard_detail` dict next to `hard` / `hard_pid`.
- A `hearing(mac)` helper: which adapter's scan is listening for a MAC.
- The passive-scan writer calls `describe` after each accept-list change.
- Placement and the scan gate are unchanged.

## What this does not claim

A MAC list does not make it safe for a second service to start its own
discovery on that card. The passive scanner re-asserts its filter policy
periodically and other scans reset the scan parameters, so the two would
trade the controller's scan state back and forth, and the second service would
see only the first one's MACs with no scan responses. The list answers "who is
already listening for this MAC", so a service can consume that scanner's
advertisement stream instead of starting a second scan. Connects already rank
rather than gate, and are unchanged.

## Open questions

- Key names and the mechanism vocabulary are a sketch; settle them with the
  first consumer.
- Whether this bumps the convention version (0.6) or counts as an optional
  extension of 0.5.
