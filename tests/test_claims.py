# -*- coding: utf-8 -*-
"""Tests for the bt-claims convention implementation (0.2: link slots added).

The first block is the bt-claims 0.1 reference suite, ported; the second
covers the 0.2 additions - numbered exclusive link slots, the qualified soft
claims, the public claims() snapshot, and release_all(). Foreign live
processes are simulated with pid-1 claim files (kill(1, 0) raises
PermissionError, which counts as alive); stale ones with a dead pid and an
aged mtime.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bleak_connection_manager.claims import ClaimManager  # noqa: E402


def _manager(tmp_path, owner="svc-a"):
    return ClaimManager(owner=owner, claim_dir=str(tmp_path))


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def _foreign_file(tmp_path, name, pid=1, aged=None):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as f:
        f.write(f"{pid} foreign-svc {int(time.time())}\n")
    if aged:
        _age(path, aged)
    return path


# -- ported from the bt-claims 0.1 reference suite -------------------------


def test_a_hard_claim_is_exclusive_and_a_racing_claimant_loses(tmp_path):
    a = _manager(tmp_path, "scanner-a")
    b = _manager(tmp_path, "scanner-b")
    claim = a.claim_hard("hci4")
    assert claim is not None
    assert b.claim_hard("hci4") is None
    a.release(claim)
    reclaimed = b.claim_hard("hci4")
    assert reclaimed is not None
    b.release(reclaimed)


def test_a_stale_hard_claim_is_reaped_and_taken(tmp_path):
    """A dead scanner must not hold its card forever: dead pid + old mtime = free."""
    a = _manager(tmp_path)
    _foreign_file(tmp_path, "hci4.scan", pid=99999999, aged=3600)
    claim = a.claim_hard("hci4")
    assert claim is not None
    a.release(claim)


def test_a_crashed_holders_claim_is_dead_immediately_not_after_the_ttl(tmp_path):
    """
    The pid check is what makes crash detection instant: a dead process with
    a still-fresh heartbeat file must not hold its card for the TTL tail.
    """
    b = _manager(tmp_path, "scanner-b")
    _foreign_file(tmp_path, "hci4.scan", pid=99999999)  # dead pid, fresh mtime
    taken = b.claim_hard("hci4")
    assert taken is not None
    b.release(taken)


def test_a_wedged_but_alive_holder_loses_its_claim_after_the_ttl(tmp_path):
    """
    Liveness needs BOTH a running pid and a fresh heartbeat. A hung scanner
    that stops beating must not hold its card forever; the TTL is the bound
    on how long a wedge can monopolize an adapter.
    """
    a = _manager(tmp_path, "scanner-a")
    b = _manager(tmp_path, "scanner-b")
    claim = a.claim_hard("hci4")
    _age(claim.path, 3600)  # pid alive, heartbeat long overdue
    taken = b.claim_hard("hci4")
    assert taken is not None
    b.release(taken)


def test_placement_avoids_a_hard_claimed_adapter(tmp_path):
    scanner = _manager(tmp_path, "scanner")
    battery = _manager(tmp_path, "battery")
    hard = scanner.claim_hard("hci1")
    adapter, claim = battery.choose(["hci1", "hci2"])
    try:
        assert adapter == "hci2"
    finally:
        battery.release(claim)
        scanner.release(hard)


def test_placement_prefers_the_less_claimed_adapter(tmp_path):
    other = _manager(tmp_path, "other-service")
    battery = _manager(tmp_path, "battery")
    theirs = other.claim_soft("hci1")
    adapter, claim = battery.choose(["hci1", "hci2"])
    try:
        assert adapter == "hci2"
    finally:
        battery.release(claim)
        other.release(theirs)


def test_soft_claims_share_when_there_is_no_alternative(tmp_path):
    """Soft means soft: a fully-claimed world ranks, it never refuses."""
    other = _manager(tmp_path, "other-service")
    battery = _manager(tmp_path, "battery")
    held = [other.claim_soft("hci1"), other.claim_soft("hci2")]
    adapter, claim = battery.choose(["hci1", "hci2"])
    try:
        assert adapter in ("hci1", "hci2")
        assert claim is not None
    finally:
        battery.release(claim)
        for h in held:
            other.release(h)


def test_a_hard_claim_never_keeps_a_battery_off_the_air(tmp_path):
    scanner = _manager(tmp_path, "scanner")
    battery = _manager(tmp_path, "battery")
    hard = scanner.claim_hard("hci1")
    adapter, claim = battery.choose(["hci1"])
    try:
        assert adapter == "hci1"
    finally:
        battery.release(claim)
        scanner.release(hard)


def test_an_unusable_claim_directory_degrades_to_uncoordinated(tmp_path):
    m = ClaimManager(owner="battery", claim_dir="/proc/definitely/not/writable")
    adapter, claim = m.choose(["hci1", "hci2"])
    assert adapter == "hci1"
    assert claim is None


def test_claim_files_carry_pid_owner_and_since(tmp_path):
    m = _manager(tmp_path, "svc")
    claim = m.claim_soft("hci1")
    try:
        with open(claim.path) as f:
            pid, service, since = f.read().split()
        assert int(pid) == os.getpid()
        assert service == "svc"  # the owner, so `ls`-level debugging names the holder
        assert int(since) > 0
    finally:
        m.release(claim)


# -- 0.2 additions: link slots, qualifiers, snapshot, release_all ----------


def test_link_slots_are_numbered_and_capped(tmp_path):
    m = _manager(tmp_path)
    first = m.claim_slot("hci1", 2)
    second = m.claim_slot("hci1", 2)
    try:
        assert first.path.endswith("hci1.link.0")
        assert second.path.endswith("hci1.link.1")
        assert m.claim_slot("hci1", 2) is None  # cap reached
    finally:
        m.release(first)
        m.release(second)


def test_a_foreign_live_slot_holder_blocks_the_slot(tmp_path):
    m = _manager(tmp_path)
    _foreign_file(tmp_path, "hci1.link.0", pid=1)  # pid 1 is alive
    assert m.claim_slot("hci1", 1) is None


def test_a_stale_slot_is_reaped_and_taken(tmp_path):
    m = _manager(tmp_path)
    _foreign_file(tmp_path, "hci1.link.0", pid=99999999, aged=3600)
    claim = m.claim_slot("hci1", 1)
    assert claim is not None
    m.release(claim)


def test_a_crashed_slot_holder_is_freed_immediately(tmp_path):
    """Dead pid with a fresh heartbeat: the slot frees instantly, like a hard claim."""
    m = _manager(tmp_path)
    _foreign_file(tmp_path, "hci1.link.0", pid=99999999)
    claim = m.claim_slot("hci1", 1)
    assert claim is not None
    m.release(claim)


def test_a_wedged_slot_holder_is_freed_after_the_ttl(tmp_path):
    a = _manager(tmp_path, "svc-a")
    b = _manager(tmp_path, "svc-b")
    claim = a.claim_slot("hci1", 1)
    _age(claim.path, 3600)  # pid alive, heartbeat long overdue
    taken = b.claim_slot("hci1", 1)
    assert taken is not None
    b.release(taken)


def test_releasing_a_slot_frees_it_for_another_claimant(tmp_path):
    a = _manager(tmp_path, "svc-a")
    b = _manager(tmp_path, "svc-b")
    claim = a.claim_slot("hci1", 1)
    assert b.claim_slot("hci1", 1) is None
    a.release(claim)
    taken = b.claim_slot("hci1", 1)
    assert taken is not None
    b.release(taken)


def test_an_unusable_directory_degrades_slots_to_a_phantom_claim(tmp_path):
    """Capacity accounting must never gate connections it cannot see: an
    unusable directory yields a truthy no-op claim, not exhaustion."""
    m = ClaimManager(owner="svc", claim_dir="/proc/definitely/not/writable")
    claim = m.claim_slot("hci1", 1)
    assert claim is not None
    claim.touch()  # no-op, must not raise
    m.release(claim)  # harmless


def test_soft_claims_can_be_qualified_per_connection(tmp_path):
    m = _manager(tmp_path, "svc")
    one = m.claim_soft("hci1", qualifier="C8478C000001")
    two = m.claim_soft("hci1", qualifier="C8478C000002")
    try:
        assert one.path != two.path
        assert one.path.endswith(".C8478C000001")
        snapshot = m.claims()
        assert snapshot["hci1"]["soft"] == 2
    finally:
        m.release(one)
        m.release(two)


def test_the_claims_snapshot_reports_hard_pid_soft_owners_and_links(tmp_path):
    m = _manager(tmp_path, "svc")
    hard = m.claim_hard("hci1")
    soft = m.claim_soft("hci1", qualifier="C8478C000001")
    slot = m.claim_slot("hci1", 3)
    try:
        entry = m.claims()["hci1"]
        assert entry["hard"] == hard.path
        assert entry["hard_pid"] == os.getpid()
        assert entry["soft"] == 1
        assert entry["soft_owners"] == ["svc.C8478C000001"]
        assert entry["links"] == 1
    finally:
        m.release_all()


def test_placement_ranking_counts_held_links_like_soft_claims(tmp_path):
    other = _manager(tmp_path, "other-service")
    battery = _manager(tmp_path, "battery")
    slot = other.claim_slot("hci1", 3)
    adapter, claim = battery.choose(["hci1", "hci2"])
    try:
        assert adapter == "hci2"
    finally:
        battery.release(claim)
        other.release(slot)


def test_a_claim_whose_validity_check_fails_is_released_on_the_beat(tmp_path):
    m = _manager(tmp_path)
    claim = m.claim_soft("hci1")
    claim.validity = lambda: False
    m._beat_once()
    assert os.listdir(str(tmp_path)) == []


def test_a_broken_validity_check_never_drops_the_claim(tmp_path):
    """A claim wrongly held is bounded by process life; a claim wrongly
    released overcommits the card. Errors keep the claim."""
    m = _manager(tmp_path)
    claim = m.claim_soft("hci1")

    def boom():
        raise RuntimeError("check failed")

    claim.validity = boom
    m._beat_once()
    assert os.path.exists(claim.path)
    m.release(claim)


def test_the_beat_touches_claims_that_are_still_valid(tmp_path):
    m = _manager(tmp_path)
    claim = m.claim_soft("hci1")
    _age(claim.path, 20)
    claim.validity = lambda: True
    m._beat_once()
    assert time.time() - os.stat(claim.path).st_mtime < 5
    m.release(claim)


def test_foreign_use_counts_only_live_claims_from_other_processes(tmp_path):
    m = _manager(tmp_path, "svc")
    own = m.claim_soft("hci1")
    _foreign_file(tmp_path, "hci1.link.0", pid=1)  # live foreign
    _foreign_file(tmp_path, "hci1.scan", pid=99999999, aged=3600)  # stale foreign
    _foreign_file(tmp_path, "hci2.use.other", pid=1)  # different adapter
    try:
        assert m.foreign_use("hci1") == 1
        assert m.foreign_use("hci2") == 1
        assert m.foreign_use("hci3") == 0
    finally:
        m.release(own)


def test_release_all_releases_everything(tmp_path):
    m = _manager(tmp_path, "svc")
    m.claim_hard("hci1")
    m.claim_soft("hci2", qualifier="C8478C000001")
    m.claim_slot("hci1", 2)
    assert len(os.listdir(str(tmp_path))) == 3
    m.release_all()
    assert os.listdir(str(tmp_path)) == []
