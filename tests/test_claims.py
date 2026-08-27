# -*- coding: utf-8 -*-
"""Tests for the claims convention implementation (0.4: MAC-keyed adapters).

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

from bleak_connection_manager import claims  # noqa: E402
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
    # four files, not three: the hard claim is a lock plus its holder (0.5)
    assert len(os.listdir(str(tmp_path))) == 4
    m.release_all()
    assert os.listdir(str(tmp_path)) == []


# -- drain (convention 0.3) ------------------------------------------------


def test_the_drain_claim_is_exclusive(tmp_path):
    m = _manager(tmp_path, "resetter")
    other = _manager(tmp_path, "second")
    drain = m.claim_drain("hci1")
    try:
        assert drain is not None
        assert os.path.exists(os.path.join(str(tmp_path), "hci1.drain"))
        assert other.claim_drain("hci1") is None  # one recovery at a time
    finally:
        m.release(drain)
    assert other.claim_drain("hci1") is not None  # released: available again


def test_a_stale_drain_is_taken_over(tmp_path):
    m = _manager(tmp_path, "resetter")
    _foreign_file(tmp_path, "hci1.drain", pid=99999999, aged=3600)
    assert m.claim_drain("hci1") is not None


def test_drain_active_sees_any_live_drain(tmp_path):
    m = _manager(tmp_path, "svc")
    assert m.drain_active("hci1") is False
    _foreign_file(tmp_path, "hci1.drain", pid=1)  # foreign live
    assert m.drain_active("hci1") is True
    assert m.drain_active("hci2") is False


def test_choose_steers_away_from_a_draining_card(tmp_path):
    """hci1 is preferred and idle, but a live drain takes it out of the
    running entirely (0.5): a drain is a gate, not a ranking penalty."""
    m = _manager(tmp_path, "svc")
    _foreign_file(tmp_path, "hci1.drain", pid=1)
    adapter, claim = m.choose(["hci1", "hci2"])
    m.release(claim)
    assert adapter == "hci2"


def test_choose_refuses_when_every_candidate_is_draining(tmp_path):
    """R3, in the standalone-participant API: nothing NEW starts on a card
    someone is emptying, even when it is the only card there is. New work
    placed onto a draining adapter tops the drain up forever, so the reset
    either never fires or fires on a card that never emptied."""
    m = _manager(tmp_path, "svc")
    _foreign_file(tmp_path, "hci1.drain", pid=1)
    adapter, claim = m.choose(["hci1"])
    assert (adapter, claim) == (None, None)
    assert not [n for n in os.listdir(str(tmp_path)) if ".use." in n]


def test_own_use_counts_held_claims_but_not_the_drain(tmp_path):
    m = _manager(tmp_path, "svc")
    soft = m.claim_soft("hci1")
    slot = m.claim_slot("hci1", 2)
    drain = m.claim_drain("hci1")
    try:
        assert m.own_use("hci1") == 2  # soft + slot; the drain is not usage
        assert m.own_use("hci2") == 0
        m.release(slot)
        assert m.own_use("hci1") == 1
    finally:
        m.release_all()


def test_the_snapshot_reports_drain(tmp_path):
    m = _manager(tmp_path, "svc")
    _foreign_file(tmp_path, "hci1.drain", pid=1)
    state = m.claims()
    assert state["hci1"]["drain"] is True
    assert state["hci1"]["drain_pid"] == 1


def test_the_on_beat_hook_fires_after_the_sweep_and_may_raise(tmp_path):
    m = _manager(tmp_path, "svc")
    fired = []
    m.on_beat = lambda: fired.append(True)
    claim = m.claim_soft("hci1")
    try:
        m._beat_once()
        assert fired == [True]
        m.on_beat = lambda: 1 / 0  # a broken hook must not stop the beat
        m._beat_once()
        assert not claim.released
    finally:
        m.release(claim)


# -- adapter identity: MAC-keyed claims (convention 0.4) -------------------


def _fake_adapters(monkeypatch, mapping):
    """Present hciN -> MAC, as the kernel would report it."""
    monkeypatch.setattr(claims, "_mac_cache", {})
    monkeypatch.setattr(claims, "present_hci_names", lambda: sorted(mapping))
    monkeypatch.setattr(claims, "_read_adapter_mac", lambda a: mapping.get(a, claims.UNKNOWN_MAC))


def test_mac_parsing_accepts_every_spelling():
    """Config values are typed by humans: any separator, any case."""
    for text in (
        "AA:BB:CC:DD:EE:FF", "aa:bb:cc:dd:ee:ff", "AABBCCDDEEFF", "aabbccddeeff",
        "AA-BB-CC-DD-EE-FF", "aa.bb.cc.dd.ee.ff", "AA BB CC DD EE FF", " Aa:bB:Cc:dD:eE:fF ",
    ):
        assert claims.mac_key(text) == "AABBCCDDEEFF", text
    for text in ("hci0", "", "not-a-mac", "AABBCCDDEE", "GG:BB:CC:DD:EE:FF"):
        assert claims.mac_key(text) is None, text


def test_claims_are_keyed_by_the_adapters_mac(tmp_path, monkeypatch):
    """The card, not the number: a claim taken on hci3 is filed under the
    MAC, so a renumbering cannot make it name a different radio."""
    _fake_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    m = _manager(tmp_path, "svc")
    claim = m.claim_soft("hci3", qualifier="C8478C000001")
    try:
        assert os.listdir(str(tmp_path)) == ["AABBCCDDEEFF.use.svc.C8478C000001"]
    finally:
        m.release(claim)


def test_a_renumbered_card_keeps_its_claims(tmp_path, monkeypatch):
    """The bug this convention change exists for: a USB reset renumbers
    hci3 to hci7 with no reboot. The claim still names the same card, and
    a query by EITHER spelling still finds it."""
    _fake_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    m = _manager(tmp_path, "svc")
    claim = m.claim_slot("hci3", 2)
    try:
        assert os.listdir(str(tmp_path)) == ["AABBCCDDEEFF.link.0"]
        # the card comes back as hci7 after a reset
        _fake_adapters(monkeypatch, {"hci7": "AA:BB:CC:DD:EE:FF"})
        assert m.own_use("hci7") == 1
        assert m.own_use("AA:BB:CC:DD:EE:FF") == 1
        assert m.claims()["AABBCCDDEEFF"]["links"] == 1
    finally:
        m.release(claim)


def test_an_adapter_may_be_named_by_mac_anywhere(tmp_path, monkeypatch):
    _fake_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    m = _manager(tmp_path, "svc")
    hard = m.claim_hard("aa-bb-cc-dd-ee-ff")  # a third spelling of the card
    try:
        assert hard is not None
        assert sorted(os.listdir(str(tmp_path)))[0] == "AABBCCDDEEFF.scan"
        assert [n for n in os.listdir(str(tmp_path)) if ".holder." in n]
        assert m.claims()["AABBCCDDEEFF"]["hard"] is not None
    finally:
        m.release(hard)


def test_a_dead_card_degrades_to_its_hci_name(tmp_path, monkeypatch):
    """All-zeros MAC (a failed controller): coordination must not fail
    closed just because the card will not identify itself."""
    _fake_adapters(monkeypatch, {"hci0": claims.UNKNOWN_MAC})
    m = _manager(tmp_path, "svc")
    claim = m.claim_soft("hci0")
    try:
        assert os.listdir(str(tmp_path)) == ["hci0.use.svc"]
    finally:
        m.release(claim)


def test_a_pre_0_4_processes_claims_are_still_counted(tmp_path, monkeypatch):
    """Mixed-version fleet: a 0.3 process files hci3.link.0. A 0.4 process
    must count it as occupancy on that card, whichever way it asks."""
    _fake_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    m = _manager(tmp_path, "svc")
    _foreign_file(tmp_path, "hci3.link.0", pid=1)

    assert m.foreign_use("hci3") == 1
    assert m.foreign_use("AA:BB:CC:DD:EE:FF") == 1
    assert m.claims()["AABBCCDDEEFF"]["links"] == 1


def test_an_exclusive_claim_does_not_double_book_a_legacy_holder(tmp_path, monkeypatch):
    """The transition hazard: without a legacy check, a 0.4 process would
    take AABBCC..link.0 while a 0.3 process holds hci3.link.0 - both
    believing they own slot 0 of one card."""
    _fake_adapters(monkeypatch, {"hci3": "AA:BB:CC:DD:EE:FF"})
    m = _manager(tmp_path, "svc")
    _foreign_file(tmp_path, "hci3.link.0", pid=1)
    _foreign_file(tmp_path, "hci3.scan", pid=1)

    slot = m.claim_slot("hci3", 2)
    try:
        assert slot is not None
        assert os.path.basename(slot.path) == "AABBCCDDEEFF.link.1"  # slot 0 respected
    finally:
        m.release(slot)
    assert m.claim_hard("hci3") is None  # legacy scanner still holds the card
    assert m.drain_active("hci3") is False
    _foreign_file(tmp_path, "hci3.drain", pid=1)
    assert m.drain_active("hci3") is True  # legacy drain seen too


def test_two_cards_sharing_one_address_are_not_merged(tmp_path, monkeypatch, caplog):
    """Counterfeit CSR dongles ship batches with one hardcoded address, and
    this deployment runs CSR-based cards. Keying claims by a shared address
    would merge two physical radios into one accounting identity - slots
    double-booked, occupancy halved, a drain on one draining the other."""
    import logging as _logging

    monkeypatch.setattr(claims, "_mac_cache", {})
    monkeypatch.setattr(claims, "_duplicate_macs", set())
    monkeypatch.setattr(claims, "_warned_duplicate_macs", set())
    monkeypatch.setattr(
        claims,
        "_read_hciconfig_table",
        lambda: {"hci1": "00:1A:7D:DA:71:05", "hci2": "00:1A:7D:DA:71:05"},
    )
    monkeypatch.setattr(claims, "_read_sysfs_mac", lambda a: None)

    with caplog.at_level(_logging.WARNING):
        first = claims.adapter_key("hci1")
        second = claims.adapter_key("hci2")

    assert first != second  # never merged
    assert (first, second) == ("hci1", "hci2")  # the number is at least unique
    assert any("more than one adapter reports" in r.message for r in caplog.records)

    m = _manager(tmp_path, "svc")
    a = m.claim_slot("hci1", 1)
    b = m.claim_slot("hci2", 1)  # a distinct card: its own slot 0, not a refusal
    try:
        assert a is not None and b is not None
        assert sorted(os.listdir(str(tmp_path))) == ["hci1.link.0", "hci2.link.0"]
    finally:
        m.release_all()


def test_distinct_addresses_are_unaffected_by_the_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_mac_cache", {})
    monkeypatch.setattr(claims, "_duplicate_macs", set())
    monkeypatch.setattr(
        claims,
        "_read_hciconfig_table",
        lambda: {"hci1": "00:1A:7D:DA:71:05", "hci2": "00:1A:7D:DA:71:06"},
    )
    monkeypatch.setattr(claims, "_read_sysfs_mac", lambda a: None)

    assert claims.adapter_key("hci1") == "001A7DDA7105"
    assert claims.adapter_key("hci2") == "001A7DDA7106"


def test_hci_for_is_fresh_by_default(tmp_path, monkeypatch):
    """A direct caller resolving a MAC to open a raw HCI socket on the
    result must not be handed a cached number: within one TTL that would
    program a scan onto whatever card had inherited it. Reported by the
    sensors-py session, whose own docstring already required this and whose
    backend did not supply it."""
    live = {"hci3": "AA:BB:CC:DD:EE:FF", "hci4": "11:22:33:44:55:66"}
    monkeypatch.setattr(claims, "_mac_cache", {})
    monkeypatch.setattr(claims, "present_hci_names", lambda: sorted(live))
    monkeypatch.setattr(claims, "_read_adapter_mac", lambda a: live.get(a, claims.UNKNOWN_MAC))

    assert claims.hci_for("AA:BB:CC:DD:EE:FF") == "hci3"   # populates the cache
    live["hci3"], live["hci4"] = "11:22:33:44:55:66", "AA:BB:CC:DD:EE:FF"

    # no TTL expiry, no manual invalidation - the card renumbered and the
    # very next resolution must follow it
    assert claims.hci_for("AA:BB:CC:DD:EE:FF") == "hci4"


def test_hci_for_honours_an_explicit_opt_out(tmp_path, monkeypatch):
    """fresh=False is for callers that already refreshed, or that can
    tolerate staleness - it must actually use the cache, or the catcher
    would pay one hciconfig call per configured adapter."""
    live = {"hci3": "AA:BB:CC:DD:EE:FF"}
    reads = []
    monkeypatch.setattr(claims, "_mac_cache", {})
    monkeypatch.setattr(claims, "present_hci_names", lambda: sorted(live))

    def counted(a):
        reads.append(a)
        return live.get(a, claims.UNKNOWN_MAC)

    monkeypatch.setattr(claims, "_read_adapter_mac", counted)

    assert claims.hci_for("AA:BB:CC:DD:EE:FF") == "hci3"
    before = len(reads)
    assert claims.hci_for("AA:BB:CC:DD:EE:FF", fresh=False) == "hci3"
    assert len(reads) == before          # served from cache, no new read


def test_an_hci_name_costs_nothing_to_resolve(monkeypatch):
    """There is nothing to resolve for a number, so it must not refresh."""
    monkeypatch.setattr(claims, "_mac_cache", {})
    monkeypatch.setattr(claims, "_read_adapter_mac",
                        lambda a: (_ for _ in ()).throw(AssertionError("should not read")))
    assert claims.hci_for("hci7") == "hci7"


# -- 0.5: hardlinked exclusive locks ---------------------------------------
#
# The two defects these close were both invisible to the 0.4 suite because
# every test held its claims in one process and never raced them: a stale
# takeover that could unlink a claim created since its diagnosis, and a
# heartbeat that refreshed a well-known NAME without ever asking whether that
# name still referred to its file. Each test below interleaves the race by
# hand, because production supplies the interleave and a single-threaded
# suite never will.


def _live_pair(manager, adapter, suffix="scan"):
    """The (lock, holder) paths of a claim this manager holds."""
    claim = manager._acquire_exclusive(adapter, suffix)
    return claim, claim.path, claim.holder


def _stale_pair(tmp_path, key, suffix="scan", pid=99999999, service="dead"):
    """A stale 0.5 claim on disk: a lock hardlinked to its holder, both with
    a dead pid and an mtime past the TTL."""
    holder = os.path.join(str(tmp_path), f"{key}.{suffix}.holder.{service}-{pid}-0")
    lock = os.path.join(str(tmp_path), f"{key}.{suffix}")
    with open(holder, "w") as f:
        f.write(f"{pid} {service} {int(time.time())} {suffix}\n")
    os.link(holder, lock)
    _age(holder, 3600)  # shared inode: ages the lock with it
    return lock, holder


def test_a_hard_claim_is_a_hardlink_to_its_own_holder_file(tmp_path):
    """The shared inode IS the claim. Without it there is nothing a holder
    can check to find out whether the name still refers to its file."""
    m = _manager(tmp_path, "svc")
    claim, lock, holder = _live_pair(m, "hci1")
    try:
        assert claim is not None and holder is not None
        assert os.stat(lock).st_ino == os.stat(holder).st_ino
        assert os.stat(lock).st_nlink == 2
        assert claim.owns() is True
    finally:
        m.release(claim)
    assert not os.path.exists(lock)
    assert not os.path.exists(holder)


def test_the_loser_of_the_link_race_creates_nothing(tmp_path):
    """EEXIST on os.link is a lost race and nothing else: the loser leaves no
    half-made claim behind for a sweep to trip over."""
    a = _manager(tmp_path, "svc-a")
    b = _manager(tmp_path, "svc-b")
    held = a.claim_hard("hci1")
    try:
        assert b.claim_hard("hci1") is None
        holders = [n for n in os.listdir(str(tmp_path)) if ".holder." in n]
        assert len(holders) == 1, holders
        assert "svc-b" not in holders[0]
    finally:
        a.release(held)


def test_the_beat_notices_a_stolen_lock_and_gives_it_up(tmp_path, caplog):
    """The 0.4 heartbeat called utime on the well-known name without asking
    whether it was still ours, so a double-hold never self-corrected - both
    processes kept refreshing what each believed was its claim. The inode
    check is the question that was never being asked."""
    import logging

    a = _manager(tmp_path, "svc-a")
    b = _manager(tmp_path, "svc-b")
    claim = a.claim_hard("hci1")
    os.unlink(claim.path)                    # somebody removes the lock
    stolen = b.claim_hard("hci1")            # and takes the name
    assert stolen is not None
    with caplog.at_level(logging.WARNING):
        a._beat_once()
    assert claim.lost is True
    assert a.own_use("hci1") == 0
    assert "lost exclusive claim" in caplog.text
    assert stolen.owns() is True             # the winner is undisturbed
    b.release(stolen)


def test_a_lost_claim_release_does_not_unlink_the_winners_lock(tmp_path):
    """Releasing a claim we have provably lost must not reach through the
    name at somebody else's file - the failure mode that turns one lost claim
    into two processes taking turns evicting each other."""
    a = _manager(tmp_path, "svc-a")
    b = _manager(tmp_path, "svc-b")
    claim = a.claim_hard("hci1")
    os.unlink(claim.path)
    stolen = b.claim_hard("hci1")
    a._beat_once()
    a.release(claim)
    assert os.path.exists(stolen.path)
    assert stolen.owns() is True
    b.release(stolen)


def test_a_stale_lock_is_taken_over_by_rename_never_by_unlink(tmp_path, monkeypatch):
    """Rename is atomic and picks one winner; unlink lets every reaper
    'succeed', and the second one destroys what the first created."""
    m = _manager(tmp_path, "svc")
    lock, stale_holder = _stale_pair(tmp_path, "hci1")
    renamed = []
    unlinked = []
    real_rename, real_unlink = os.rename, os.unlink

    def watched_rename(src, dst):
        renamed.append((str(src), str(dst)))
        return real_rename(src, dst)

    def watched_unlink(path):
        unlinked.append(str(path))
        return real_unlink(path)

    monkeypatch.setattr(claims.os, "rename", watched_rename)
    monkeypatch.setattr(claims.os, "unlink", watched_unlink)
    claim = m.claim_hard("hci1")
    monkeypatch.undo()
    try:
        assert claim is not None
        assert [dst for src, dst in renamed if src == lock], "the lock was not renamed aside"
        assert lock not in unlinked, "the live lock name was unlinked"
        assert not os.path.exists(stale_holder), "the dead holder was left behind"
        assert not [n for n in os.listdir(str(tmp_path)) if ".reaping." in n]
        assert claim.owns() is True
    finally:
        m.release(claim)


def test_a_takeover_cannot_destroy_a_claim_created_since_its_diagnosis(tmp_path, monkeypatch):
    """The TOCTOU the rename-aside exists for. A diagnoses the lock stale;
    before it acts, B reaps it and creates a LIVE claim. Under 0.4's
    check-unlink-create, A's unlink deleted B's fresh claim and B never found
    out. Here A moves B's file aside, sees an inode it did not diagnose, puts
    it back and concedes."""
    a = _manager(tmp_path, "svc-a")
    b = _manager(tmp_path, "svc-b")
    lock, _holder = _stale_pair(tmp_path, "hci1")
    state = {"raced": False, "claim": None}
    real_rename = os.rename

    def racing_rename(src, dst):
        if not state["raced"] and str(src) == lock:
            state["raced"] = True
            state["claim"] = b.claim_hard("hci1")   # B wins the window
        return real_rename(src, dst)

    monkeypatch.setattr(claims.os, "rename", racing_rename)
    lost = a.claim_hard("hci1")
    monkeypatch.undo()
    try:
        assert state["raced"] is True
        assert lost is None, "A took a card B already holds"
        assert state["claim"] is not None
        assert os.path.exists(lock), "B's fresh claim was destroyed"
        assert state["claim"].owns() is True
        assert not [n for n in os.listdir(str(tmp_path)) if ".reaping." in n]
    finally:
        b.release(state["claim"])


def test_the_sweep_reaps_orphan_holders_and_abandoned_reaping_files(tmp_path):
    """A crashed process leaves both kinds of litter: a holder whose lock is
    already gone, and a file it had renamed aside when it died. Neither
    coordinates anything, and both are keyed on the pid in the FILENAME for
    the reaping file - its CONTENT belongs to the dead claim it moved."""
    m = _manager(tmp_path, "svc")
    orphan = _foreign_file(tmp_path, "hci1.scan.holder.dead-99999999-1", pid=99999999, aged=3600)
    abandoned = _foreign_file(tmp_path, "hci1.scan.reaping.dead-99999999-2", pid=99999999)
    live = _foreign_file(tmp_path, "hci2.scan.holder.other-1-3", pid=1)
    m.claims()
    assert not os.path.exists(orphan)
    assert not os.path.exists(abandoned)
    assert os.path.exists(live), "reaped a live process's holder file"


def test_a_plain_0_4_hard_claim_still_gates_a_0_5_claimant(tmp_path):
    """Mixed-version minutes during a deploy: a 0.4 process writes a bare
    O_EXCL file with no holder sibling. It is a hard claim and must be
    honoured as one, judged the 0.4 way on content pid and mtime."""
    m = _manager(tmp_path, "svc")
    path = _foreign_file(tmp_path, "hci1.scan", pid=1)
    assert os.stat(path).st_nlink == 1
    assert m.claim_hard("hci1") is None
    _age(path, 3600)                       # same file, now past the TTL
    taken = m.claim_hard("hci1")
    assert taken is not None
    m.release(taken)


def _parse_like_0_4(claim_dir, is_live):
    """The convention 0.4 claims() loop, verbatim, so a 0.5 directory can be
    fed to the parser a not-yet-updated process is still running."""
    state = {}
    for name in sorted(os.listdir(claim_dir)):
        prefix, sep, rest = name.partition(".")
        if not sep:
            continue
        path = os.path.join(claim_dir, name)
        entry = state.setdefault(prefix, {"hard": None, "soft": 0, "links": 0, "drain": False})
        if rest == "scan":
            if is_live(path):
                entry["hard"] = path
        elif rest == "drain":
            if is_live(path):
                entry["drain"] = True
        elif rest.startswith("use."):
            if is_live(path):
                entry["soft"] += 1
        elif rest.startswith("link."):
            if is_live(path):
                entry["links"] += 1
    return state


def test_a_0_4_parser_reads_a_0_5_directory_without_miscounting(tmp_path):
    """The compatibility direction that cannot be fixed after the fact: an
    old process reading new files. 0.4 matches the remainder after the first
    dot exactly ("scan", "drain") or by prefix ("use.", "link."), so every
    0.5 name has to miss all four - a holder counted as a claim would double
    every exclusive name, and a scanwait ticket counted as one would invent
    a scanner that does not exist."""
    m = _manager(tmp_path, "svc-a")
    hard = m.claim_hard("hci1")
    drain = m.claim_drain("hci2")
    wait = m.claim_scanwait("hci1")
    m.claim_soft("hci1", qualifier="C8478C000001")
    m.claim_slot("hci1", 2)
    try:
        seen = _parse_like_0_4(str(tmp_path), m._is_live)
        assert set(seen) == {"hci1", "hci2"}
        assert seen["hci1"]["hard"] == hard.path
        assert seen["hci1"]["soft"] == 1, "a 0.5 file was miscounted as a soft claim"
        assert seen["hci1"]["links"] == 1, "a 0.5 file was miscounted as a link slot"
        assert seen["hci1"]["drain"] is False
        assert seen["hci2"]["drain"] is True
        assert seen["hci2"]["hard"] is None
    finally:
        m.release(wait)
        m.release(drain)
        m.release_all()


# -- 0.5: the scanwait queue -----------------------------------------------


def test_a_scanwait_ticket_is_visible_as_a_queue_and_claims_nothing(tmp_path):
    m = _manager(tmp_path, "svc-a")
    other = _manager(tmp_path, "svc-b")
    first = m.claim_scanwait("hci1")
    second = other.claim_scanwait("hci1")
    try:
        waiters = m.claims()["hci1"]["waiters"]
        assert len(waiters) == 2
        assert [w[0] for w in waiters] == sorted(w[0] for w in waiters)
        # a ticket is not use: it must never rank a card as busier, and it
        # must never veto a reset
        assert m.claims()["hci1"]["soft"] == 0
        assert m.claims()["hci1"]["links"] == 0
        assert m.own_use("hci1") == 0
        assert m.foreign_use("hci1") == 0
    finally:
        m.release(first)
        other.release(second)
    assert m.claims().get("hci1", {}).get("waiters", []) == []


def test_queue_order_is_the_filename_sequence_not_the_mtime(tmp_path):
    """A Venus box has no RTC and steps its clock at the first NTP sync, so
    an mtime-ordered queue reorders itself under its readers. The stamp in
    the filename cannot move."""
    m = _manager(tmp_path, "svc-a")
    first = m.claim_scanwait("hci1")
    second = m.claim_scanwait("hci1")
    try:
        assert first.seq < second.seq
        _age(first.path, 5)                  # the older ticket looks NEWER
        os.utime(second.path, (time.time() - 20, time.time() - 20))
        waiters = m.claims()["hci1"]["waiters"]
        assert [w[0] for w in waiters] == [first.seq, second.seq]
    finally:
        m.release_all()


def test_a_stale_scanwait_ticket_is_reaped(tmp_path):
    m = _manager(tmp_path, "svc")
    _foreign_file(tmp_path, "hci1.scanwait.dead-99999999-4", pid=99999999, aged=3600)
    assert m.claims().get("hci1", {}).get("waiters", []) == []
    assert not os.path.exists(os.path.join(str(tmp_path), "hci1.scanwait.dead-99999999-4"))


def test_the_release_hook_fires_for_every_release(tmp_path):
    """The in-process wake for queued waiters. An optimization only - every
    waiter still polls - but it is what makes a handover take milliseconds
    instead of a second."""
    m = _manager(tmp_path, "svc")
    seen = []
    m.on_release = lambda claim: seen.append(os.path.basename(claim.path))
    hard = m.claim_hard("hci1")
    m.release(hard)
    assert seen == ["hci1.scan"]
    m.on_release = lambda claim: (_ for _ in ()).throw(RuntimeError("boom"))
    m.release(m.claim_hard("hci1"))          # a broken hook must not escape
