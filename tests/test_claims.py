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
    assert len(os.listdir(str(tmp_path))) == 3
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
    """hci1 is preferred and idle, but a live drain outranks idleness; the
    fallback still uses a draining card when it is the only candidate."""
    m = _manager(tmp_path, "svc")
    _foreign_file(tmp_path, "hci1.drain", pid=1)
    adapter, claim = m.choose(["hci1", "hci2"])
    m.release(claim)
    assert adapter == "hci2"
    adapter, claim = m.choose(["hci1"])  # never gates
    m.release(claim)
    assert adapter == "hci1"


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
        assert os.listdir(str(tmp_path)) == ["AABBCCDDEEFF.scan"]
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
