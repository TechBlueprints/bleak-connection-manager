# -*- coding: utf-8 -*-
"""Tests for claims-gated adapter recovery."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bleak_connection_manager import recovery  # noqa: E402
from bleak_connection_manager import claims as claims_mod  # noqa: E402
from bleak_connection_manager.claims import ClaimManager  # noqa: E402


def _foreign_file(tmp_path, name, pid=1):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as f:
        f.write(f"{pid} foreign-svc {int(time.time())}\n")
    return path


def _fake_recovery(monkeypatch):
    calls = []

    async def fake_recover_adapter(idx, mac, gone_silent):
        calls.append((idx, mac, gone_silent))
        return True

    monkeypatch.setattr(recovery, "HAS_AUTO_RECOVERY", True)
    monkeypatch.setattr(recovery, "recover_adapter", fake_recover_adapter)
    return calls


def test_reset_refuses_while_other_processes_hold_claims(tmp_path, monkeypatch):
    """A reset kills every link on the card; foreign live claims veto it.
    drain_timeout=0 is the immediate gate - no drain, no waiting."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    _foreign_file(tmp_path, "hci1.link.0")

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=0)) is False
    assert calls == []


def test_reset_proceeds_only_from_a_completely_empty_card(tmp_path, monkeypatch):
    """Zero foreign AND zero own. A stale foreign file is not use - it names
    no live process - so it does not veto either."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    stale = _foreign_file(tmp_path, "hci1.link.0", pid=99999999)
    old = time.time() - 3600
    os.utime(stale, (old, old))

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, gone_silent=True, drain_timeout=0)) is True
    assert calls == [(1, recovery.UNKNOWN_MAC, True)]


def test_our_own_link_vetoes_the_reset_exactly_like_a_foreign_one(tmp_path, monkeypatch):
    """R2, and the reversal of the rule this replaces. "Our own links are
    ours to kill" reads reasonably right up until you notice that a straggler
    is by definition a claim that COULD NOT MOVE - a device's only working
    card, an operator pin - and that cycling the card makes every device on
    it disappear in every process at once, which on BlueZ 5.72 is the
    gatt-client use-after-free going off. Blowing up the one link a device
    still had, to fix a card that link proves is working, was never a trade
    worth making."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    own = manager.claim_slot("hci1", 1)
    assert own is not None
    try:
        assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=0)) is False
        assert calls == []
        assert not os.path.exists(os.path.join(str(tmp_path), "hci1.drain"))
    finally:
        manager.release(own)


def test_our_own_soft_claim_with_no_link_does_not_veto_the_reset(tmp_path, monkeypatch):
    """Clint, 2026-09-06. A soft claim is taken BEFORE the connect and the
    link slot only after it succeeds, so our own .use with no .link behind
    it is our own attempt in flight - and on a card that has earned a cycle
    that attempt fails anyway. Vetoing on it only delayed the cycle by one
    attempt (prod hci7, 2026-09-04: easytouch's own .use sat on the card it
    could not scan). A foreign soft claim still vetoes: another process's
    attempt is not ours to fail."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    own = manager.claim_soft("hci1", "AA:BB")
    try:
        assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, gone_silent=True, drain_timeout=0)) is True
        assert calls == [(1, recovery.UNKNOWN_MAC, True)]
    finally:
        manager.release(own)

    calls.clear()
    _foreign_file(tmp_path, "hci1.use.other-svc-7.AA:BB")
    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=0)) is False
    assert calls == []


def test_force_still_respects_the_in_use_veto(tmp_path, monkeypatch):
    """force bypasses the EVIDENCE gates (which live in the catcher), never
    the empty-card veto. An override that could cycle a card carrying live
    links would make the path an operator reaches for in a crisis the one
    path that can detonate the fleet."""
    calls = _fake_recovery(monkeypatch)
    monkeypatch.setattr(recovery, "_DRAIN_POLL", 0.03)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    _foreign_file(tmp_path, "hci1.scan")

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, force=True, drain_timeout=0.15)) is False
    assert calls == []


def test_force_takes_the_drain_claim_like_any_other_reset(tmp_path, monkeypatch):
    """It still coordinates: the drain is what keeps other processes from
    placing new work onto the card mid-reset, and it is released after."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    seen = []
    manager.on_release = lambda claim: seen.append(os.path.basename(claim.path))

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, force=True, drain_timeout=0)) is True
    assert len(calls) == 1
    assert "hci1.drain" in seen
    assert not os.path.exists(os.path.join(str(tmp_path), "hci1.drain"))


def test_an_uncoordinated_reset_says_so(tmp_path, monkeypatch, caplog):
    """With no manager this reset cannot see, drain or honour another
    process's links. It still runs - some callers genuinely have no
    coordination available - but the mass device removal it causes elsewhere
    must not be unattributable."""
    import logging

    _fake_recovery(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(recovery.reset_adapter("hci1")) is True
    assert "UNCOORDINATED" in caplog.text


def test_reset_rejects_non_hci_adapter_names(monkeypatch):
    calls = _fake_recovery(monkeypatch)
    assert asyncio.run(recovery.reset_adapter("bogus0")) is False
    assert calls == []


def test_reset_falls_back_to_the_native_sequence(tmp_path, monkeypatch):
    """Without bluetooth-auto-recovery the stdlib-native reset runs instead
    of degrading to a no-op."""
    native_calls = []

    async def fake_native(dev_id, adapter, gone_silent):
        native_calls.append((dev_id, adapter, gone_silent))
        return True

    monkeypatch.setattr(recovery, "HAS_AUTO_RECOVERY", False)
    monkeypatch.setattr(recovery, "_native_recover", fake_native)
    assert asyncio.run(recovery.reset_adapter("hci1", gone_silent=True)) is True
    assert native_calls == [(1, "hci1", True)]


class _FakeCompleted:
    def __init__(self, stdout):
        self.stdout = stdout


def test_adapter_mac_falls_back_to_hciconfig(monkeypatch):
    """Some kernels (Venus OS Cerbos) expose no sysfs address attribute at
    all; hciconfig still reports BD Address - including the all-zeros one
    that marks a dead card."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        # bare hciconfig prints every interface at once
        return _FakeCompleted(
            "hci1:\tType: Primary  Bus: USB\n\tBD Address: 00:11:22:33:44:55  ACL MTU: 310:10\n"
            "\n"
            "hci0:\tType: Primary  Bus: UART\n\tBD Address: 00:00:00:00:00:00  ACL MTU: 0:0\n"
        )

    monkeypatch.setattr(claims_mod, "_mac_cache", {})
    monkeypatch.setattr(claims_mod.subprocess, "run", fake_run)
    assert recovery.adapter_mac("hci1") == "00:11:22:33:44:55"
    assert recovery.adapter_mac("hci0") == recovery.UNKNOWN_MAC  # dead card detected


def test_adapter_mac_is_cached_between_selections(monkeypatch):
    """Selection asks per candidate per connect; without the cache a kernel
    with no sysfs address would spawn hciconfig in a tight loop."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _FakeCompleted(
            "hci1:\n\tBD Address: 00:11:22:33:44:55\n"
            "\n"
            "hci2:\n\tBD Address: 00:11:22:33:44:66\n"
        )

    monkeypatch.setattr(claims_mod, "_mac_cache", {})
    monkeypatch.setattr(claims_mod.subprocess, "run", fake_run)
    assert recovery.adapter_mac("hci1") == "00:11:22:33:44:55"
    assert recovery.adapter_mac("hci1") == "00:11:22:33:44:55"
    # the same single read already answered for the other card: seven
    # adapters on prod must not mean seven spawns
    assert recovery.adapter_mac("hci2") == "00:11:22:33:44:66"
    assert len(calls) == 1


# -- drain-coordinated reset (convention 0.3) ------------------------------


def _touching_foreign(tmp_path, name, stop):
    """A foreign holder that heartbeats: touches its claim file until told
    to stop, from a thread, the way a live process would."""
    import threading

    path = _foreign_file(tmp_path, name)

    def beat():
        while not stop.is_set():
            os.utime(path, None)
            stop.wait(0.02)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    return path, t


def test_drain_waits_out_a_holder_that_never_leaves_then_refuses(tmp_path, monkeypatch):
    """A foreign holder that cannot migrate keeps its card: the drain claim
    is taken, the wait runs to the deadline, the reset is refused, and the
    drain file is released so placement stops steering away."""
    import threading

    calls = _fake_recovery(monkeypatch)
    monkeypatch.setattr(recovery, "_DRAIN_POLL", 0.03)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    stop = threading.Event()
    _touching_foreign(tmp_path, "hci1.link.0", stop)
    try:
        assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=0.2)) is False
    finally:
        stop.set()
    assert calls == []
    assert not os.path.exists(os.path.join(str(tmp_path), "hci1.drain"))


def test_drain_resets_once_the_foreign_holder_migrates(tmp_path, monkeypatch):
    """The cooperative path end to end: the holder sees the drain and moves
    (its claim file disappears); the resetter proceeds within its deadline
    and cleans up its drain claim."""
    calls = _fake_recovery(monkeypatch)
    monkeypatch.setattr(recovery, "_DRAIN_POLL", 0.03)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    path = _foreign_file(tmp_path, "hci1.link.0")

    async def scenario():
        async def migrate():
            await asyncio.sleep(0.08)
            os.unlink(path)

        results = await asyncio.gather(
            recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=2.0),
            migrate(),
        )
        return results[0]

    assert asyncio.run(scenario()) is True
    assert len(calls) == 1
    assert not os.path.exists(os.path.join(str(tmp_path), "hci1.drain"))


def test_a_second_resetter_backs_off_a_foreign_drain(tmp_path, monkeypatch):
    """One recovery at a time: a live foreign drain claim means the card is
    already being handled."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    _foreign_file(tmp_path, "hci1.drain")

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=0.2)) is False
    assert calls == []


def test_drain_refuses_at_the_deadline_for_our_own_unmigratable_claims(tmp_path, monkeypatch):
    """Own claims are waited on AND vetoed. The wait gives them a chance to
    migrate; the veto is what happens when they could not, and it is the
    same answer a foreign holder gets. Existing work always beats
    maintenance, and the drain file is released so placement stops steering
    away from a card nobody is going to reset."""
    calls = _fake_recovery(monkeypatch)
    monkeypatch.setattr(recovery, "_DRAIN_POLL", 0.03)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    assert manager.claim_slot("hci1", 1) is not None  # our own LINK: work that cannot move
    try:
        assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=0.15)) is False
    finally:
        manager.release_all()
    assert calls == []
    assert not os.path.exists(os.path.join(str(tmp_path), "hci1.drain"))


def test_a_drain_that_our_own_work_leaves_in_time_still_resets(tmp_path, monkeypatch):
    """The veto is a veto on work that CANNOT move, not on our own work as
    such: a claim released inside the window lets the reset through."""
    calls = _fake_recovery(monkeypatch)
    monkeypatch.setattr(recovery, "_DRAIN_POLL", 0.03)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    own = manager.claim_soft("hci1")

    async def scenario():
        async def migrate():
            await asyncio.sleep(0.08)
            manager.release(own)

        results = await asyncio.gather(
            recovery.reset_adapter("hci1", claims_manager=manager, drain_timeout=2.0),
            migrate(),
        )
        return results[0]

    assert asyncio.run(scenario()) is True
    assert len(calls) == 1


# -- native reset primitives -----------------------------------------------


def test_rfkill_unblock_clears_bluetooth_soft_blocks(tmp_path, monkeypatch):
    rf = tmp_path / "rfkill0"
    rf.mkdir()
    (rf / "type").write_text("bluetooth\n")
    (rf / "soft").write_text("1\n")
    wifi = tmp_path / "rfkill1"
    wifi.mkdir()
    (wifi / "type").write_text("wlan\n")
    (wifi / "soft").write_text("1\n")
    monkeypatch.setattr(recovery, "RFKILL_SYSFS", str(tmp_path))

    assert recovery._rfkill_unblock("hci0") is True
    assert (rf / "soft").read_text() == "0"
    assert (wifi / "soft").read_text() == "1\n"  # not ours to touch


def test_usb_reset_finds_the_device_node_and_ioctls_it(tmp_path, monkeypatch):
    bt = tmp_path / "bt" / "hci1"
    bt.mkdir(parents=True)
    usb_if = tmp_path / "usb" / "1-1.2:1.0"
    usb_if.mkdir(parents=True)
    (bt / "device").symlink_to(usb_if)
    parent = tmp_path / "usb" / "1-1.2"
    parent.mkdir()
    (parent / "busnum").write_text("1\n")
    (parent / "devnum").write_text("7\n")
    devfs = tmp_path / "devfs" / "001"
    devfs.mkdir(parents=True)
    (devfs / "007").write_text("")
    monkeypatch.setattr(recovery, "BT_SYSFS", str(tmp_path / "bt"))
    monkeypatch.setattr(recovery, "USB_SYSFS", str(tmp_path / "usb"))
    monkeypatch.setattr(recovery, "USB_DEVFS", str(tmp_path / "devfs"))
    ioctls = []
    import fcntl

    monkeypatch.setattr(fcntl, "ioctl", lambda fd, op, arg=0: ioctls.append(op))

    assert recovery._usb_reset("hci1") is True
    assert ioctls == [recovery.USBDEVFS_RESET]


def test_usb_reset_reports_a_uart_controller_as_not_usb(tmp_path, monkeypatch):
    bt = tmp_path / "bt" / "hci0"
    bt.mkdir(parents=True)
    serial = tmp_path / "serial0-0"
    serial.mkdir()
    (bt / "device").symlink_to(serial)
    monkeypatch.setattr(recovery, "BT_SYSFS", str(tmp_path / "bt"))

    assert recovery._usb_reset("hci0") is None  # None, not False: no failure


def test_native_recover_succeeds_on_a_bounced_uart_card(tmp_path, monkeypatch):
    """The gone_silent asymmetry fixed: a non-USB card whose bounce worked
    and whose MAC reads back is a SUCCESSFUL reset, where the library path
    reported False after doing the same work."""
    monkeypatch.setattr(recovery, "_rfkill_unblock", lambda adapter: False)
    monkeypatch.setattr(recovery, "_bounce_interface", lambda dev_id: True)
    monkeypatch.setattr(recovery, "_usb_reset", lambda adapter: None)
    monkeypatch.setattr(claims_mod, "_read_adapter_mac", lambda adapter: "AA:BB:CC:DD:EE:FF")
    claims_mod.invalidate_adapter_mac()

    assert asyncio.run(recovery._native_recover(0, "hci0", True)) is True


def test_native_recover_fails_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(recovery, "_rfkill_unblock", lambda adapter: False)
    monkeypatch.setattr(recovery, "_bounce_interface", lambda dev_id: False)
    monkeypatch.setattr(recovery, "_usb_reset", lambda adapter: None)
    claims_mod.invalidate_adapter_mac()

    assert asyncio.run(recovery._native_recover(0, "hci0", True)) is False


def test_is_bluetoothd_alive_scans_proc_comm(tmp_path, monkeypatch):
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "comm").write_text("bluetoothd\n")
    monkeypatch.setattr(recovery, "PROC", str(tmp_path))
    assert recovery.is_bluetoothd_alive() is True
    (proc / "comm").write_text("python3\n")
    assert recovery.is_bluetoothd_alive() is False


def _no_bt_socket(monkeypatch):
    def raise_os():
        raise OSError(97, "socket(AF_BLUETOOTH) failed")

    monkeypatch.setattr(recovery.mgmt, "open_bt_socket", raise_os)


def test_bounce_falls_back_to_hciconfig_without_a_bt_socket(monkeypatch):
    """When neither the socket module nor the ctypes syscall can produce a
    bluetooth socket, the ioctl bounce degrades to the runbook's hciconfig
    down/up."""
    _no_bt_socket(monkeypatch)
    runs = []

    def fake_run(argv, **kwargs):
        runs.append(argv)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)

    assert recovery._bounce_interface(1) is True
    assert runs == [["hciconfig", "hci1", "down"], ["hciconfig", "hci1", "up"]]


def test_hciconfig_bounce_reports_a_failed_up(monkeypatch):
    _no_bt_socket(monkeypatch)

    def fake_run(argv, **kwargs):
        class R:
            returncode = 1 if argv[-1] == "up" else 0
            stdout = ""
            stderr = "Operation not possible due to RF-kill"

        return R()

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)

    assert recovery._bounce_interface(0) is False



def test_a_card_that_scans_after_draining_is_not_reset(tmp_path, monkeypatch):
    """Probation (Clint, 2026-09-02): emptying the card may itself release
    what was wedged, so it gets one real use before the hardware is
    touched. It passes: no reset, drain released, reported as recovered."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    probed = []

    async def probe(adapter):
        probed.append(adapter)
        return True

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, gone_silent=True, drain_timeout=0, probe=probe)) is True
    assert probed == ["hci1"]
    assert calls == []                                                  # hardware untouched
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".drain")]  # drain released


def test_a_card_that_still_cannot_scan_after_draining_is_reset(tmp_path, monkeypatch):
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))

    async def probe(adapter):
        return False

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, gone_silent=True, drain_timeout=0, probe=probe)) is True
    assert calls == [(1, recovery.UNKNOWN_MAC, True)]


def test_a_probe_that_raises_counts_as_failed(tmp_path, monkeypatch):
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))

    async def probe(adapter):
        raise RuntimeError("scan machinery broke")

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, gone_silent=True, drain_timeout=0, probe=probe)) is True
    assert calls == [(1, recovery.UNKNOWN_MAC, True)]


def test_a_card_that_filled_up_during_the_probe_is_not_reset(tmp_path, monkeypatch):
    """The in-use veto applies at the moment of reset, not just at the
    drain deadline: a link that landed while we were probing keeps its
    card."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))

    async def probe(adapter):
        _foreign_file(tmp_path, "hci1.link.0")
        return False

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, gone_silent=True, drain_timeout=0, probe=probe)) is False
    assert calls == []
