# -*- coding: utf-8 -*-
"""Tests for claims-gated adapter recovery."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bleak_connection_manager import recovery  # noqa: E402
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
    """A reset kills every link on the card; foreign live claims veto it."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    _foreign_file(tmp_path, "hci1.link.0")

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager)) is False
    assert calls == []


def test_reset_proceeds_when_the_card_is_unused_by_others(tmp_path, monkeypatch):
    """Our own claims do not veto - we are the one asking - and a stale
    foreign file does not either."""
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    own = manager.claim_soft("hci1")
    stale = _foreign_file(tmp_path, "hci1.link.0", pid=99999999)
    old = time.time() - 3600
    os.utime(stale, (old, old))

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, gone_silent=True)) is True
    assert calls == [(1, recovery.UNKNOWN_MAC, True)]
    manager.release(own)


def test_force_overrides_the_foreign_claims_gate(tmp_path, monkeypatch):
    calls = _fake_recovery(monkeypatch)
    manager = ClaimManager(owner="svc", claim_dir=str(tmp_path))
    _foreign_file(tmp_path, "hci1.scan")

    assert asyncio.run(recovery.reset_adapter("hci1", claims_manager=manager, force=True)) is True
    assert len(calls) == 1


def test_reset_rejects_non_hci_adapter_names(monkeypatch):
    calls = _fake_recovery(monkeypatch)
    assert asyncio.run(recovery.reset_adapter("bogus0")) is False
    assert calls == []


def test_reset_degrades_without_bluetooth_auto_recovery(tmp_path):
    if recovery.HAS_AUTO_RECOVERY:  # environment-dependent guard
        return
    assert asyncio.run(recovery.reset_adapter("hci1")) is False


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
        name = argv[1]
        out = "hci1:\tType: Primary  Bus: USB\n\tBD Address: 00:11:22:33:44:55  ACL MTU: 310:10\n"
        if name == "hci0":
            out = "hci0:\tType: Primary  Bus: UART\n\tBD Address: 00:00:00:00:00:00  ACL MTU: 0:0\n"
        return _FakeCompleted(out)

    monkeypatch.setattr(recovery, "_mac_cache", {})
    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    assert recovery.adapter_mac("hci1") == "00:11:22:33:44:55"
    assert recovery.adapter_mac("hci0") == recovery.UNKNOWN_MAC  # dead card detected


def test_adapter_mac_is_cached_between_selections(monkeypatch):
    """Selection asks per candidate per connect; without the cache a kernel
    with no sysfs address would spawn hciconfig in a tight loop."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _FakeCompleted("hci1:\n\tBD Address: 00:11:22:33:44:55\n")

    monkeypatch.setattr(recovery, "_mac_cache", {})
    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    assert recovery.adapter_mac("hci1") == "00:11:22:33:44:55"
    assert recovery.adapter_mac("hci1") == "00:11:22:33:44:55"
    assert len(calls) == 1
