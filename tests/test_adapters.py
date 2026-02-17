"""Tests for adapters module."""

import tempfile
from unittest.mock import MagicMock, patch

from bleak.backends.device import BLEDevice

from bleak_connection_manager.adapters import (
    discover_adapters,
    make_device_for_adapter,
    pick_adapter,
    score_adapter,
    select_best_adapter,
)
from bleak_connection_manager.const import LockConfig
from bleak_connection_manager.recovery import EscalationConfig, EscalationPolicy


def test_pick_adapter_single():
    assert pick_adapter(["hci0"], 1) == "hci0"
    assert pick_adapter(["hci0"], 2) == "hci0"
    assert pick_adapter(["hci0"], 3) == "hci0"


def test_pick_adapter_round_robin():
    adapters = ["hci0", "hci1", "hci2"]
    assert pick_adapter(adapters, 1) == "hci0"
    assert pick_adapter(adapters, 2) == "hci1"
    assert pick_adapter(adapters, 3) == "hci2"
    assert pick_adapter(adapters, 4) == "hci0"  # wraps around


def test_pick_adapter_empty():
    assert pick_adapter([], 1) == "hci0"


def test_make_device_for_adapter():
    device = BLEDevice(
        "AA:BB:CC:DD:EE:FF",
        "TestDevice",
        {"path": "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"},
    )
    new_device = make_device_for_adapter(device, "hci1")

    assert new_device.address == "AA:BB:CC:DD:EE:FF"
    assert new_device.name == "TestDevice"
    assert "hci1" in new_device.details["path"]


def test_make_device_for_adapter_no_details():
    device = BLEDevice("AA:BB:CC:DD:EE:FF", "TestDevice", {})
    new_device = make_device_for_adapter(device, "hci0")
    assert new_device.details["path"] == "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"


@patch("bleak_connection_manager.adapters.IS_LINUX", False)
def test_discover_adapters_non_linux():
    assert discover_adapters() == ["hci0"]


# ── score_adapter tests ───────────────────────────────────────────


def test_score_adapter_no_context():
    """Base score when no policy/lock info is provided."""
    s = score_adapter("hci0")
    assert s == 100.0


def test_score_adapter_penalizes_failures():
    """Adapters with failures get penalized."""
    policy = EscalationPolicy(["hci0", "hci1"])
    policy.on_failure("hci0")
    policy.on_failure("hci0")

    s0 = score_adapter("hci0", escalation_policy=policy)
    s1 = score_adapter("hci1", escalation_policy=policy)

    # hci0 has 2 failures, hci1 has 0 — hci1 should score higher
    assert s1 > s0


def test_score_adapter_penalizes_in_progress():
    """Adapters with in-progress connections get penalized."""
    in_progress = {"hci0": 2, "hci1": 0}

    s0 = score_adapter("hci0", in_progress=in_progress)
    s1 = score_adapter("hci1", in_progress=in_progress)

    assert s1 > s0


def test_score_adapter_penalizes_no_free_slots():
    """Adapter with no free slots gets heavily penalized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LockConfig(
            enabled=True,
            lock_dir=tmpdir,
            max_slots=1,
        )
        # Hold the only slot
        import fcntl
        import os

        slot_path = config.path_for_slot("hci0", 0)
        fd = os.open(slot_path, os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            s = score_adapter("hci0", lock_config=config)
            # Should be heavily penalized (below 0)
            assert s < 0
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ── select_best_adapter tests ─────────────────────────────────────


def test_select_best_adapter_prefers_no_failures():
    """Selects adapter with fewer failures."""
    policy = EscalationPolicy(["hci0", "hci1"])
    policy.on_failure("hci0")
    policy.on_failure("hci0")
    policy.on_failure("hci0")

    best = select_best_adapter(
        ["hci0", "hci1"], escalation_policy=policy
    )
    assert best == "hci1"


def test_select_best_adapter_empty():
    assert select_best_adapter([]) == "hci0"


def test_select_best_adapter_single():
    assert select_best_adapter(["hci1"]) == "hci1"


def test_select_best_adapter_equal_scores():
    """When scores are equal, first adapter wins."""
    best = select_best_adapter(["hci0", "hci1"])
    assert best == "hci0"
