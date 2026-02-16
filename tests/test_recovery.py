"""Tests for recovery module."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bleak_connection_manager.recovery import (
    PROFILE_BATTERY,
    PROFILE_ON_DEMAND,
    PROFILE_SENSOR,
    EscalationAction,
    EscalationConfig,
    EscalationPolicy,
    is_bluetoothd_alive,
)


def test_default_config():
    config = EscalationConfig()
    assert config.reset_adapter is False
    assert config.rotate_after == 2
    assert config.clear_after == 4
    assert config.reset_after == 6


def test_profiles():
    assert PROFILE_BATTERY.reset_adapter is True
    assert PROFILE_SENSOR.reset_adapter is False
    assert PROFILE_ON_DEMAND.rotate_after == 1


def test_escalation_retry():
    policy = EscalationPolicy(["hci0"])
    action = policy.on_failure("hci0")
    assert action == EscalationAction.DIAGNOSE  # after 1 failure


def test_escalation_rotate_after_threshold():
    config = EscalationConfig(rotate_after=2)
    policy = EscalationPolicy(["hci0", "hci1"], config=config)
    policy.on_failure("hci0")  # 1 -> DIAGNOSE
    action = policy.on_failure("hci0")  # 2 -> ROTATE
    assert action == EscalationAction.ROTATE_ADAPTER


def test_escalation_clear_after_threshold():
    config = EscalationConfig(clear_after=3, rotate_after=2)
    policy = EscalationPolicy(["hci0"], config=config)
    policy.on_failure("hci0")  # 1
    policy.on_failure("hci0")  # 2
    action = policy.on_failure("hci0")  # 3 -> CLEAR_BLUEZ
    assert action == EscalationAction.CLEAR_BLUEZ


def test_escalation_reset_disabled():
    config = EscalationConfig(reset_adapter=False, reset_after=3)
    policy = EscalationPolicy(["hci0"], config=config)
    for _ in range(10):
        action = policy.on_failure("hci0")
    assert action != EscalationAction.RESET_ADAPTER


def test_escalation_reset_enabled():
    config = EscalationConfig(reset_adapter=True, reset_after=3, reset_cooldown=0)
    policy = EscalationPolicy(["hci0"], config=config)
    policy.on_failure("hci0")  # 1
    policy.on_failure("hci0")  # 2
    action = policy.on_failure("hci0")  # 3 -> RESET
    assert action == EscalationAction.RESET_ADAPTER


def test_escalation_reset_cooldown():
    config = EscalationConfig(
        reset_adapter=True, reset_after=2, reset_cooldown=9999
    )
    policy = EscalationPolicy(["hci0"], config=config)
    # Simulate a recent reset so cooldown applies
    policy.record_reset("hci0")
    policy.on_failure("hci0")
    action = policy.on_failure("hci0")
    # Should not reset because cooldown hasn't elapsed since last reset
    assert action != EscalationAction.RESET_ADAPTER


def test_on_success_resets_counter():
    policy = EscalationPolicy(["hci0"])
    policy.on_failure("hci0")
    policy.on_failure("hci0")
    policy.on_success("hci0")
    action = policy.on_failure("hci0")
    # Should start fresh — 1 failure -> DIAGNOSE
    assert action == EscalationAction.DIAGNOSE


def test_max_escalation_ceiling():
    config = EscalationConfig(
        reset_adapter=True,
        reset_after=2,
        max_escalation=EscalationAction.ROTATE_ADAPTER,
    )
    policy = EscalationPolicy(["hci0"], config=config)
    for _ in range(10):
        action = policy.on_failure("hci0")
    assert action != EscalationAction.RESET_ADAPTER


def test_record_reset():
    config = EscalationConfig(reset_adapter=True, reset_after=2, reset_cooldown=0)
    policy = EscalationPolicy(["hci0"], config=config)
    policy.record_reset("hci0")
    # Failure counter should be reset to 0
    action = policy.on_failure("hci0")
    assert action == EscalationAction.DIAGNOSE  # fresh start


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", False)
async def test_reset_adapter_non_linux():
    from bleak_connection_manager.recovery import reset_adapter

    result = await reset_adapter("hci0")
    assert result is False


# ── is_bluetoothd_alive tests ──────────────────────────────────────


@patch("bleak_connection_manager.recovery.IS_LINUX", False)
def test_is_bluetoothd_alive_non_linux():
    assert is_bluetoothd_alive() is True


@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.os.path.exists", return_value=True)
@patch("bleak_connection_manager.recovery.os.listdir")
def test_is_bluetoothd_alive_found(mock_listdir, mock_exists):
    mock_listdir.return_value = ["1", "42", "abc"]

    comm_values = {
        "/proc/1/comm": "systemd\n",
        "/proc/42/comm": "bluetoothd\n",
    }

    def fake_open(path, *args, **kwargs):
        content = comm_values.get(path, "other\n")
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=content)))
        m.__exit__ = MagicMock(return_value=False)
        return m

    with patch("builtins.open", side_effect=fake_open):
        assert is_bluetoothd_alive() is True


@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.os.path.exists", return_value=True)
@patch("bleak_connection_manager.recovery.os.listdir")
def test_is_bluetoothd_alive_not_found(mock_listdir, mock_exists):
    mock_listdir.return_value = ["1", "42"]

    def fake_open(path, *args, **kwargs):
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value="other\n")))
        m.__exit__ = MagicMock(return_value=False)
        return m

    with patch("builtins.open", side_effect=fake_open):
        assert is_bluetoothd_alive() is False


@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.os.listdir")
def test_is_bluetoothd_alive_proc_unavailable(mock_listdir):
    mock_listdir.side_effect = OSError("No such file or directory")
    assert is_bluetoothd_alive() is False


# ── reset_adapter with bluetoothd check ────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive", return_value=True)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_reset_adapter_bluetoothd_alive(mock_sleep, mock_alive):
    """Successful reset with bluetoothd surviving."""
    from bleak_connection_manager.recovery import reset_adapter

    mock_recover = AsyncMock(return_value=True)
    mock_module = MagicMock()
    mock_module.recover_adapter = mock_recover

    import sys
    with patch.dict(sys.modules, {"bluetooth_auto_recovery": mock_module}):
        result = await reset_adapter("hci0")

    assert result is True
    mock_recover.assert_awaited_once_with(0, "00:00:00:00:00:00")
    mock_sleep.assert_awaited_once_with(1.0)
    mock_alive.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive", return_value=False)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_reset_adapter_bluetoothd_dead(mock_sleep, mock_alive):
    """If bluetoothd crashes during reset, returns False."""
    from bleak_connection_manager.recovery import reset_adapter

    mock_recover = AsyncMock(return_value=True)
    mock_module = MagicMock()
    mock_module.recover_adapter = mock_recover

    import sys
    with patch.dict(sys.modules, {"bluetooth_auto_recovery": mock_module}):
        result = await reset_adapter("hci0")

    assert result is False
    mock_alive.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive", return_value=True)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_reset_adapter_recover_fails(mock_sleep, mock_alive):
    """If bluetooth-auto-recovery reports failure, returns False."""
    from bleak_connection_manager.recovery import reset_adapter

    mock_recover = AsyncMock(return_value=False)
    mock_module = MagicMock()
    mock_module.recover_adapter = mock_recover

    import sys
    with patch.dict(sys.modules, {"bluetooth_auto_recovery": mock_module}):
        result = await reset_adapter("hci0")

    assert result is False
    mock_alive.assert_not_called()  # never reaches bluetoothd check
