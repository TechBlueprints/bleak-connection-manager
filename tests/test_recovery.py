"""Tests for recovery module."""

import asyncio
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


# ── failure_count tests ────────────────────────────────────────────


def test_failure_count_initial():
    policy = EscalationPolicy(["hci0", "hci1"])
    assert policy.failure_count("hci0") == 0
    assert policy.failure_count("hci1") == 0


def test_failure_count_tracks():
    policy = EscalationPolicy(["hci0"])
    policy.on_failure("hci0")
    assert policy.failure_count("hci0") == 1
    policy.on_failure("hci0")
    assert policy.failure_count("hci0") == 2


def test_failure_count_resets_on_success():
    policy = EscalationPolicy(["hci0"])
    policy.on_failure("hci0")
    policy.on_failure("hci0")
    policy.on_success("hci0")
    assert policy.failure_count("hci0") == 0


def test_failure_count_unknown_adapter():
    policy = EscalationPolicy(["hci0"])
    assert policy.failure_count("hci99") == 0


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
@patch("bleak_connection_manager.recovery.os.path.isfile", return_value=False)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_reset_adapter_bluetoothd_dead(mock_sleep, mock_isfile, mock_alive):
    """If bluetoothd crashes during reset and restart fails, returns False."""
    from bleak_connection_manager.recovery import reset_adapter

    mock_recover = AsyncMock(return_value=True)
    mock_module = MagicMock()
    mock_module.recover_adapter = mock_recover

    import sys
    with patch.dict(sys.modules, {"bluetooth_auto_recovery": mock_module}):
        result = await reset_adapter("hci0")

    assert result is False
    assert mock_alive.call_count >= 1


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


# ── restart_bluetoothd tests ──────────────────────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", False)
async def test_restart_bluetoothd_non_linux():
    """Non-Linux returns True (assume OK)."""
    from bleak_connection_manager.recovery import restart_bluetoothd

    assert await restart_bluetoothd() is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive", return_value=True)
async def test_restart_bluetoothd_already_running(mock_alive):
    """If already alive, returns True without starting anything."""
    from bleak_connection_manager.recovery import restart_bluetoothd

    assert await restart_bluetoothd() is True
    mock_alive.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive")
@patch("bleak_connection_manager.recovery.os.path.isfile", return_value=False)
async def test_restart_bluetoothd_no_init_script(mock_isfile, mock_alive):
    """If init script is missing, returns False."""
    from bleak_connection_manager.recovery import restart_bluetoothd

    mock_alive.return_value = False
    assert await restart_bluetoothd() is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive")
@patch("bleak_connection_manager.recovery.os.path.isfile", return_value=True)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_restart_bluetoothd_success(mock_sleep, mock_isfile, mock_alive):
    """Init script runs, bluetoothd comes back alive."""
    from bleak_connection_manager.recovery import restart_bluetoothd

    mock_alive.side_effect = [False, True]

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"Starting bluetooth\n", b""))

    with patch(
        "bleak_connection_manager.recovery.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ):
        assert await restart_bluetoothd() is True


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive")
@patch("bleak_connection_manager.recovery.os.path.isfile", return_value=True)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_restart_bluetoothd_script_fails(mock_sleep, mock_isfile, mock_alive):
    """Init script exits non-zero — returns False."""
    from bleak_connection_manager.recovery import restart_bluetoothd

    mock_alive.return_value = False

    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error\n"))

    with patch(
        "bleak_connection_manager.recovery.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ):
        assert await restart_bluetoothd() is False


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive")
@patch("bleak_connection_manager.recovery.os.path.isfile", return_value=True)
async def test_restart_bluetoothd_timeout(mock_isfile, mock_alive):
    """Init script hangs past timeout — returns False."""
    from bleak_connection_manager.recovery import restart_bluetoothd

    mock_alive.return_value = False

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch(
        "bleak_connection_manager.recovery.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ):
        assert await restart_bluetoothd(timeout=0.1) is False
    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive")
@patch("bleak_connection_manager.recovery.os.path.isfile", return_value=True)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_restart_bluetoothd_exits_ok_but_proc_dead(
    mock_sleep, mock_isfile, mock_alive,
):
    """Init script exits 0 but bluetoothd not found in /proc — returns False."""
    from bleak_connection_manager.recovery import restart_bluetoothd

    mock_alive.return_value = False

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))

    with patch(
        "bleak_connection_manager.recovery.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ):
        assert await restart_bluetoothd() is False


# ── reset_adapter auto-restart of bluetoothd ──────────────────────


@pytest.mark.asyncio
@patch("bleak_connection_manager.recovery.IS_LINUX", True)
@patch("bleak_connection_manager.recovery.is_bluetoothd_alive")
@patch("bleak_connection_manager.recovery.os.path.isfile", return_value=True)
@patch("bleak_connection_manager.recovery.asyncio.sleep", new_callable=AsyncMock)
async def test_reset_adapter_auto_restarts_bluetoothd(
    mock_sleep, mock_isfile, mock_alive,
):
    """If bluetoothd dies during reset but restart succeeds, returns True."""
    from bleak_connection_manager.recovery import reset_adapter

    # is_bluetoothd_alive: first call (post-reset check) = False,
    # second call (restart_bluetoothd guard) = False,
    # third call (restart_bluetoothd verify) = True
    mock_alive.side_effect = [False, False, True]

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))

    mock_recover = AsyncMock(return_value=True)
    mock_module = MagicMock()
    mock_module.recover_adapter = mock_recover

    import sys
    with (
        patch.dict(sys.modules, {"bluetooth_auto_recovery": mock_module}),
        patch(
            "bleak_connection_manager.recovery.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ),
    ):
        result = await reset_adapter("hci0")

    assert result is True
