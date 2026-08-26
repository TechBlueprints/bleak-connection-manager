# -*- coding: utf-8 -*-
"""Tests for the sitewide autowire hook (packaging/bcm_autowire.py).

Run in a subprocess, because the hook's whole contract is about interpreter
startup state: a stub bleak stands in for a process's own vendored copy,
bcm_autowire is imported the way the .pth would, and the assertion is made
on what `import bleak` then yields. The autowire root is pointed at this
repository so the catcher comes from the working tree.
"""

import os
import subprocess
import sys
import textwrap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(script, tmp_path, **extra_env):
    stub = tmp_path / "stub" / "bleak"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(
        textwrap.dedent(
            """
            class BleakClient:
                def __init__(self, *a, **k): pass
            class BleakScanner:
                def __init__(self, *a, **k): pass
            """
        )
    )
    (stub / "exc.py").write_text(
        "class BleakError(Exception):\n    pass\n"
        "class BleakDBusError(BleakError):\n"
        "    def __init__(self, dbus_error, error_body):\n"
        "        super().__init__(dbus_error, *error_body)\n"
        "    @property\n"
        "    def dbus_error(self):\n"
        "        return self.args[0]\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path / "stub"), os.path.join(REPO, "packaging"), os.path.join(REPO, "src")]
    )
    env["BCM_AUTOWIRE_ROOT"] = REPO
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=60
    )


def test_importing_bleak_installs_the_catcher(tmp_path):
    """The .pth path: dormant registration, then a plain `import bleak`
    yields the rebound catcher class - before the importing code could
    have captured the original."""
    result = _run(
        "import bcm_autowire\n"
        "import bleak\n"
        "print(bleak.BleakClient.__name__)\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BLEConnection"


def test_the_kill_switch_leaves_bleak_alone(tmp_path):
    result = _run(
        "import bcm_autowire\n"
        "import bleak\n"
        "print(bleak.BleakClient.__name__)\n",
        tmp_path,
        BCM_AUTOWIRE="0",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BleakClient"


def test_a_broken_root_never_breaks_the_process(tmp_path):
    """The one absolute rule: whatever state /data/bcm is in, importing
    bcm_autowire and then bleak must not raise."""
    result = _run(
        "import bcm_autowire\n"
        "import bleak\n"
        "print(bleak.BleakClient.__name__)\n",
        tmp_path,
        BCM_AUTOWIRE_ROOT=str(tmp_path / "definitely-not-there"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BleakClient"


def test_a_bcm_aware_process_is_never_autowired(tmp_path):
    """Clint's rule: no autowiring from inside BCM. A process that imports
    bleak_connection_manager is a deliberate consumer that will install
    the catcher itself, with its own owner and config - the finder stands
    down the moment the package appears in sys.modules, including when the
    bleak import is triggered BY the consumer's own explicit install."""
    result = _run(
        "import bcm_autowire\n"
        "import bleak_connection_manager\n"
        "import bleak\n"
        "print(bleak.BleakClient.__name__)\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BleakClient"  # untouched: theirs to install


def test_an_explicit_install_after_autowire_registration_wins_cleanly(tmp_path):
    """The consumer's own install_bleak_catcher call must see a clean
    world: its internal bleak import must not be intercepted mid-install."""
    result = _run(
        "import bcm_autowire\n"
        "from bleak_connection_manager import install_bleak_catcher\n"
        "install_bleak_catcher('svc', claim_dir='%s')\n"
        "import bleak\n"
        "print(bleak.BleakClient.__name__, bleak.BleakClient is not None)\n" % (tmp_path / "claims"),
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BLEConnection True"  # installed once, by the consumer


def test_a_wire_leaves_a_durable_event_record(tmp_path):
    """Clint's observability request: /data/bcm must answer "what BLE
    software has ever run on this box" after the process is gone - one
    appended line per wire with pid, full argv, cwd, and whether the
    process brought its own bleak or was served the shared stack."""
    import re

    events = os.path.join(REPO, "autowire-events.log")
    if os.path.exists(events):
        os.unlink(events)
    try:
        result = _run(
            "import bcm_autowire\n"
            "import bleak\n"
            "print(bleak.BleakClient.__name__)\n",
            tmp_path,
        )
        assert result.returncode == 0, result.stderr
        with open(events) as f:
            line = f.read()
        assert re.search(r"pid=\d+", line)
        assert "owner=autowire-" in line
        assert "bleak=own" in line  # the stub was the process's own bleak
        assert 'argv="' in line and "cwd=" in line
        # On Linux (where this ships) argv comes from /proc/self/cmdline and
        # carries the -c code; macOS has no /proc, so the subprocess here
        # exercises the sys.argv fallback. The parsing itself is covered
        # directly below, on both platforms.
        if os.path.exists("/proc/self/cmdline"):
            assert "import bcm_autowire" in line
            assert "owner=autowire--c" not in line
    finally:
        if os.path.exists(events):
            os.unlink(events)


def test_no_event_record_without_a_wire(tmp_path):
    events = os.path.join(REPO, "autowire-events.log")
    if os.path.exists(events):
        os.unlink(events)
    result = _run(
        "import bcm_autowire\n"
        "import bleak\n",
        tmp_path,
        BCM_AUTOWIRE="0",
    )
    assert result.returncode == 0, result.stderr
    assert not os.path.exists(events)  # kill switch: no wire, no record


def test_the_owner_is_derived_from_the_real_command_line(tmp_path):
    """A script gives its basename, -m gives the module, -c has no name to
    give and says so rather than degrading to a flag."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bcm_autowire_probe", os.path.join(REPO, "packaging", "bcm_autowire.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._derive_owner(["/usr/bin/python3", "/data/apps/foo/driver.py"]) == "driver.py"
    assert mod._derive_owner(["python3", "-u", "/data/apps/foo/driver.py"]) == "driver.py"
    assert mod._derive_owner(["python3", "-m", "pkg.mod"]) == "pkg.mod"
    assert mod._derive_owner(["python3", "-c", "import bleak"]) == "inline"
    assert mod._derive_owner(["python3"]) == "python"


def _probe_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bcm_autowire_probe", os.path.join(REPO, "packaging", "bcm_autowire.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_command_line_comes_from_proc_not_sys_argv(tmp_path, monkeypatch):
    """The prod bug: `python3 -c "..."` has sys.argv == ['-c'], so the code -
    the only identifying thing about a mystery one-liner - was lost. /proc
    carries the true NUL-separated line."""
    mod = _probe_module()
    fake = tmp_path / "cmdline"
    fake.write_bytes(b"python3\x00-c\x00import bleak; go()\x00")
    monkeypatch.setattr(mod, "_CMDLINE", str(fake))

    assert mod._command_line() == ["python3", "-c", "import bleak; go()"]
    assert mod._derive_owner(mod._command_line()) == "inline"


def test_the_command_line_falls_back_when_proc_is_absent(tmp_path, monkeypatch):
    mod = _probe_module()
    monkeypatch.setattr(mod, "_CMDLINE", str(tmp_path / "nope"))
    monkeypatch.setattr(mod.sys, "argv", ["/data/apps/foo/driver.py", "--flag"])

    assert mod._command_line() == ["/data/apps/foo/driver.py", "--flag"]
