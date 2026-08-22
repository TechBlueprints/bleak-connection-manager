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
    (stub / "exc.py").write_text("class BleakError(Exception):\n    pass\n")
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
