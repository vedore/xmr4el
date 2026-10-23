"""procinfo: own identity, a TUI-style child is an isolated group, a gone or changed process does not match."""
import os
import subprocess
import sys

from xmr4el_tui import procinfo


def test_identity():
    own = procinfo.own_identity()
    assert own["pid"] == os.getpid() and own["uid"] == os.getuid() and own["runtime"] == procinfo.runtime()
    assert own["start"] and procinfo.same(own, procinfo.identity(os.getpid()))
    assert not procinfo.same({**own, "start": "other"}, procinfo.identity(os.getpid()))
    assert not procinfo.same({**own, "sid": None}, procinfo.identity(os.getpid()))

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        live = procinfo.identity(child.pid)
        assert live["pgid"] == live["sid"] == child.pid and live["start"] and not procinfo.same(own, live)
    finally:
        child.kill()
        child.wait()
    assert procinfo.identity(child.pid) is None
