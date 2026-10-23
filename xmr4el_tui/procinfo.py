"""Process identity (tui.md §9), stdlib only: the runner records its own in status.json, the TUI compares it with
the live process before calling a run RUNNING or signalling it. No Textual import (the runner imports this module
without the tui extra).

A PID alone is no proof: PIDs are reused. A process is the recorded one only when, seen from the same runtime (host
boot and, on Linux, PID namespace), its UID, start token, process group and session all match.
"""
import os
import platform
import socket
import subprocess
from pathlib import Path

FIELDS = ("uid", "start", "pgid", "sid")


def _run(*cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def runtime():
    """Identity of the runtime this process sees PIDs in, or None when it cannot be read."""
    if platform.system() == "Linux":
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return f"{boot}:{os.readlink('/proc/self/ns/pid')}"
        except OSError:
            return None
    return _run("sysctl", "-n", "kern.boottime")


def identity(pid):
    """uid, start token, pgid, sid of live process `pid` (a key is None when inspection is denied), or None if no
    such process exists."""
    if platform.system() == "Linux":
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            uid = os.stat(f"/proc/{pid}").st_uid
        except FileNotFoundError:
            return None
        except OSError:
            stat, uid = None, None
        # field 22 = start time in clock ticks since boot; the command name (field 2) may contain spaces/parens
        start = stat.rsplit(")", 1)[1].split()[19] if stat else None
    else:
        out = _run("ps", "-o", "uid=,lstart=", "-p", str(pid))
        if out is None:
            return None if not _alive(pid) else {k: None for k in FIELDS}
        uid, start = out.split(None, 1)
        uid = int(uid)
    ident = {"uid": uid, "start": start, "pgid": None, "sid": None}
    try:
        ident["pgid"], ident["sid"] = os.getpgid(pid), os.getsid(pid)
    except ProcessLookupError:
        return None
    except OSError:
        pass
    return ident


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def own_identity():
    """What the runner records in status.json. isolated_group: this process leads its own group and session (TUI
    children, start_new_session=True), so signalling the group reaches only this run."""
    pid = os.getpid()
    ident = identity(pid) or {k: None for k in FIELDS}
    return {"pid": pid, "host": socket.gethostname(), **ident, "runtime": runtime(),
            "isolated_group": pid == ident["pgid"] == ident["sid"]}


def same(recorded, live):
    """True when every identity field is known on both sides and equal."""
    return live is not None and all(recorded.get(k) is not None and recorded.get(k) == live.get(k) for k in FIELDS)
