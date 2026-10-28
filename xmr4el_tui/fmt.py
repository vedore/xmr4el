"""Display formats (tui.md §4). Stdlib only."""
import locale
import math
import os
import re
import sys

from xmr4el_tui.theme import style

DASH = "—"
MARKERS = {"done": "+", "current": ">", "pending": "·", "failed": "x"}  # plain text, no symbol glyphs
STATE_STYLE = {"RUNNING": style("info", "bold"), "UNVERIFIED": style("warning"), "LOST": style("error"),
               "COMPLETED": style("success"), "FAILED": style("error"), "STOPPED": style("warning"), "UNKNOWN": ""}
STATE_SHORT = {"RUNNING": "RUN", "UNVERIFIED": "UNVER", "LOST": "LOST", "COMPLETED": "DONE", "FAILED": "FAIL",
               "STOPPED": "STOP", "UNKNOWN": "?"}
ASCII_MARKERS = {"done": "ok", "current": ">>", "pending": "..", "failed": "xx"}
# terminal escape sequences (CSI, OSC ended by BEL or ST) and any other C0/C1 control character but tab and newline
ESCAPES = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]|(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)?")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean(text):
    """File text made safe to draw: escape sequences removed, other control characters shown as \\xNN. Rich does
    not strip ESC, so text from run files would otherwise reach the terminal as commands."""
    return CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", ESCAPES.sub("", text))


def use_ascii():
    """ASCII markers when NO_COLOR is set or the terminal encoding is not UTF-8."""
    encoding = getattr(sys.stdout, "encoding", None) or locale.getpreferredencoding(False)
    return "NO_COLOR" in os.environ or "utf" not in encoding.lower()


def duration(seconds):
    """m:ss under an hour, h:mm:ss above; None (or not a finite number) -> dash."""
    if not number(seconds):
        return DASH
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def stamp(dt, long=False):
    """MM-DD HH:MM in tables, YYYY-MM-DD HH:MM in details."""
    if dt is None:
        return DASH
    return dt.strftime("%Y-%m-%d %H:%M" if long else "%m-%d %H:%M")


def number(x):
    """A finite JSON number (not a bool, NaN or inf): what the formats below accept; anything else shows as a dash."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and (isinstance(x, int) or math.isfinite(x))


def metric(x):
    return f"{x:.4f}" if number(x) else DASH


def delta(x):
    return f"{x:+.4f}" if number(x) else DASH


def count(n):
    return f"{n:,}" if number(n) else DASH


def pct(a, b):
    return f"{a / b:.1%}" if number(a) and number(b) and b else DASH


def size(n):
    """Bytes in binary units: 9.0 GiB."""
    if not number(n):
        return DASH
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def ms(x):
    return f"{x:.2f} ms/query" if number(x) else DASH


def tree(tree_id):
    return tree_id[:8] if isinstance(tree_id, str) and tree_id else DASH


def marker(state, ascii=False):
    return (ASCII_MARKERS if ascii else MARKERS)[state]
