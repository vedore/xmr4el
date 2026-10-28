"""Display formats (tui.md §4). Stdlib only."""
import locale
import os
import sys

DASH = "—"
MARKERS = {"done": "✓", "current": "▸", "pending": "·", "failed": "✗"}
STATE_STYLE = {"RUNNING": "bold #5CC8FF", "UNVERIFIED": "#F5C451", "LOST": "#FF6B6B", "COMPLETED": "#7BD88F",
               "FAILED": "#FF6B6B", "STOPPED": "#F5C451", "UNKNOWN": ""}
STATE_SHORT = {"RUNNING": "RUN", "UNVERIFIED": "UNVER", "LOST": "LOST", "COMPLETED": "DONE", "FAILED": "FAIL",
               "STOPPED": "STOP", "UNKNOWN": "?"}
ASCII_MARKERS = {"done": "ok", "current": ">>", "pending": "..", "failed": "xx"}


def use_ascii():
    """ASCII markers when NO_COLOR is set or the terminal encoding is not UTF-8."""
    encoding = getattr(sys.stdout, "encoding", None) or locale.getpreferredencoding(False)
    return "NO_COLOR" in os.environ or "utf" not in encoding.lower()


def duration(seconds):
    """m:ss under an hour, h:mm:ss above; None -> dash."""
    if seconds is None:
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
    """A JSON number (not a bool): what the formats below accept; anything else shows as a dash."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


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
