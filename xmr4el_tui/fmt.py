"""Display formats (tui.md §4). Stdlib only."""
import locale
import os
import sys

DASH = "—"
MARKERS = {"done": "✓", "current": "▸", "pending": "·", "failed": "✗"}
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


def metric(x):
    return DASH if x is None else f"{x:.4f}"


def delta(x):
    return DASH if x is None else f"{x:+.4f}"


def count(n):
    return DASH if n is None else f"{n:,}"


def pct(a, b):
    return DASH if a is None or not b else f"{a / b:.1%}"


def ms(x):
    return DASH if x is None else f"{x:.2f} ms/query"


def tree(tree_id):
    return tree_id[:8] if tree_id else DASH


def marker(state, ascii=False):
    return (ASCII_MARKERS if ascii else MARKERS)[state]
