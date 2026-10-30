"""The one source of the TUI's colours (tui.md §4 Style). Stdlib only.

A palette is a few brand colours (`BRANDS`); `build` derives every semantic token from them, lifting each hue toward
the text colour until it reads (WCAG AA, 4.5:1) on the surface. `TOKENS` is the active palette, switched in place by
`use`, so the Textual theme (app.py: one per palette, every token also as `$xmr-<token>` in app.tcss) and the Rich
styles drawn in Python (`color`, `style`, `Lazy`) follow a switch while the app runs. The choice is kept in
`outputs/tui/settings.json` (`load` / `save`).
"""
import json
import os
from pathlib import Path

SETTINGS = Path("outputs", "tui", "settings.json")
AA = 4.5


def mix(a, b, share):
    """`share` of colour `a` over colour `b` (hex), as one opaque hex colour."""
    pa, pb = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(y + (x - y) * share):02X}" for x, y in zip(pa, pb))


def contrast(a, b):
    """WCAG contrast ratio of two hex colours (1 to 21)."""
    def lum(c):
        ch = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        ch = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def readable(c, bg, toward):
    """`c` mixed toward `toward` (the text colour) by the smallest step that reaches AA on `bg`."""
    for step in range(51):
        out = mix(toward, c, step / 50)
        if contrast(out, bg) >= AA:
            return out
    return toward


def behind(c, bg, text):
    """A tint of `c` over `bg` for `text` to sit on: the strongest share (at most 60 %) that keeps text at AA."""
    for step in range(30, -1, -1):
        out = mix(c, bg, step / 50)
        if contrast(text, out) >= AA:
            return out
    return bg


# Brand colours per palette: paper, ink, accent, info, success, then the two status-only colours (warning, error;
# never the accent's hue). XMR4EL's own palette is Notebook; Classic is the TUI's first look.
BRANDS = {
    "Notebook": dict(paper="#FAF6EE", ink="#2B2A28", accent="#B8492A", info="#3F6E8C", success="#7A8F4E",
                     warning="#E2B35A", error="#E9707F", dark=True),
    "Notebook light": dict(paper="#FAF6EE", ink="#2B2A28", accent="#B8492A", info="#3F6E8C", success="#7A8F4E",
                           warning="#E2B35A", error="#E9707F", dark=False),
    "Classic": dict(paper="#F0F4F8", ink="#1B2229", accent="#2CF5C5", info="#5CC8FF", success="#7BD88F",
                    warning="#F5C451", error="#FF6B6B", dark=True),
}
DEFAULT = "Notebook"


def build(paper, ink, accent, info, success, warning, error, dark):
    """Every token of one palette. Dark: ink is the surface, paper the text; light: the reverse."""
    ground, surface, text = (mix(ink, "#000000", 0.7), ink, paper) if dark else (paper, mix(ink, paper, 0.05), ink)
    acc, inf, ok, warn, err = (readable(c, surface, text) for c in (accent, info, success, warning, error))
    on_accent = max((ground, text), key=lambda c: contrast(c, acc))
    dim = ground if dark else ink  # text on the pale beam chips
    border, selection = mix(text, surface, 0.2), behind(info, surface, text)
    return {
        # application
        "background": ground, "surface": surface, "panel": surface, "panel-border": border, "text": text,
        "text-muted": readable(mix(text, surface, 0.5), surface, text), "text-disabled": mix(text, surface, 0.35),
        # selection and focus
        "accent": acc,                                   # focused panel border, selected rows, current area
        "on-accent": on_accent,                          # text on an accent background
        "cursor-row": mix(acc, surface, 0.4),            # keyboard cursor: ~40 % of the selection highlight
        "cursor-blurred": mix(acc, surface, 0.2),        # Dashboard cursor row of an unfocused panel
        "focus-outline": on_accent,                      # edge bars of the cursor row when it is also selected
        "text-selection": selection, "match": selection,  # mouse selection, search matches
        # status
        "success": ok, "warning": warn, "error": err, "info": inf, "unknown": mix(text, surface, 0.35),
        "warning-on-accent": on_accent,                  # caution symbol over a selected (accent) row
        # comparison
        "baseline-row": acc, "member-row": acc, "on-selected": on_accent,
        "delta": readable(mix(text, surface, 0.5), surface, text),
        "delta-unavailable": mix(text, surface, 0.35),   # Δ — : a delta withheld (not verifiably comparable)
        "spec-differs": inf,                             # Specs value other than the baseline's
        "divider": border,                               # rule between summary rows and Caution notes
        # beam identity: a beam value always gets the same one (beam_color); pale chips, none is the accent
        "beams": (mix(paper, accent, 0.55), mix(paper, info, 0.45), mix(paper, success, 0.3), mix(paper, warning, 0.2),
                  mix(paper, info, 0.75), mix(paper, success, 0.65)),
        "beam-unknown": mix(text, surface, 0.35), "on-beam": dim,
        # navbar, notifications, footer: a severity tints only the title chip (text = "text"), never the whole box
        "overlay": mix(text, surface, 0.07),             # toasts, notification center, footer feedback row
        "tint-success": mix(ok, surface, 0.3), "tint-error": mix(err, surface, 0.3),
        "tint-warning": mix(warn, surface, 0.3), "tint-information": border,
        "confirm-row": mix(warn, surface, 0.15),         # footer while a confirmation waits
        "feedback-row": mix(acc, surface, 0.22),         # footer while transient feedback shows (never confirm-row)
        # Model Builder
        "block": mix(text, surface, 0.07),               # a pipeline slot (grid) and an unselected tree tab
        "section": mix(acc, surface, 0.14),              # JSON preview lines of the component being edited
        "dark": dark,
    }


PALETTES = {name: build(**brand) for name, brand in BRANDS.items()}
TOKENS = dict(PALETTES[DEFAULT])  # the active palette; `use` switches it in place


def use(name):
    """Make palette `name` the active one (unknown: the default). Returns the name used."""
    name = name if name in PALETTES else DEFAULT
    TOKENS.clear()
    TOKENS.update(PALETTES[name])
    return name


def load(root):
    """The saved palette name, or the default (missing, unreadable or unknown never fails)."""
    try:
        name = json.loads((root / SETTINGS).read_text()).get("palette")
    except (OSError, ValueError, AttributeError):
        return DEFAULT
    return name if name in PALETTES else DEFAULT


def save(root, name):
    """Keep palette `name` in the settings file (other keys kept; write-then-rename). Raises OSError."""
    path = root / SETTINGS
    try:
        data = json.loads(path.read_text())
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    data["palette"] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


class Lazy:
    """A style table built from the active palette at each lookup, so a palette switch reaches it at once:
    `Lazy(lambda: {"done": style("success")})[key]`."""

    def __init__(self, table):
        self.table = table

    def __getitem__(self, key):
        return self.table()[key]

    def get(self, key, default=None):
        return self.table().get(key, default)


def color(token):
    return TOKENS[token]


def style(token, extra=""):
    """Rich style string of a colour token: `style("warning", "bold")`."""
    return f"{extra} {TOKENS[token]}".strip()


def chip(severity):
    """Rich style of a title chip: text on the severity's muted tint (unknown severity: neutral)."""
    return f"{TOKENS['text']} on {TOKENS.get(f'tint-{severity}', TOKENS['tint-information'])}"


def beam_color(beam):
    """Palette colour of a beam size: `beam` mod the palette size, so one value has one colour whatever the
    discovery order (10 and 20 differ; values that collide still show their number). Unknown: beam-unknown."""
    beams = TOKENS["beams"]
    ok = isinstance(beam, int) and not isinstance(beam, bool) and beam >= 0
    return beams[beam % len(beams)] if ok else TOKENS["beam-unknown"]


def css_variables(tokens=None):
    """Tokens (default: the active palette's) as Textual theme variables, prefixed so none shadows Textual's own:
    $xmr-cursor-row, ..."""
    return {f"xmr-{k}": v for k, v in (TOKENS if tokens is None else tokens).items() if isinstance(v, str)}
