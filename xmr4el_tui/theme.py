"""The one source of the TUI's colours (tui.md §4 Style). Stdlib only.

`DARK` holds every hex value; the Textual theme (app.THEME: Textual's own variables, and every token as
`$xmr-<token>` in app.tcss) and the Rich styles drawn in Python (`color`, `style`) both read `TOKENS`. Another
theme = another dict with the same keys, assigned to `TOKENS` before the app starts.
"""


def mix(a, b, share):
    """`share` of colour `a` over colour `b` (hex), as one opaque hex colour."""
    pa, pb = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(y + (x - y) * share):02X}" for x, y in zip(pa, pb))


ACCENT, SURFACE = "#2CF5C5", "#1B2229"
SUCCESS, WARNING, ERROR = "#7BD88F", "#F5C451", "#FF6B6B"
DARK = {
    # application
    "background": "#101418", "surface": SURFACE, "panel": SURFACE,
    "panel-border": "#3A4754", "text": "#F0F4F8", "text-muted": "#A5B4C3", "text-disabled": "#5C6B7A",
    # selection and focus
    "accent": ACCENT,                                # focused panel border, selected rows, current area
    "on-accent": "#101418",                          # text on an accent background
    "cursor-row": mix(ACCENT, SURFACE, 0.4),         # keyboard cursor: ~40 % of the selection highlight
    "cursor-blurred": "#2B5C55",                     # Dashboard cursor row of an unfocused panel
    "focus-outline": "#101418",                      # edge bars of the cursor row when it is also selected
    "text-selection": "#3B6E99", "match": "#3B6E99",  # mouse selection, search matches
    # status
    "success": SUCCESS, "warning": WARNING, "error": ERROR, "info": "#5CC8FF", "unknown": "#5C6B7A",
    "warning-on-accent": "#7A4300",                  # caution symbol over a selected (accent) row
    # comparison
    "baseline-row": ACCENT, "member-row": ACCENT, "on-selected": "#101418", "delta": "#A5B4C3",
    "delta-unavailable": "#5C6B7A",                  # Δ — : a delta withheld (not verifiably comparable)
    "spec-differs": "#5CC8FF",                       # Specs value other than the baseline's
    "divider": "#3A4754",                            # rule between summary rows and Caution notes
    # beam identity: a beam value always gets the same one (beam_color)
    "beams": ("#E6C7A1", "#B48CFF", "#FF6FB5", "#5CC8FF", "#A3E635", "#FF9F43"),
    "beam-unknown": "#5C6B7A", "on-beam": "#101418",
    # navbar, notifications, footer: a severity tints only the title chip (text = "text"), never the whole box
    "overlay": "#232C35",                            # toasts, notification center, footer feedback row
    "tint-success": mix(SUCCESS, SURFACE, 0.3), "tint-error": mix(ERROR, SURFACE, 0.3),
    "tint-warning": mix(WARNING, SURFACE, 0.3), "tint-information": "#3A4754",
    "confirm-row": mix(WARNING, SURFACE, 0.15),    # footer while a confirmation waits
}
TOKENS = DARK


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


def css_variables():
    """Tokens as Textual theme variables, prefixed so none shadows Textual's own: $xmr-cursor-row, ..."""
    return {f"xmr-{k}": v for k, v in TOKENS.items() if isinstance(v, str)}
