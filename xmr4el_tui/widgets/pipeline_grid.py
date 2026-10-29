"""Model Builder widgets (tui.md §5.14): the horizontal pipeline grid and the tree tab strip."""
from dataclasses import dataclass

from rich.text import Text
from textual.binding import Binding
from textual.geometry import Region
from textual.widget import Widget

from xmr4el_tui import components, theme

# Feature construction: three parallel branches, then the fixed PIFA step and the single-slot stages.
BRANCHES = (("Transformer", ("transformer",)), ("Mention", ("char_tfidf", "mention_svd")),
            ("Context", ("context_tfidf", "context_svd")))
RIGHT = ("hierarchy", "matcher", "search")
LEVELS = ("compact", "normal", "expanded")


@dataclass(frozen=True)
class Geo:
    """Cell geometry of one zoom level: branch label column, block width, connector, PIFA width, tallest block, rows
    between branches. Zoom changes only this density, never the configuration."""
    label: int
    w: int
    c: int
    p: int
    hmax: int
    gap: int

    @property
    def x0(self):
        return self.label

    @property
    def x1(self):
        return self.x0 + self.w + self.c

    @property
    def xj(self):
        return self.x1 + self.w + 1     # where the three branches join

    @property
    def xp(self):
        return self.xj + self.c + 1     # PIFA

    @property
    def xr(self):
        return [self.xp + self.p + self.c + i * (self.w + self.c) for i in range(len(RIGHT))]

    @property
    def width(self):
        return self.xr[-1] + self.w + 1


GEOS = {"compact": Geo(12, 18, 1, 6, 2, 0), "normal": Geo(12, 20, 2, 10, 3, 1), "expanded": Geo(12, 30, 4, 10, 3, 2)}


def neighbor(sid, dx, dy, last_feature):
    """Slot reached from `sid` by an arrow: left / right follow a branch (a branch's end goes on to Hierarchy;
    Hierarchy goes back to `last_feature`), up / down change branch inside Feature construction. Never a missing
    neighbour: an arrow with nothing there keeps `sid`."""
    rows = [slots for _, slots in BRANCHES]
    for b, row in enumerate(rows):
        if sid in row:
            i = row.index(sid)
            if dy:
                return rows[b + dy][min(i, len(rows[b + dy]) - 1)] if 0 <= b + dy < len(rows) else sid
            if dx < 0:
                return row[i - 1] if i else sid
            return row[i + 1] if i + 1 < len(row) else RIGHT[0]
    i = RIGHT.index(sid)
    if dy:
        return sid
    if dx < 0:
        return RIGHT[i - 1] if i else last_feature
    return RIGHT[i + 1] if i + 1 < len(RIGHT) else sid


def fit(height, geo):
    """(block height, gap between branches, heading row) for `height` rows at `geo`'s density; the smallest layout
    scrolls."""
    for h, gap, head in ((3, 2, 1), (3, 1, 1), (3, 0, 1), (3, 0, 0), (2, 0, 1), (2, 0, 0), (1, 1, 1), (1, 1, 0),
                         (1, 0, 0)):
        if h <= geo.hmax and gap <= geo.gap + (h == 1) and 3 * h + 2 * gap + head <= height:
            return h, gap, head
    return 1, 0, 0


class Canvas:
    def __init__(self, width, height):
        self.cells = [[(" ", "")] * width for _ in range(height)]

    def put(self, x, y, text, style=""):
        row = self.cells[y]
        for i, ch in enumerate(text):
            if 0 <= x + i < len(row):
                row[x + i] = (ch, style)

    def text(self):
        out = Text(no_wrap=True, overflow="crop")
        for y, row in enumerate(self.cells):
            if y:
                out.append("\n")
            run, style = "", None
            for ch, st in row:
                if st != style and run:
                    out.append(run, style)
                    run = ""
                run, style = run + ch, st
            out.append(run, style)
        return out


def cut(text, width):
    return text if len(text) <= width else text[:width - 1] + "…"


class PipelineGrid(Widget, can_focus=True):
    """The tree's pipeline: one block per slot (name, installed component, short settings), the fixed PIFA step,
    connectors. Arrows move the cursor (`neighbor`), enter opens the slot's components, a click selects a slot,
    + / - / 0 zoom (density only: compact, normal, expanded, or fit to the viewport). The model is centred in the
    scroll box when it fits and scrolls when it does not. The screen owns the model state; this draws `config`
    with the cursor on `cursor`."""
    BINDINGS = [Binding("enter", "screen.library", "components"),
                Binding("left", "move(-1, 0)", show=False), Binding("right", "move(1, 0)", show=False),
                Binding("up", "move(0, -1)", show=False), Binding("down", "move(0, 1)", show=False),
                Binding("plus,equals_sign", "zoom(1)", "zoom", key_display="+/-"),
                Binding("minus", "zoom(-1)", show=False), Binding("0", "zoom_fit", "fit"),
                Binding("1", "screen.tree(0)", "tree", key_display="1-9")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config, self.cursor, self.problems = {}, "transformer", {}
        self.regions = {}  # slot id -> Region on the grid
        self.layout_ = (1, 0, 0)
        self.zoom = "normal"  # a level or "fit": UI state of this widget, never part of the model
        self.level = "normal"  # the level drawn (the fitted one under "fit")
        self.margin_ = (0, 0)  # (left, top) that centre the model in the scroll box

    @property
    def geo(self):
        return GEOS[self.level]

    def show(self, config, cursor):
        self.config, self.cursor = config, cursor
        self.problems = components.problems(config)
        self.relayout()
        self.refresh()
        self.call_after_refresh(self.reveal)

    def relayout(self):
        """Level, block height and centring from the room the scroll box gives (width first: a horizontal scrollbar
        takes a row of the height)."""
        box = self.parent.scrollable_content_region if self.parent else None
        vw, vh = (box.width, box.height) if box else (0, 0)
        self.level = next((n for n in reversed(LEVELS) if GEOS[n].width <= vw), "compact") if self.zoom == "fit" \
            else self.zoom
        g = self.geo
        if g.width > vw and self.parent is not None and not self.parent.show_horizontal_scrollbar:
            vh -= 1
        self.layout_ = h, gap, head = fit(vh, g)
        height = head + 3 * h + 2 * gap
        self.margin_ = (max(0, (vw - g.width) // 2), max(0, (vh - height) // 2))
        self.styles.width, self.styles.height = g.width, height
        self.styles.margin = (self.margin_[1], 0, 0, self.margin_[0])
        for bar in self.screen.query(ZoomBar):
            bar.show(self.zoom, self.level)

    def set_zoom(self, zoom):
        self.zoom = zoom
        self.relayout()
        self.refresh()
        self.call_after_refresh(self.reveal)

    def action_zoom(self, step):
        i = LEVELS.index(self.level) + step
        if 0 <= i < len(LEVELS):
            self.set_zoom(LEVELS[i])

    def action_zoom_fit(self):
        self.set_zoom("fit")

    def reveal(self):
        region = self.regions.get(self.cursor)
        if region is not None and self.parent is not None:
            g, (left, top) = self.geo, self.margin_
            self.parent.scroll_to_region(region.grow((0, g.c, 0, g.label if region.x == g.x0 else g.c))
                                         .translate((left, top)), animate=False)

    def on_focus(self):
        self.refresh()

    def on_blur(self):
        self.refresh()

    def action_move(self, dx, dy):
        self.screen.select_slot(neighbor(self.cursor, dx, dy, self.screen.draft.feature_slot))

    def on_click(self, event):
        hit = next((sid for sid, r in self.regions.items() if r.contains(event.x, event.y)), None)
        if hit is not None:
            self.focus()
            self.screen.select_slot(hit)

    def render(self):
        g = self.geo
        h, gap, head = self.layout_
        canvas = Canvas(g.width, head + 3 * h + 2 * gap)
        line, muted = theme.style("panel-border"), theme.style("text-muted")
        if head:
            canvas.put(g.x0, 0, "Feature construction", theme.style("text-muted", "bold"))
        tops = [head + b * (h + gap) for b in range(3)]
        mids = [t + h // 2 for t in tops]
        self.regions = {}
        for b, (label, slots) in enumerate(BRANCHES):
            canvas.put(0, mids[b], label, muted)
            for i, sid in enumerate(slots):
                self.block(canvas, g.x0 if i == 0 else g.x1, tops[b], sid)
            end = g.x0 + g.w if len(slots) == 1 else g.x1 + g.w
            if len(slots) == 2:
                canvas.put(g.x0 + g.w, mids[b], "─" * g.c, line)
            canvas.put(end, mids[b], "─" * (g.xj - end), line)
        for y in range(mids[0], mids[2] + 1):
            canvas.put(g.xj, y, {mids[0]: "┐", mids[1]: "┼", mids[2]: "┘"}.get(y, "│"), line)
        canvas.put(g.xj + 1, mids[1], "─" * (g.xp - g.xj - 1), line)
        pifa = ["PIFA", "labels", "fixed"] if h >= 3 else ["PIFA", "fixed"][:h]
        for i in range(h):
            canvas.put(g.xp, tops[1] + i, f" {pifa[i] if i < len(pifa) else '':<{g.p - 1}}",
                       f"italic {theme.color('text-muted')} on {theme.color('background')}")
        canvas.put(g.xp + g.p, mids[1], "─" * g.c, line)
        for i, sid in enumerate(RIGHT):
            self.block(canvas, g.xr[i], tops[1], sid)
            if i + 1 < len(RIGHT):
                canvas.put(g.xr[i] + g.w, mids[1], "─" * g.c, line)
        return canvas.text()

    def block(self, canvas, x, y, sid):
        h, w = self.layout_[0], self.geo.w
        slot = components.slot(sid)
        bad = sid in self.problems
        comp = components.component_label(self.config, slot)
        summ = "needs attention" if bad else components.summary(self.config, slot)
        lines = [slot.name] if h == 1 else [slot.name, f"{comp} · {summ}" if summ else comp] if h == 2 else \
            [slot.name, comp, summ]
        if sid == self.cursor and self.has_focus:
            base = f"{theme.color('on-accent')} on {theme.color('accent')}"
        elif sid == self.cursor:
            base = f"{theme.color('text')} on {theme.color('cursor-blurred')}"
        else:
            base = f"{theme.color('text')} on {theme.color('block')}"
        for i in range(h):
            text = f" {cut(lines[i], w - 2) if i < len(lines) else '':<{w - 1}}"
            if sid != self.cursor and bad and i == len(lines) - 1:  # invalid: its last line in the error colour
                style = f"{theme.color('error')} on {theme.color('block')}"
            elif sid != self.cursor and i:
                style = f"{theme.color('text-muted')} on {theme.color('block')}"
            else:
                style = base
            style += " bold" if i == 0 else ""
            canvas.put(x, y + i, text, style)
        self.regions[sid] = Region(x, y, w, h)


class ZoomBar(Widget):
    """` -  Normal  +  Fit ` beside the tab strip: clicks zoom the grid; the keys + - 0 do the same on the grid."""
    DEFAULT_CSS = "ZoomBar { width: 20; height: 1; }"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zoom, self.level, self.spans = "normal", "normal", []

    def show(self, zoom, level):
        self.zoom, self.level = zoom, level
        self.refresh()

    def render(self):
        accent, muted = theme.style("accent", "bold"), theme.style("text-muted")
        fit = self.zoom == "fit"
        parts = [(" - ", accent, "out"), (f" {self.level.capitalize():<8} ", muted, None), (" + ", accent, "in"),
                 (" Fit ", f"{theme.color('on-accent')} on {theme.color('accent')} bold" if fit else accent, "fit")]
        out, x, self.spans = Text(no_wrap=True), 0, []
        for label, style, ref in parts:
            out.append(label, style)
            self.spans.append((x, x + len(label), ref))
            x += len(label)
        return out

    def on_click(self, event):
        hit = next((ref for x0, x1, ref in self.spans if x0 <= event.x < x1 and ref), None)
        grid = self.screen.query_one(PipelineGrid)
        if hit == "fit":
            grid.action_zoom_fit()
        elif hit:
            grid.action_zoom(1 if hit == "in" else -1)


class TreeTabs(Widget, can_focus=True):
    """The batch's trees in execution order (`[1] Tree 1`, ...), then `Add tree`. This is Tree Selection: left / right
    move the highlight only (the shown tree and every draft stay), enter shows the highlighted tree and returns to
    the grid (on `Add tree`: creates a tree and opens it), 1-9 jump to a tree, x removes the highlighted tree (asks), shift+left / right
    reorder it, F2 renames. A click on a tab shows it. Tabs after the ninth have no number: arrows reach them."""
    BINDINGS = [Binding("left", "cursor(-1)", "navigate", key_display="←→"),
                Binding("right", "cursor(1)", show=False),
                Binding("enter", "activate", "select"),
                Binding("1", "screen.tree(0)", "tree", key_display="1-9"),
                Binding("x", "screen.remove_tree", "remove"),
                Binding("shift+left", "screen.move_tree(-1)", show=False),
                Binding("shift+right", "screen.move_tree(1)", show=False),
                Binding("n", "screen.add_tree", show=False), Binding("d", "screen.duplicate_tree", show=False),
                Binding("delete", "screen.remove_tree", show=False),
                Binding("y", "screen.confirm_remove", show=False)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spans = []  # [(x0, x1, tree id or "add")]
        self.trees, self.current, self.cursor, self.bad = [], None, None, set()

    def show(self, trees, current):
        self.trees, self.current = list(trees), current
        if self.cursor not in {t.id for t in trees} | {"add"}:
            self.cursor = current
        self.bad = {t.id for t in trees if components.problems(t.config)}
        self.refresh()

    def action_cursor(self, step):
        ids = [t.id for t in self.trees] + ["add"]
        j = ids.index(self.cursor) + step if self.cursor in ids else 0
        if 0 <= j < len(ids):
            self.cursor = ids[j]
            self.refresh()

    def action_activate(self):
        if self.cursor == "add":
            self.screen.action_add_tree()
        else:
            self.screen.activate_tree(self.cursor)

    def on_focus(self):
        self.refresh()

    def on_blur(self):
        if self.screen.renaming is None:
            self.cursor = self.current  # leaving Tree Selection drops the highlight back on the shown tree
        self.refresh()

    def render(self):
        width = self.size.width or 80
        accent, muted = theme.color("accent"), theme.color("text-muted")
        parts = []
        for i, t in enumerate(self.trees):
            num = f"[{i + 1}]" if i < 9 else ""
            name = f" {t.name}{' !' if t.id in self.bad else ''} "
            if self.has_focus and t.id == self.cursor:
                num_style = name_style = f"{theme.color('on-accent')} on {accent} bold"
            elif t.id == self.current:
                num_style = f"{accent} on {theme.color('cursor-blurred')} bold"
                name_style = f"{theme.color('text')} on {theme.color('cursor-blurred')} bold"
            else:
                num_style = f"{accent} on {theme.color('block')}"
                name_style = f"{muted} on {theme.color('block')}"
            parts.append(([(" " + num if num else " ", num_style), (name, name_style)], t.id))
        add = f"{theme.color('on-accent')} on {accent} bold" if self.has_focus and self.cursor == "add" \
            else theme.style("accent")
        parts.append(([(" Add tree ", add)], "add"))
        here = next((i for i, p in enumerate(parts) if p[1] == self.cursor), 0)
        size = lambda p: sum(len(x) for x, _ in p[0]) + 1  # noqa: E731
        start = 0  # drop tabs from the left until the highlighted one fits
        while start < here and sum(size(p) for p in parts[start:here + 1]) + 2 > width:
            start += 1
        out, x, self.spans = Text(no_wrap=True, overflow="ellipsis"), 0, []
        if start:
            out.append("… ", theme.style("text-muted"))
            x = 2
        for segments, ref in parts[start:]:
            n = 0
            for text, style in segments:
                out.append(text, style)
                n += len(text)
            self.spans.append((x, x + n, ref))
            out.append(" ")
            x += n + 1
        return out

    def on_click(self, event):
        hit = next((ref for x0, x1, ref in self.spans if x0 <= event.x < x1), None)
        if hit == "add":
            self.cursor = "add"
            self.screen.action_add_tree()
        elif hit is not None:
            self.cursor = hit
            self.screen.switch_to(hit)
