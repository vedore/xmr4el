"""Model Builder widgets (tui.md §5.14): the horizontal pipeline grid and the tree tab strip."""
from rich.text import Text
from textual.binding import Binding
from textual.geometry import Region
from textual.widget import Widget

from xmr4el_tui import components, theme

# Feature construction: three parallel branches, then the fixed PIFA step and the single-slot stages.
BRANCHES = (("Transformer", ("transformer",)), ("Mention", ("char_tfidf", "mention_svd")),
            ("Context", ("context_tfidf", "context_svd")))
RIGHT = ("hierarchy", "matcher", "search")
LABEL, W, C, P = 12, 20, 2, 10  # branch label column, block width, connector, PIFA block width
X0 = LABEL
X1 = X0 + W + C
XJ = X1 + W + 1           # where the three branches join
XP = XJ + C + 1           # PIFA
XR = [XP + P + C + i * (W + C) for i in range(len(RIGHT))]
WIDTH = XR[-1] + W + 1


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


def fit(height):
    """(block height, gap between branches, heading row) for `height` rows; the smallest layout scrolls."""
    for h, gap, head in ((3, 1, 1), (3, 0, 1), (3, 0, 0), (2, 0, 1), (2, 0, 0), (1, 1, 1), (1, 1, 0), (1, 0, 0)):
        if 3 * h + 2 * gap + head <= height:
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
    connectors. Arrows move the cursor (`neighbor`), enter opens the slot's components, a click selects a slot. The
    screen owns the state; this draws `config` with the cursor on `cursor`."""
    BINDINGS = [Binding("enter", "screen.library", "components"),
                Binding("left", "move(-1, 0)", show=False), Binding("right", "move(1, 0)", show=False),
                Binding("up", "move(0, -1)", show=False), Binding("down", "move(0, 1)", show=False)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config, self.cursor, self.problems = {}, "transformer", {}
        self.regions = {}  # slot id -> Region on the grid
        self.layout_ = (1, 0, 0)

    def show(self, config, cursor):
        self.config, self.cursor = config, cursor
        self.problems = components.problems(config)
        self.relayout()
        self.refresh()
        self.call_after_refresh(self.reveal)

    def relayout(self):
        """Block height from the room the scroll box gives; the grid scrolls when the terminal is shorter."""
        box = self.parent.scrollable_content_region.height if self.parent else 0
        self.layout_ = h, gap, head = fit(box)
        self.styles.width = WIDTH
        self.styles.height = head + 3 * h + 2 * gap

    def reveal(self):
        region = self.regions.get(self.cursor)
        if region is not None and self.parent is not None:
            self.parent.scroll_to_region(region.grow((0, C, 0, LABEL if region.x == X0 else C)), animate=False)

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
        h, gap, head = self.layout_
        canvas = Canvas(WIDTH, head + 3 * h + 2 * gap)
        line, muted = theme.style("panel-border"), theme.style("text-muted")
        if head:
            canvas.put(X0, 0, "Feature construction", theme.style("text-muted", "bold"))
        tops = [head + b * (h + gap) for b in range(3)]
        mids = [t + h // 2 for t in tops]
        self.regions = {}
        for b, (label, slots) in enumerate(BRANCHES):
            canvas.put(0, mids[b], label, muted)
            for i, sid in enumerate(slots):
                self.block(canvas, X0 if i == 0 else X1, tops[b], sid)
            end = X0 + W if len(slots) == 1 else X1 + W
            if len(slots) == 2:
                canvas.put(X0 + W, mids[b], "─" * C, line)
            canvas.put(end, mids[b], "─" * (XJ - end), line)
        for y in range(mids[0], mids[2] + 1):
            canvas.put(XJ, y, {mids[0]: "┐", mids[1]: "┼", mids[2]: "┘"}.get(y, "│"), line)
        canvas.put(XJ + 1, mids[1], "─" * (XP - XJ - 1), line)
        pifa = ["PIFA", "labels", "fixed"] if h >= 3 else ["PIFA", "fixed"][:h]
        for i in range(h):
            canvas.put(XP, tops[1] + i, f" {pifa[i] if i < len(pifa) else '':<{P - 1}}",
                       f"italic {theme.color('text-muted')} on {theme.color('background')}")
        canvas.put(XP + P, mids[1], "─" * C, line)
        for i, sid in enumerate(RIGHT):
            self.block(canvas, XR[i], tops[1], sid)
            if i + 1 < len(RIGHT):
                canvas.put(XR[i] + W, mids[1], "─" * C, line)
        return canvas.text()

    def block(self, canvas, x, y, sid):
        h = self.layout_[0]
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
            text = f" {cut(lines[i], W - 2) if i < len(lines) else '':<{W - 1}}"
            if sid != self.cursor and bad and i == len(lines) - 1:  # invalid: its last line in the error colour
                style = f"{theme.color('error')} on {theme.color('block')}"
            elif sid != self.cursor and i:
                style = f"{theme.color('text-muted')} on {theme.color('block')}"
            else:
                style = base
            style += " bold" if i == 0 else ""
            canvas.put(x, y + i, text, style)
        self.regions[sid] = Region(x, y, W, h)


class TreeTabs(Widget, can_focus=True):
    """The batch's trees in execution order (`1 Tree 1`, ...), then `Add tree`. left / right switch, shift+left /
    shift+right reorder, a adds, d duplicates, delete removes, F2 renames (screen). A click on a tab switches."""
    BINDINGS = [Binding("left", "screen.switch_tree(-1)", "tree", key_display="←→"),
                Binding("right", "screen.switch_tree(1)", show=False),
                Binding("shift+left", "screen.move_tree(-1)", "move", key_display="shift+←→"),
                Binding("shift+right", "screen.move_tree(1)", show=False),
                Binding("a", "screen.add_tree", "add"), Binding("d", "screen.duplicate_tree", "duplicate"),
                Binding("delete", "screen.remove_tree", "remove")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spans = []  # [(x0, x1, tree id or "add")]
        self.trees, self.current, self.bad = [], None, set()

    def show(self, trees, current):
        self.trees, self.current = list(trees), current
        self.bad = {t.id for t in trees if components.problems(t.config)}
        self.refresh()

    def on_focus(self):
        self.refresh()

    def on_blur(self):
        self.refresh()

    def render(self):
        width = self.size.width or 80
        parts = []
        for i, t in enumerate(self.trees):
            label = f" {i + 1} {t.name}{' !' if t.id in self.bad else ''} "
            if t.id == self.current:
                style = f"{theme.color('on-accent')} on {theme.color('accent')} bold" if self.has_focus else \
                    f"{theme.color('text')} on {theme.color('cursor-blurred')} bold"
            else:
                style = f"{theme.color('text-muted')} on {theme.color('block')}"
            parts.append((label, style, t.id))
        parts.append((" Add tree ", theme.style("accent"), "add"))
        here = next((i for i, p in enumerate(parts) if p[2] == self.current), 0)
        start = 0  # drop tabs from the left until the current one fits
        while start < here and sum(len(p[0]) + 1 for p in parts[start:here + 1]) + 2 > width:
            start += 1
        out, x, self.spans = Text(no_wrap=True, overflow="ellipsis"), 0, []
        if start:
            out.append("… ", theme.style("text-muted"))
            x = 2
        for label, style, ref in parts[start:]:
            out.append(label, style)
            self.spans.append((x, x + len(label), ref))
            out.append(" ")
            x += len(label) + 1
        return out

    def on_click(self, event):
        hit = next((ref for x0, x1, ref in self.spans if x0 <= event.x < x1), None)
        if hit == "add":
            self.screen.action_add_tree()
        elif hit is not None:
            self.screen.switch_to(hit)
