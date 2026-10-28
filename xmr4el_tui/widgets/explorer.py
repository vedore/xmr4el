"""Explorer widgets: a Tree of explorer.build() nodes, resizable by dragging its right border. Explorer is the
Dashboard's; ResultsExplorer is the same widget with comparison row states."""
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.strip import Strip
from textual.widgets import Tree

from xmr4el_tui import explorer
from xmr4el_tui.fmt import STATE_SHORT, STATE_STYLE
from xmr4el_tui.services.comparison import CAUTION, INCOMPATIBLE, beam, beam_label
from xmr4el_tui.theme import beam_color, color, style
from xmr4el_tui.widgets.resize import DragRightBorder, column_width
from xmr4el_tui.widgets.row_rendering import FullRow, Selectable, paint

INFO, CAUTION_MARK, INCOMPATIBLE_MARK = "ⓘ", "⚠", "✗"  # Results: inspect (mouse), Caution, Incompatible

EXPLORER_SHARE, EXPLORER_MIN, EXPLORER_MAX_SHARE, RIGHT_MIN, EXPLORER_STEP = 0.22, 18, 0.5, 40, 2


def explorer_width(want, total):
    """Explorer cells on a `total`-cell screen: `want` (None: EXPLORER_SHARE of it), at least EXPLORER_MIN, at most
    half the screen and never leaving the right panels under RIGHT_MIN (the minimum wins on tiny screens)."""
    return column_width(want, total, EXPLORER_SHARE, EXPLORER_MIN, EXPLORER_MAX_SHARE, RIGHT_MIN)


def node_label(node):
    t = Text(node.label, style="bold" if node.kind == "group" else "")
    if node.kind in ("group", "dataset"):
        t.append(f"  {len(node.children)}", style="dim")
    if node.state:
        t.append("  ")
        t.append(STATE_SHORT[node.state], style=STATE_STYLE[node.state])
    if node.note:
        t.append(f"  {node.note}", style=style("warning") if "missing" in node.note else "dim")
    return t


class Explorer(DragRightBorder, Selectable, FullRow, Tree):
    """Keyboard tree of explorer.build() nodes; node data = node key. Expanded keys and the cursor survive
    rebuilds; a cursor whose item vanished lands on its nearest surviving ancestor. Dragging its right border
    resizes it."""
    CURSOR = "tree--cursor"
    lit = None  # class default: Tree.__init__ already runs the cursor watcher
    BINDINGS = [Binding("tab", "cursor_down", "next", show=False),
                Binding("shift+tab", "cursor_up", "previous", show=False),
                Binding("left", "collapse_or_parent", "collapse", show=False),
                Binding("right", "expand", "expand", show=False),
                Binding("enter", "screen.describe", "describe")]

    def __init__(self, **kwargs):
        super().__init__("outputs", **kwargs)
        self.show_root, self.auto_expand = False, False
        self.expanded, self.seen, self.sig, self.index, self.nodes = set(), set(), None, {}, {}

    def cursor_at(self, y):
        return self.cursor_line >= 0 and y + self.scroll_offset.y == self.cursor_line

    def watch_cursor_line(self, previous_line, line):
        super().watch_cursor_line(previous_line, line)
        self.light()

    def light(self):
        """Accent guides on the cursor's branch: Tree lights only the cursor node's own children, so on a leaf (or a
        collapsed node) its parent's branch is lit too, keeping the line from the tree / dataset down to it."""
        if self.lit is not None and self.lit is not self.cursor_node:
            self.lit._selected = False
        node = self.cursor_node
        self.lit = node.parent if node is not None and not node.is_expanded and node.parent is not self.root \
            else None
        if self.lit is not None:
            self.lit._selected = True
        self._line_cache.clear()
        self.refresh()

    def selected_key(self):
        return self.cursor_node.data if self.cursor_node else None

    def label(self, node):
        """Row text of explorer.Node `node`."""
        return node_label(node)

    def fill(self, roots):
        """Show `roots`; returns the selected key that disappeared, or None."""
        keep = self.selected_key()
        path = (*self.index[keep][1], keep) if keep in self.index else ()
        self.index = explorer.index(roots)
        for key, (node, _) in self.index.items():
            if key not in self.seen and node.kind in ("group", "dataset"):
                self.expanded.add(key)
            self.seen.add(key)
        sig = explorer.signature(roots)
        if sig == self.sig:
            return None
        self.sig, y = sig, self.scroll_y
        self.clear()
        self.forget_lines()
        nodes = self.nodes = {}

        def add(parent, node):
            if node.children:
                nodes[node.key] = parent.add(self.label(node), data=node.key, expand=node.key in self.expanded)
                for c in node.children:
                    add(nodes[node.key], c)
            else:
                nodes[node.key] = parent.add_leaf(self.label(node), data=node.key)
        for r in roots:
            add(self.root, r)
        if self.last_line >= 0:  # builds the lines, so node.line is current
            self.move_cursor(next((nodes[k] for k in reversed(path) if k in nodes and nodes[k].line >= 0),
                                  self.get_node_at_line(0)))
        self.scroll_to(y=y, animate=False)
        return keep if keep is not None and keep not in self.index else None

    def on_tree_node_expanded(self, event):
        self.expanded.add(event.node.data)
        self.light()

    def on_tree_node_collapsed(self, event):
        self.expanded.discard(event.node.data)
        self.light()

    def action_collapse_or_parent(self):
        node = self.cursor_node
        if node is not None and node.allow_expand and node.is_expanded:
            node.collapse()
        elif node is not None and node.parent is not None and node.parent is not self.root:
            self.move_cursor(node.parent)

    def action_expand(self):
        node = self.cursor_node
        if node is not None and node.allow_expand and not node.is_expanded:
            node.expand()


class ResultsExplorer(Explorer):
    """The Dashboard's explorer on the Results screen: datasets > trees > evaluations, rows painted by state.
    Cursor row: `cursor-row` background. Comparison member (baseline included): `member-row` background, kept when
    the cursor leaves; the cursor on a member adds `focus-outline` bars at both edges. space adds / removes the
    evaluation, enter inspects it; a click on an evaluation adds / removes it, a click on its ⓘ inspects it, a click
    on a tree or dataset expands / collapses it. The screen sets `marks` and `compat` and calls `restyle`."""
    BINDINGS = [Binding("space", "screen.toggle_member", "add / remove"),
                Binding("enter", "screen.inspect", "inspect")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_expand = True  # a click (select) on a tree / dataset toggles it; enter is inspect
        self.marks, self.compat = {}, {}  # key -> "base" | "member", key -> Compat

    def label(self, node):
        return results_label(node, self.marks.get(node.key), self.compat)

    def restyle(self):
        """Relabel every row from marks / compat (no rebuild: cursor, expansion and scroll stay)."""
        for key, tree_node in self.nodes.items():
            tree_node.set_label(self.label(self.index[key][0]))
        self.refresh()

    def render_label(self, node, base_style, style):
        """The label as built: row colours are painted over the whole row by paint_row, so Tree's cursor style must
        not recolour the beam chip and caution symbol."""
        return super().render_label(node, base_style, Style())

    def paint_row(self, strip, y):
        node = self.get_node_at_line(self.scroll_offset.y + y)
        mark, cursor = self.marks.get(node.data) if node else None, self.cursor_at(y)
        if mark:
            strip = paint(strip, Style(bgcolor=color("baseline-row" if mark == "base" else "member-row")))
            if cursor and strip.cell_length > 2:
                width = strip.cell_length
                strip = Strip.join([edge(strip.crop(0, 1), "▌"), strip.crop(1, width - 1),
                                    edge(strip.crop(width - 1, width), "▐")])
            return strip
        return paint(strip, Style(bgcolor=color("cursor-row"))) if cursor else strip

    async def _on_click(self, event):
        """A click on ⓘ inspects (Tree's click, which would add / remove, is skipped)."""
        meta = event.style.meta
        if meta.get("inspect") and "line" in meta:
            event.prevent_default()
            self.cursor_line = meta["line"]
            self.screen.action_inspect()


def edge(cell, glyph):
    """One-cell strip `cell` redrawn as `glyph` in the focus-outline colour (its click metadata kept)."""
    seg = next(iter(cell), None)
    base = seg.style if seg and seg.style else Style()
    return Strip([Segment(glyph, base + Style(color=color("focus-outline"), bold=True))])


def chip(beam):
    """Beam marker: `B10` on the beam's palette colour (theme.beam_color), the number always shown."""
    return Text(beam_label(beam), style=Style(color=color("on-beam"), bgcolor=beam_color(beam), bold=True))


def results_label(node, mark, compat):
    """Row text of a Results explorer row; `mark`: "base" / "member" / None."""
    fg = color("on-selected") if mark else color("text")
    t = Text(style=Style(color=fg))
    if node.kind != "eval":
        t.append(node.label, style="bold" if node.kind == "dataset" else "")
        if node.kind == "dataset":
            t.append(f"  {len(node.children)}", style="dim")
        if any(compat.get(c.key) and compat[c.key].level == CAUTION for c in node.children):
            t.append(f"  {CAUTION_MARK}", style=style("warning"))  # some evaluation below is Caution
        return t
    c = compat.get(node.key)
    t.append(INFO, style=Style(meta={"inspect": True}))
    t.append(" ")
    t.append_text(chip(beam(node.ref[1])))
    t.append(" ")
    incompatible = c is not None and c.level == INCOMPATIBLE
    t.append(node.label, style=Style(color=color("text-disabled")) if incompatible and not mark else "")
    if mark == "base":
        t.append("  baseline", style="bold")
    elif c is not None and c.level == CAUTION:
        t.append(f"  {CAUTION_MARK}", style=Style(color=color("warning-on-accent" if mark else "warning"), bold=True))
    elif incompatible:
        t.append(f"  {INCOMPATIBLE_MARK}", style=Style(color=color("text-disabled")))
    if not node.ref[1].metrics and node.state:
        t.append(f"  {STATE_SHORT[node.state]}", style="" if mark else STATE_STYLE[node.state])
    return t
