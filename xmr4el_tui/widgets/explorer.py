"""Explorer widgets: a Tree of explorer.build() nodes, resizable by dragging its right border. Explorer is the
Dashboard's; ResultsExplorer is the same widget with comparison row states. Both draw a tree on two rows (title,
then date and short id), leave a gap row above each tree / dataset / group but the first, and search with `/`."""
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widgets import Tree

from xmr4el_tui import explorer
from xmr4el_tui.fmt import STATE_SHORT, STATE_STYLE
from xmr4el_tui.services.comparison import CAUTION, INCOMPATIBLE, beam, beam_label
from xmr4el_tui.theme import beam_color, color, style
from xmr4el_tui.widgets.resize import DragRightBorder, column_width
from xmr4el_tui.widgets.row_rendering import FullRow, Selectable, paint

INFO, CAUTION_MARK, INCOMPATIBLE_MARK = "i", "!", "x"  # Results: inspect (mouse), Caution, Incompatible
GAP_KINDS = ("group", "dataset", "tree")  # a gap row above each one that has a previous sibling

EXPLORER_SHARE, EXPLORER_MIN, EXPLORER_MAX_SHARE, RIGHT_MIN, EXPLORER_STEP = 0.22, 18, 0.5, 40, 2


def explorer_width(want, total):
    """Explorer cells on a `total`-cell screen: `want` (None: EXPLORER_SHARE of it), at least EXPLORER_MIN, at most
    half the screen and never leaving the right panels under RIGHT_MIN (the minimum wins on tiny screens)."""
    return column_width(want, total, EXPLORER_SHARE, EXPLORER_MIN, EXPLORER_MAX_SHARE, RIGHT_MIN)


def node_label(node):
    t = Text(explorer.tree_title(node) if node.kind == "tree" else node.label,
             style="bold" if node.kind == "group" else "")
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
    resizes it.

    Rows: Tree has one screen row per node; here a node takes one or two (`detail`), with a gap row above some
    (GAP_KINDS). `rows` maps each screen row (virtual y) to (Tree line, 0 label / 1 detail / -1 gap); every row of
    an item carries its line in the click metadata, so the whole item is one hit area.
    Search: `/` types a query (shown in the border subtitle), enter keeps it, esc clears it; nodes outside it are
    hidden (explorer.prune), their ancestors opened only while it lasts, so the expansion the user chose returns."""
    CURSOR = "tree--cursor"
    OPEN_KINDS = ("group", "dataset")  # expanded when first seen
    lit = None  # class defaults: Tree.__init__ already runs the cursor watcher
    rows, first, query, typing, index = (), (), "", False, {}
    BINDINGS = [Binding("tab", "cursor_down", "next", show=False),
                Binding("shift+tab", "cursor_up", "previous", show=False),
                Binding("left", "collapse_or_parent", "collapse", show=False),
                Binding("right", "expand", "expand", show=False),
                Binding("slash", "search", "search"),
                Binding("enter", "screen.describe", "describe")]

    def __init__(self, **kwargs):
        super().__init__("outputs", **kwargs)
        self.show_root, self.auto_expand = False, False
        self.expanded, self.seen, self.sig, self.index, self.nodes = set(), set(), None, {}, {}
        self.source = ()

    # --- rows -------------------------------------------------------------------------------------------------
    def _build(self):
        super()._build()
        rows, first = [], []
        for i, line in enumerate(self._tree_lines_cached):
            node = self.index.get(line.node.data, (None,))[0]
            if node is not None and node.kind in GAP_KINDS and line.node.previous_sibling is not None:
                rows.append((i, -1))
            first.append(len(rows))
            rows.append((i, 0))
            if node is not None and node.detail:
                rows.append((i, 1))
        self.rows, self.first = rows, first
        self.virtual_size = Size(self.virtual_size.width, len(rows))

    def item_at(self, y):
        """Tree line of screen row `y` (on screen), -1 on a gap row or below the last item."""
        row = y + self.scroll_offset.y
        return self.rows[row][0] if row < len(self.rows) and self.rows[row][1] >= 0 else -1

    def row_of(self, tree_node):
        """Virtual y of the first row of `tree_node` (shown)."""
        self._tree_lines  # noqa: B018 - builds rows
        return self.first[tree_node.line]

    def cursor_at(self, y):
        return self.cursor_line >= 0 and self.item_at(y) == self.cursor_line

    def _render_line(self, y, x1, x2, base_style):
        lines = self._tree_lines  # builds rows
        if y >= len(self.rows):
            return Strip.blank(self.size.width, base_style)
        line, sub = self.rows[y]
        if sub == 0:
            return super()._render_line(line, x1, x2, base_style)
        return self.extra_row(lines[line], line, sub, base_style).crop(x1, x2)

    def extra_row(self, tree_line, line, sub, base_style):
        """Row `sub` of a Tree line that is not its label: 1 = the detail under the label, -1 = the gap above it.
        Guides continue through both (lit like Tree lights the label row's); the gap is not part of the item."""
        path = tree_line.path
        base = self.get_component_rich_style("tree--guides", partial=True)
        lit = base + self.get_component_rich_style("tree--guides-selected", partial=True)
        depth = max(0, self.guide_depth - 2)
        space, vertical = (f"{c[0]}{c[1] * depth} " for c in self.LINES["default"][:2])
        row_style = base_style + Style(meta={"line": line}) if sub > 0 else base_style
        t = Text(style=row_style)
        on = path[0]._selected and self.has_focus
        for node in path[1:]:
            t.append(vertical if (node is path[-1] and sub < 0) or not node.is_last else space,
                     style=lit if on else base)
            on = (on or node._selected) and self.has_focus
        if sub > 0:
            t.append("  ")  # under the label, past the expand icon
            t.append_text(self.detail(self.index[tree_line.node.data][0]))
        strip = Strip(list(t.render(self.app.console)))
        return strip.extend_cell_length(max(self.virtual_size.width, self.size.width), row_style)

    def _get_label_region(self, line):
        region = super()._get_label_region(line)  # builds rows
        if region is None:
            return None
        y = self.first[line]
        return Region(region.x, y, region.width, 2 if (line, 1) in self.rows[y + 1:y + 2] else 1)

    def _refresh_line(self, line):
        self.refresh()  # rows are not lines: repaint the (small) explorer

    def _refresh_node(self, node):
        self.refresh()

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

    def detail(self, node):
        """Second row of explorer.Node `node` (a tree's date and short id)."""
        return Text(node.detail, style=style("text-muted"))

    def fill(self, roots):
        """Show `roots` (those the search accepts); returns the selected key that is no longer in `roots`, or None."""
        keep = self.selected_key()
        path = (*self.index[keep][1], keep) if keep in self.index else ()
        self.source, full = roots, explorer.index(roots)
        shown, opened = explorer.prune(roots, explorer.matches(self.query)) if self.query.strip() else (roots, set())
        for key, (node, _) in full.items():
            if key not in self.seen and node.kind in self.OPEN_KINDS:
                self.expanded.add(key)
            self.seen.add(key)
        self.index = explorer.index(shown)
        sig = (explorer.signature(shown), frozenset(opened))
        if sig == self.sig:
            return None
        self.sig, y = sig, self.scroll_y
        self.clear()
        self.forget_lines()
        nodes = self.nodes = {}

        def add(parent, node):
            if node.children:
                nodes[node.key] = parent.add(self.label(node), data=node.key,
                                             expand=node.key in self.expanded or node.key in opened)
                for c in node.children:
                    add(nodes[node.key], c)
            else:
                nodes[node.key] = parent.add_leaf(self.label(node), data=node.key)
        for r in shown:  # mounted nodes = self.index
            add(self.root, r)
        if self.last_line >= 0:  # builds the lines, so node.line is current
            self.move_cursor(next((nodes[k] for k in reversed(path) if k in nodes and nodes[k].line >= 0),
                                  self.get_node_at_line(0)))
        self.scroll_to(y=y, animate=False)
        return keep if keep is not None and keep not in full else None

    # --- search -----------------------------------------------------------------------------------------------
    def action_search(self):
        self.typing = True
        self.show_query()

    def set_query(self, query, typing):
        self.query, self.typing = query, typing
        self.fill(self.source)
        self.show_query()

    def show_query(self):
        shown = f"/{self.query}{'▏' if self.typing else ''}"
        found = "" if not self.query.strip() or self.index else "  no match"
        self.border_subtitle = shown + found if self.typing or self.query else ""

    def on_key(self, event):
        """While typing: characters, backspace, enter (keep the query), esc (clear it); arrows still move. With a
        query kept, esc clears it before the screen's esc runs."""
        if not (self.typing or self.query):
            return
        key = event.key
        if key == "escape":
            self.set_query("", False)
        elif not self.typing:
            return
        elif key == "enter":
            self.typing = False
            self.show_query()
        elif key == "backspace":
            self.set_query(self.query[:-1], True)
        elif event.is_printable and event.character:
            self.set_query(self.query + event.character, True)
        else:
            return
        event.stop()
        event.prevent_default()

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
    evaluation, enter inspects it; a click on an evaluation adds / removes it, a click on its i inspects it, a click
    on a tree or dataset expands / collapses it. The screen sets `marks` and `compat` and calls `restyle`."""
    OPEN_KINDS = ("group", "dataset", "tree")  # candidates in view at once
    BINDINGS = [Binding("space", "screen.toggle_member", "add / remove"),
                Binding("enter", "screen.inspect", "inspect")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_expand = True  # a click (select) on a tree / dataset toggles it; enter is inspect
        self.marks, self.compat = {}, {}  # key -> "base" | member's role number ("1"), key -> Compat

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
        line = self.item_at(y)
        node = self.get_node_at_line(line) if line >= 0 else None
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
        """A click on i inspects (Tree's click, which would add / remove, is skipped)."""
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
    """Row text of a Results explorer row; `mark`: "base" / a member's role number (its Summary row and Analysis
    column, which a cut name cannot tell apart) / None."""
    fg = color("on-selected") if mark else color("text")
    t = Text(style=Style(color=fg))
    if node.kind != "eval":
        t.append(explorer.tree_title(node) if node.kind == "tree" else node.label,
                 style="bold" if node.kind == "dataset" else "")
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
    if mark and mark != "base":
        t.append(f"{mark} ", style="bold")  # before the name: a narrow explorer cuts the row's end
    incompatible = c is not None and c.level == INCOMPATIBLE
    t.append(node.label, style=Style(color=color("text-disabled")) if incompatible and not mark else "")
    if mark == "base":
        t.append("  baseline", style="bold")
    if mark != "base" and c is not None and c.level == CAUTION:
        t.append(f"  {CAUTION_MARK}", style=Style(color=color("warning-on-accent" if mark else "warning"), bold=True))
    elif incompatible:
        t.append(f"  {INCOMPATIBLE_MARK}", style=Style(color=color("text-disabled")))
    if not node.ref[1].metrics and node.state:
        t.append(f"  {STATE_SHORT[node.state]}", style="" if mark else STATE_STYLE[node.state])
    return t
