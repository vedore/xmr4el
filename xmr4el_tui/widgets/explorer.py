"""Dashboard explorer widget: a Tree of explorer.build() nodes, resizable by dragging its right border."""
from rich.text import Text
from textual.binding import Binding
from textual.widgets import Tree

from xmr4el_tui import explorer
from xmr4el_tui.fmt import STATE_SHORT, STATE_STYLE
from xmr4el_tui.widgets.row_rendering import FullRow, Selectable

EXPLORER_SHARE, EXPLORER_MIN, EXPLORER_MAX_SHARE, RIGHT_MIN, EXPLORER_STEP = 0.22, 18, 0.5, 40, 2


def explorer_width(want, total):
    """Explorer cells on a `total`-cell screen: `want` (None: EXPLORER_SHARE of it), at least EXPLORER_MIN, at most
    half the screen and never leaving the right panels under RIGHT_MIN (the minimum wins on tiny screens)."""
    want = round(total * EXPLORER_SHARE) if want is None else want
    return max(EXPLORER_MIN, min(want, int(total * EXPLORER_MAX_SHARE), total - RIGHT_MIN))


def node_label(node):
    t = Text(node.label, style="bold" if node.kind == "group" else "")
    if node.kind in ("group", "dataset"):
        t.append(f"  {len(node.children)}", style="dim")
    if node.state:
        t.append("  ")
        t.append(STATE_SHORT[node.state], style=STATE_STYLE[node.state])
    if node.note:
        t.append(f"  {node.note}", style="yellow" if "missing" in node.note else "dim")
    return t


class Explorer(Selectable, FullRow, Tree):
    """Keyboard tree of explorer.build() nodes; node data = node key. Expanded keys and the cursor survive
    rebuilds; a cursor whose item vanished lands on its nearest surviving ancestor. Dragging its right border
    resizes it."""
    CURSOR = "tree--cursor"
    dragging, lit = False, None  # class defaults: Tree.__init__ already runs the cursor watcher
    BINDINGS = [Binding("tab", "cursor_down", "next", show=False),
                Binding("shift+tab", "cursor_up", "previous", show=False),
                Binding("left", "collapse_or_parent", "collapse", show=False),
                Binding("right", "expand", "expand", show=False),
                Binding("enter", "screen.describe", "describe")]

    def __init__(self, **kwargs):
        super().__init__("outputs", **kwargs)
        self.show_root, self.auto_expand = False, False
        self.expanded, self.seen, self.sig, self.index = set(), set(), None, {}

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

    def on_mouse_down(self, event):
        if event.x >= self.outer_size.width - 1:
            self.dragging = True
            self.capture_mouse()
            self.screen.clear_selection()  # a resize is not a text selection
            event.stop()

    def on_mouse_move(self, event):
        if self.dragging:
            self.screen.set_explorer_width(event.screen_x - self.region.x + 1)

    def on_mouse_up(self, _):
        if self.dragging:
            self.dragging = False
            self.release_mouse()

    def selected_key(self):
        return self.cursor_node.data if self.cursor_node else None

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
        nodes = {}

        def add(parent, node):
            if node.children:
                nodes[node.key] = parent.add(node_label(node), data=node.key, expand=node.key in self.expanded)
                for c in node.children:
                    add(nodes[node.key], c)
            else:
                nodes[node.key] = parent.add_leaf(node_label(node), data=node.key)
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
