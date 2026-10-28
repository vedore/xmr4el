"""Run monitor Pipeline panel: the run's stages (progress.stages) as a tree, parts and hierarchy layers below their
stage. Rows are added and relabelled in place, so the cursor and expanded rows survive updates."""
from rich.text import Text
from textual.binding import Binding
from textual.widgets import Tree

from xmr4el_tui.screens.monitor_details import RUN_ROW, run_row, stage_row
from xmr4el_tui.widgets.resize import DragRightBorder
from xmr4el_tui.widgets.row_rendering import FullRow, Selectable


class Pipeline(DragRightBorder, Selectable, FullRow, Tree):
    """Row data = RUN_ROW, (stage,) or (stage, part). Follows the current stage (or the failed / stopped one) until
    the user moves the cursor elsewhere; `follow()` resumes."""
    CURSOR = "tree--cursor"
    ALLOW_MAXIMIZE = True
    # class defaults: Tree.__init__ already runs the cursor watcher. target: the row following last moved to
    following, target = True, None
    BINDINGS = [Binding("tab", "cursor_down", "next", show=False),
                Binding("shift+tab", "cursor_up", "previous", show=False),
                Binding("left", "collapse_or_parent", "collapse", show=False),
                Binding("right", "expand", "expand", show=False)]

    def __init__(self, **kwargs):
        super().__init__("run", **kwargs)
        self.show_root, self.auto_expand = False, True  # enter toggles a stage's parts
        self.rows, self.labels, self.opened = {}, {}, set()  # key -> TreeNode / its label; keys auto-expanded once

    def cursor_at(self, y):
        return self.cursor_line >= 0 and y + self.scroll_offset.y == self.cursor_line

    def selected_key(self):
        return self.cursor_node.data if self.cursor_node else None

    def fill(self, run, stages, initial=None):
        """Show `run`'s `stages`; `initial` (a row key) is selected once it exists and stops following."""
        # columns line up: a row without parts has no expand icon (2 cells), a part sits one guide (4) deeper
        rows = [(RUN_ROW, Text("  ").append_text(run_row(run)))]
        for s in stages:
            rows.append(((s.key,), stage_row(s, run.kind) if s.parts else Text("  ").append_text(stage_row(s, run.kind))))
            rows += [((s.key, p.key), stage_row(p, run.kind, width=18)) for p in s.parts]
        self.prune({key for key, _ in rows})
        for key, label in rows:
            node = self.rows.get(key)
            if node is None:
                parent = self.root if len(key) == 1 else self.rows[key[:1]]
                if parent is not self.root:
                    parent.allow_expand = True
                self.rows[key] = parent.add(label, data=key, allow_expand=False)
            elif self.labels.get(key) != label:
                node.set_label(label)
            self.labels[key] = label
        active = next((s for s in stages if s.state == "current"), None) or \
            next((s for s in stages if s.state in ("failed", "stopped")), None)
        if active and active.parts and active.key not in self.opened:  # its parts shown once; the user may collapse
            self.opened.add(active.key)
            self.rows[(active.key,)].expand()
        if self.last_line < 0:  # builds the lines, so node.line is current
            return
        if initial in self.rows:
            self.following = False
            self.move_cursor(self.rows[initial])
        elif self.following:
            self.target = self.rows[(active.key,)] if active else self.rows[RUN_ROW]
            if self.target is not self.cursor_node:
                self.move_cursor(self.target)

    def prune(self, keep):
        """Remove rows whose stage / part is no longer in the events (a replaced or truncated events.jsonl); a
        selection that goes with them resumes following."""
        gone = sorted(set(self.rows) - keep, key=len, reverse=True)  # parts before their stage
        if self.selected_key() in gone:
            self.follow()
        for key in gone:
            node = self.rows.pop(key)
            self.labels.pop(key, None)
            if len(key) == 1:
                self.opened.discard(key[0])
            node.remove()
            if len(key) == 2 and key[:1] in self.rows:
                parent = self.rows[key[:1]]
                parent.allow_expand = bool(parent.children)

    def follow(self):
        self.following = True
        self.target = None

    def watch_cursor_line(self, previous_line, line):
        super().watch_cursor_line(previous_line, line)
        if self.cursor_node is not None and self.target is not None and self.cursor_node is not self.target:
            self.following = False  # the user moved: stop following

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
