"""Dashboard, the TUI's only screen: explorer of saved trees and their evaluations, read-only description, active runs.

Read-only: nothing here writes, moves or deletes a file. Data comes from the app's scan (outputs/runs/ and
outputs/saved_trees/, every RESCAN_SECONDS in a worker thread); the explorer redraws only when what it shows changed.
"""
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable

from xmr4el_tui import explorer, fmt
from xmr4el_tui.runs import elapsed, state_word
from xmr4el_tui.screens.dashboard_details import (DESCRIPTION_TITLE, KIND, VIEW_LOGS, eval_text, group_text,
                                                   run_text, state_text, tree_text)
from xmr4el_tui.screens.batch_monitor import BatchesScreen, BatchMonitorScreen, duration, word
from xmr4el_tui.screens.log_screen import LogScreen
from xmr4el_tui.screens.monitor_screen import RunMonitorScreen
from xmr4el_tui.screens.results_screen import ResultsScreen
from xmr4el_tui.services.experiment_repository import Activity, Details
from xmr4el_tui.widgets.description import Description
from xmr4el_tui.widgets.explorer import EXPLORER_STEP, Explorer, explorer_width
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.row_rendering import FullRow, Selectable
from xmr4el_tui.widgets.top_bar import TopBar

ACTIVE_COLUMNS = (("run", "Run"), ("type", "Type"), ("model", "Model"), ("state", "State"), ("elapsed", "Elapsed"),
                  ("stage", "Stage"), ("started", "Started"), ("device", "Device"), ("id", "Run id"))
NARROW = {"run": 16, "model": 12}  # the long names are cropped so State and Elapsed stay in view at 80 columns


class ActiveRuns(Selectable, FullRow, DataTable):
    """Active training and evaluation runs; enter opens the Run monitor."""
    CURSOR = "datatable--cursor"
    BINDINGS = [Binding("enter", "select_cursor", "monitor")]

    def cursor_at(self, y):
        return bool(self.row_count) and y >= 1 and y - 1 + self.scroll_offset.y == self.cursor_row


class DetailsLoaded(Message):
    """The repository read the Details of explorer item `key` (posted from a worker thread)."""

    def __init__(self, key, seq, details):
        super().__init__()
        self.key, self.seq, self.details = key, seq, details


class DashboardScreen(Screen):
    """Explorer | Description over Active runs. tab / shift+tab (also ctrl+tab, ctrl+shift+tab, ] and [)
    cycle panels; hidden keys (in the ? panel): ctrl+b hides / shows the explorer, < / > resize it, ctrl+r rescans."""
    crumbs = ("Dashboard",)
    BINDINGS = [Binding("tab,ctrl+tab,right_square_bracket", "panel(1)", "next panel", show=False, priority=True),
                Binding("shift+tab,ctrl+shift+tab,left_square_bracket", "panel(-1)", "previous panel", show=False,
                        priority=True),
                Binding("ctrl+b", "toggle_explorer", "hide / show explorer", show=False),
                Binding("less_than_sign", f"resize_explorer({-EXPLORER_STEP})", "narrower explorer", show=False),
                Binding("greater_than_sign", f"resize_explorer({EXPLORER_STEP})", "wider explorer", show=False),
                Binding("ctrl+r", "rescan", "rescan", show=False),
                Binding("r", "results", "results"), Binding("b", "batches", "batches"),
                Binding("n", "app.new_experiment", "new"),
                Binding("c", "app.resume_experiment", "resume")]

    def check_action(self, action, parameters):
        """Resume Experiment is offered while an unfinished draft exists."""
        return self.app.resumable() if action == "app.resume_experiment" else True

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="dash"):
            yield Explorer(id="explorer")
            with Vertical(id="dash-right"):
                yield Description(id="description")
                yield ActiveRuns(id="active-runs", cursor_type="row")
        yield ContextFooter()

    def on_mount(self):
        for wid, title in (("explorer", "Explorer"), ("description", "Description"),
                           ("active-runs", "Active runs")):
            self.query_one(f"#{wid}").border_title = title
        table = self.query_one(ActiveRuns)
        for key, label in ACTIVE_COLUMNS:
            table.add_column(label, key=key, width=NARROW.get(key))
        self.active_ids, self.explorer_want = [], None  # None: EXPLORER_SHARE of the width
        self.loaded, self.detail_seq = (None, Details()), 0  # (explorer key, its Details), latest request
        self.set_left_width(None)
        self.refresh_all()
        self.query_one(Explorer).focus()
        self.set_interval(1, self.tick)

    # --- explorer size ----------------------------------------------------------------------------------------
    def set_left_width(self, want):
        """Keep `want` cells (clamped) for this session; terminal resizes re-clamp it."""
        self.explorer_want = None if want is None else explorer_width(want, self.size.width)
        self.query_one(Explorer).styles.width = explorer_width(self.explorer_want, self.size.width)

    def on_resize(self, _):
        self.set_left_width(self.explorer_want)

    def action_resize_explorer(self, step):
        self.query_one(Explorer).display = True
        self.set_left_width(explorer_width(self.explorer_want, self.size.width) + step)

    def action_toggle_explorer(self):
        tree = self.query_one(Explorer)
        tree.display = not tree.display
        if tree.display:
            tree.focus()
        elif self.focused is None or self.focused is tree:
            self.query_one(Description).focus()

    def on_runs_changed(self, _):
        self.refresh_all()

    def refresh_all(self):
        gone = self.query_one(Explorer).fill(explorer.build(self.app.snapshot.runs, self.app.snapshot.trees))
        if gone is not None:
            self.app.say(f"The selected {gone[0]} is no longer found (moved, renamed or deleted outside the TUI).",
                         "warning", self)
        self.fill_active()
        self.show()

    # --- panels ---------------------------------------------------------------------------------------------
    def active_runs(self):
        return [r for r in self.app.snapshot.runs if state_word(r) in explorer.ACTIVE]

    def fill_active(self):
        table = self.query_one(ActiveRuns)
        keep = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value if table.row_count else None
        active = self.active_runs()
        live = [i for i in self.app.snapshot.batches if i.active]
        table.clear()
        table.forget_lines()
        for i in live:  # a running batch: one row above its trees' own runs
            cur = i.current
            table.add_row(i.name, "Batch", f"{len(i.status['trees'])} trees", word(i.state), duration(i),
                          cur["name"] if cur else fmt.DASH, fmt.stamp(None), fmt.DASH, i.id, key=f"batch:{i.id}")
        for r in active:
            activity = self.app.snapshot.active.get(r.id, Activity())
            stage, device = activity.stage, activity.device
            model = (r.train_path and str(Path(r.train_path).parent).removeprefix("datasets/")) if r.kind == "train" \
                else f"tree {fmt.tree(r.tree_id)}"
            table.add_row(r.name, KIND[r.kind], model or fmt.DASH, state_text(state_word(r)),
                          fmt.duration(elapsed(r, datetime.now())), stage or fmt.DASH, fmt.stamp(r.started),
                          device or fmt.DASH, r.id, key=r.id)
        self.active_ids = [f"batch:{i.id}" for i in live] + [r.id for r in active]
        if keep in self.active_ids:
            table.move_cursor(row=self.active_ids.index(keep))

    def tick(self):
        table = self.query_one(ActiveRuns)
        for r in self.app.snapshot.runs:
            if r.id in self.active_ids:
                table.update_cell(r.id, "elapsed", fmt.duration(elapsed(r, datetime.now())))

    def selected(self):
        """(node, parent node) under the explorer cursor, (None, None) without one."""
        tree = self.query_one(Explorer)
        node, path = tree.index.get(tree.selected_key(), (None, ()))
        return node, tree.index[path[-1]][0] if path else None

    def show(self, load=True):
        """Describe the explorer's item now, with the Details loaded for it so far; `load` rereads its files in a
        worker (no file I/O on the UI thread), redrawn when they land if no newer request was made."""
        node, parent = self.selected()
        desc, snap = self.query_one(Description), self.app.snapshot
        key = node and node.key
        details = self.loaded[1] if self.loaded[0] == key else Details()
        if node is None:
            text = Text("Nothing selected.")
        elif node.kind == "tree":
            text = tree_text(node, details)
        elif node.kind == "eval":
            text = eval_text(node, parent, snap.active.get(node.ref[0].id, Activity()), details)
        elif node.kind == "run":
            text = run_text(node.ref, snap.active.get(node.ref.id, Activity()), details)
        else:
            text = group_text(node, snap, self.app.root)
        kind = f"{KIND[node.ref.kind]} run" if node and node.kind == "run" else DESCRIPTION_TITLE.get(node and node.kind)
        desc.border_title = f"Description — {kind}" if kind else "Description"
        desc.show(key, text)
        if load and node is not None and node.kind in ("tree", "eval", "run"):
            self.detail_seq += 1
            self.load_details(node, self.detail_seq)

    @work(thread=True, exclusive=True, group="details")
    def load_details(self, node, seq):
        self.post_message(DetailsLoaded(node.key, seq, self.app.repo.details(node)))

    def on_details_loaded(self, message):
        if message.seq == self.detail_seq:
            self.loaded = (message.key, message.details)
            self.show(load=False)

    def on_tree_node_highlighted(self, _):
        self.show()

    def on_option_list_option_selected(self, event):
        """enter / click on the Description's View Logs row: the run's log, fullscreen; esc comes back here with the
        row still selected."""
        desc = self.query_one(Description)
        if event.option_list is not desc or desc.keys[event.option_index] != VIEW_LOGS:
            return
        node, _ = self.selected()
        kind = node and node.kind
        run = node.ref[0] if kind == "eval" else node.ref if kind == "run" else \
            node.ref.trained_by if kind == "tree" else None
        if run and self.run_here(run.id):
            self.app.push_screen(LogScreen(run, self.app.root))
        else:
            self.app.say("That run directory is no longer found.", "warning")

    def on_data_table_row_selected(self, event):
        if event.data_table.id == "active-runs":
            key = event.row_key.value
            if key.startswith("batch:"):
                info = next((i for i in self.app.snapshot.batches if i.id == key[6:]), None)
                self.app.push_screen(BatchMonitorScreen(key[6:], info)) if info else None
            else:
                self.open_run(key)

    def action_describe(self):
        self.show()
        self.query_one(Description).focus()

    def action_focus_explorer(self):
        tree = self.query_one(Explorer)
        tree.display = True
        tree.focus()

    def action_panel(self, step):
        panels = [p for p in (self.query_one(Explorer), self.query_one(Description), self.query_one(ActiveRuns))
                  if p.display]
        i = panels.index(self.focused) if self.focused in panels else -step
        panels[(i + step) % len(panels)].focus()

    # --- navigation -------------------------------------------------------------------------------------------
    def run_here(self, run_id):
        return any(r.id == run_id for r in self.app.snapshot.runs)

    def open_run(self, run_id, split=None):
        """Run monitor of `run_id` (its `split` stage selected); esc comes back here with the selection kept."""
        if run_id and self.run_here(run_id):
            self.app.push_screen(RunMonitorScreen(run_id, split))
        else:
            self.app.say("That run directory is no longer found.", "warning")

    def open_results(self, key):
        """Results with evaluation `key` as the baseline; esc comes back here with the selection kept."""
        self.app.push_screen(ResultsScreen(key))

    def action_results(self):
        """Results with the selected evaluation as the baseline (a tree with exactly one evaluation with metrics:
        that one). Never picks among several: the tree opens so one can be chosen."""
        node, _ = self.selected()
        if node is None or node.kind not in ("tree", "eval"):
            self.app.say("Select an evaluation to open Results.")
            return
        done = [node] if node.kind == "eval" else [c for c in node.children if c.ref[1].metrics]
        if node.kind == "eval" and not node.ref[1].metrics:
            self.app.say("This evaluation has no metrics yet: it cannot be a baseline.")
        elif len(done) == 1:
            self.open_results(done[0].key)
        elif not done:
            self.app.say("No evaluation of this tree has metrics yet.")
        else:
            tree = self.query_one(Explorer)
            tree.nodes[node.key].expand()
            tree.focus()
            self.app.say(f"This tree has {len(done)} evaluations: select one as the baseline, then r.")

    def action_monitor(self):
        """Run screen of the selected run: the evaluation's run (on its split), a tree's training run."""
        node, _ = self.selected()
        if node is None or node.kind in ("group", "dataset"):
            self.app.say("Select a tree, an evaluation or a run.")
        elif node.kind == "eval":
            self.open_run(node.ref[0].id, node.ref[1].split)
        elif node.kind == "run":
            self.open_run(node.ref.id)
        elif node.ref.trained_by:
            self.open_run(node.ref.trained_by.id)
        else:
            self.app.say("No recorded training run for this tree (trained outside run_experiment, or not synced).")

    def action_batches(self):
        self.app.push_screen(BatchesScreen())

    def action_rescan(self):
        self.app.rescan(announce=True)
