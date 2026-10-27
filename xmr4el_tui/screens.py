"""Runs and Run screens (tui.md §5.1-5.4, §5.9, §7)."""
import time
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Input, Select, Static, TabbedContent, TabPane

from xmr4el_tui import compare, fmt, process, runs
from xmr4el_tui.launch import SetupScreen
from xmr4el_tui.widgets import (STATE_SHORT, LogView, TopBar, acc1, details_text, elapsed, inputs_text,
                                overview_text, split_cell, state_text, state_word)

SORTS = ("started", "run", "acc@1")
DETAILS_WIDTH = 40  # = HelpPanel width in app.tcss
NARROW = 100  # below this terminal width the details panel sits under the table
FORCE_AFTER = 30  # seconds after SIGTERM before `k` (SIGKILL) is offered (§5.9)


class RunsChanged(Message):
    """The app rescanned outputs/runs."""


def columns_for(width):
    """Visible columns at a terminal width; Duration, then Started, then Kind drop first (§5.1)."""
    cols = ["mark", "started", "duration", "run", "kind", "tree", "split", "acc@1", "state"]
    if width < 100:
        cols.remove("duration")
    if width < 80:
        cols.remove("started")
    if width < 72:
        cols.remove("kind")
    return cols


HEADERS = {"mark": " ", "started": "Started", "duration": "Duration", "run": "Run", "kind": "Kind", "tree": "Tree",
           "split": "Split", "acc@1": "acc@1", "state": "State"}


class RunTable(DataTable):
    """home/end move the row cursor (DataTable only scrolls)."""
    BINDINGS = [Binding("home", "first", "first run", show=False), Binding("end", "last", "last run", show=False)]

    def action_first(self):
        self.move_cursor(row=0)

    def action_last(self):
        self.move_cursor(row=self.row_count - 1)


class RunsScreen(Screen):
    BINDINGS = [Binding("enter", "open", "open"), Binding("n", "new", "new run"), Binding("t", "trees", "trees"),
                Binding("space", "mark", "mark"), Binding("c", "compare", "compare"),
                Binding("slash", "filter", "filter", show=False),
                Binding("o", "sort", "sort", show=False), Binding("ctrl+r", "rescan", "rescan", show=False),
                Binding("escape", "clear_filter", "clear filter", show=False)]

    def __init__(self):
        super().__init__()
        self.filter, self.sort, self.marked, self.cols = "", "started", [], []

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="body"):
            with Vertical(id="main"):
                yield RunTable(id="runs", cursor_type="row")
                yield Input(placeholder="filter: run name, run id, tree id or state", id="filter")
            with VerticalScroll(classes="side-panel"):
                yield Static(id="details")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        self.query_one("#filter").display = False
        self.set_class((self.size.width or self.app.size.width) < NARROW, "-narrow")
        self.rebuild()

    def on_runs_changed(self, _):
        self.rebuild()

    def table_width(self):
        width = self.size.width or self.app.size.width
        return width if width < NARROW else width - DETAILS_WIDTH

    def on_resize(self, _):
        self.set_class((self.size.width or self.app.size.width) < NARROW, "-narrow")
        if columns_for(self.table_width()) != self.cols:
            self.rebuild()

    def visible(self):
        needle = self.filter.lower()
        shown = [r for r in self.app.runs if not needle or any(
            needle in (s or "").lower() for s in (r.name, r.id, r.tree_id, state_word(r),
                                                  STATE_SHORT[state_word(r)]))]
        if self.sort == "run":
            shown.sort(key=lambda r: r.name)  # stable: newest first within a name
        elif self.sort == "acc@1":  # dev and test never ranked together
            shown.sort(key=lambda r: (split_cell(r), -(acc1(r) if acc1(r) is not None else -1)))
        return shown

    def selected_id(self):
        table = self.query_one(DataTable)
        if not table.row_count:
            return None
        return table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value

    def selected(self):
        rid = self.selected_id()
        return next((r for r in self.app.runs if r.id == rid), None)

    def cells(self, run, bold_tree):
        value = {"mark": "*" if run.id in self.marked else " ", "started": fmt.stamp(run.started),
                 "duration": fmt.duration(elapsed(run)), "run": run.name, "kind": run.kind,
                 "tree": Text(fmt.tree(run.tree_id), style="bold" if bold_tree else ""),
                 "split": split_cell(run), "acc@1": Text(fmt.metric(acc1(run)), justify="right"),
                 "state": state_text(run, short=True)}
        return [value[c] for c in self.cols]

    def rebuild(self):
        """Redraw the table, keeping the selected run by id."""
        table = self.query_one(DataTable)
        keep = self.selected_id()
        self.cols = columns_for(self.table_width())
        table.clear(columns=True)
        for c in self.cols:
            table.add_column(HEADERS[c], key=c)
        shown = self.visible()
        tree = next((r.tree_id for r in shown if r.id == keep), None)
        for r in shown:
            table.add_row(*self.cells(r, tree and r.tree_id == tree), key=r.id)
        ids = [r.id for r in shown]
        if keep in ids:
            table.move_cursor(row=ids.index(keep))
        self.update_top()
        self.update_details()

    def update_top(self):
        running = sum(r.recorded == "running" for r in self.app.runs)
        sort = "" if self.sort == "started" else f" · sorted by {self.sort}"
        filt = f" · filter '{self.filter}'" if self.filter else ""
        self.query_one(TopBar).set(f"Runs  {len(self.app.runs)}{filt}{sort}",
                                   f"mark {len(self.marked)}/2 · {self.app.host} · {running} running")

    def update_details(self):
        details = self.query_one("#details", Static)
        if not self.app.runs_found:
            details.update(f"outputs/runs/ not found under {self.app.root}. Start in the repository or pass --root.")
        elif not self.app.runs:
            details.update("No runs in outputs/runs/. On a laptop, sync outputs/runs/ from the server.")
        elif (run := self.selected()) is None:
            details.update("No run matches the filter (esc clears it).")
        else:
            details.update(details_text(run, self.app.runs))

    def on_data_table_row_highlighted(self, event):
        run = self.selected()
        if run is None:
            return
        table = self.query_one(DataTable)
        for r in self.visible():  # bold Tree cells on the selected run's tree
            table.update_cell(r.id, "tree", Text(fmt.tree(r.tree_id),
                                                 style="bold" if run.tree_id and r.tree_id == run.tree_id else ""))
        self.update_details()

    def on_data_table_row_selected(self, event):
        self.app.push_screen(RunScreen(event.row_key.value))

    def action_open(self):  # the table handles enter itself; this is its footer entry
        if (rid := self.selected_id()) is not None:
            self.app.push_screen(RunScreen(rid))

    def action_new(self):
        self.app.push_screen(SetupScreen())

    def action_trees(self):
        self.app.push_screen(TreesScreen())

    def action_mark(self):
        rid = self.selected_id()
        if rid is None:
            return
        if rid in self.marked:
            self.marked.remove(rid)
        elif len(self.marked) == 2:
            self.notify("Two runs are marked; unmark one first (space).")
            return
        else:
            self.marked.append(rid)
        if "mark" in self.cols:
            self.query_one(DataTable).update_cell(rid, "mark", "*" if rid in self.marked else " ")
        self.update_top()

    def action_compare(self):
        ids = {r.id for r in self.app.runs}
        self.marked = [rid for rid in self.marked if rid in ids]
        if len(self.marked) < 2:
            self.notify("Mark one more run (space)." if self.marked else "Mark two runs (space), then c.")
            return
        self.app.push_screen(CompareScreen(*self.marked))

    def action_filter(self):
        box = self.query_one("#filter", Input)
        box.display = True
        box.focus()

    def on_input_changed(self, event):
        self.filter = event.value
        self.rebuild()

    def on_input_submitted(self, _):
        self.query_one(DataTable).focus()

    def action_clear_filter(self):
        box = self.query_one("#filter", Input)
        if box.display:
            box.value = ""
            box.display = False
            self.query_one(DataTable).focus()

    def action_sort(self):
        self.sort = SORTS[(SORTS.index(self.sort) + 1) % len(SORTS)]
        self.rebuild()

    def action_rescan(self):
        self.app.rescan(announce=True)


class StopModal(ModalScreen):
    """Confirm a stop; focus starts on Keep running (§5.9). Dismisses with True to stop."""
    BINDINGS = [Binding("escape", "dismiss(False)", "keep running")]

    def __init__(self, run):
        super().__init__()
        self.run = run

    def compose(self) -> ComposeResult:
        with Vertical(id="stop-box"):
            yield Static(Text(f"Stop {self.run.name}?", style="bold"))
            yield Static(f"Sends SIGTERM to {self.run.name} (pid {self.run.identity['pid']}) and its workers. It is "
                         "recorded as STOPPED. The current stage is lost; stages already done and saved files are "
                         "kept.", id="stop-text")
            with Horizontal(id="stop-buttons"):
                yield Button("Keep running", id="keep", compact=True)
                yield Button("Stop run", id="confirm", variant="error", compact=True)

    def on_mount(self):
        self.query_one("#keep", Button).focus()

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "confirm")


class RunScreen(Screen):
    BINDINGS = [Binding("1", "tab('overview')", "overview"), Binding("2", "tab('log')", "log"),
                Binding("3", "tab('inputs')", "inputs"), Binding("w", "warnings", "warnings"),
                Binding("s", "stop", "stop"), Binding("k", "force_stop", "force stop"),
                Binding("y", "copy", "copy path", show=False),
                Binding("escape", "app.pop_screen", "runs")]

    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id
        self.run = next(r for r in self.app.runs if r.id == run_id)
        self.split = self.run.evals[0].split if self.run.evals else None
        self.stopping = None  # time.monotonic() of the SIGTERM this screen sent

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="run-actions"):
            yield Button("Stop run", id="stop", variant="error", compact=True)
            yield Static(id="stop-line")
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                with VerticalScroll():
                    if len(self.run.evals) > 1:
                        yield Select([(e.split, e.split) for e in self.run.evals], value=self.split,
                                     allow_blank=False, id="split")
                    yield Static(id="overview-text")
            with TabPane("Log", id="log"):
                yield LogView(self.run.path / "run.log", id="logview")
            with TabPane("Inputs", id="inputs"):
                with VerticalScroll():
                    yield Static(id="inputs-text")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        self.render_run()
        self.set_interval(1, self.update_top)

    def on_runs_changed(self, _):
        found = next((r for r in self.app.runs if r.id == self.run_id), None)
        if found is not None:
            self.run = found
            if not self.stoppable():  # recorded STOPPED, or gone (LOST) after a force stop
                self.stopping = None
            self.render_run()

    def stoppable(self):
        return self.run.live is not None and self.run.live.stoppable

    def force_offered(self):
        return self.stopping is not None and time.monotonic() - self.stopping >= FORCE_AFTER and self.stoppable()

    def check_action(self, action, parameters):
        if action == "stop":
            return self.stoppable() and self.stopping is None
        if action == "force_stop":
            return self.force_offered()
        return True

    def update_top(self):
        state = Text("STOPPING", style="yellow") if self.stopping is not None else state_text(self.run)
        right = Text.assemble(state, f" {fmt.duration(elapsed(self.run, datetime.now()))} · {self.app.host}")
        self.query_one(TopBar).set(self.run.name, right)
        self.query_one("#run-actions").display = self.stoppable() or self.stopping is not None
        self.query_one("#stop", Button).display = self.stopping is None
        if self.stopping is not None:
            waited = time.monotonic() - self.stopping
            self.query_one("#stop-line", Static).update(
                Text("k force stop (SIGKILL; shown as LOST if no final status was written)", style="yellow")
                if self.force_offered() else Text(f"SIGTERM sent {fmt.duration(waited)} ago; waiting for the run "
                                                  "to record STOPPED", style="dim"))
        self.refresh_bindings()

    def render_run(self):
        self.update_top()
        self.query_one("#overview-text", Static).update(overview_text(self.run, self.app.runs, self.split))
        self.query_one("#inputs-text", Static).update(inputs_text(self.run))

    def on_select_changed(self, event):
        self.split = event.value
        self.render_run()

    def action_tab(self, tab):
        self.query_one(TabbedContent).active = tab

    def action_warnings(self):
        self.action_tab("log")
        on = self.query_one(LogView).toggle_warnings()
        self.notify("Log: warnings and errors only" if on else "Log: all lines")

    def action_copy(self):
        self.app.copy_to_clipboard(str(self.run.path))
        self.notify(f"Copied {self.run.path}")

    def on_button_pressed(self, event):
        if event.button.id == "stop":
            self.action_stop()

    def action_stop(self):
        if self.stoppable() and self.stopping is None:
            self.app.push_screen(StopModal(self.run), self.stop_confirmed)

    def stop_confirmed(self, confirmed):
        if not confirmed:
            return
        result = process.stop(self.run)  # re-verifies identity and group before signalling (§9)
        if result.startswith("SIGTERM"):
            self.stopping = time.monotonic()
        self.notify(result, severity="information" if self.stopping is not None else "warning")
        self.update_top()

    def action_force_stop(self):
        if self.force_offered():
            self.notify(process.stop(self.run, force=True), severity="warning")


class TreesScreen(Screen):
    """Saved trees (from the runs' tree.json and outputs/saved_trees/); enter evaluates the selected one (§5.6)."""
    BINDINGS = [Binding("e", "evaluate", "evaluate"), Binding("enter", "evaluate", show=False),
                Binding("escape", "app.pop_screen", "runs")]

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="body"):
            yield RunTable(id="trees", cursor_type="row")
            with VerticalScroll(classes="side-panel"):
                yield Static(id="tree-details")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        self.set_class((self.size.width or self.app.size.width) < NARROW, "-narrow")
        self.rebuild()

    def on_resize(self, _):
        self.set_class((self.size.width or self.app.size.width) < NARROW, "-narrow")

    def on_runs_changed(self, _):
        self.rebuild()

    def rebuild(self):
        table = self.query_one(DataTable)
        keep = self.selected()
        self.trees = runs.trees(self.app.root, self.app.runs)
        table.clear(columns=True)
        for c in ("Tree", "Trained by", "Training file", "Labels", "Runs", "Here"):
            table.add_column(c, key=c)
        for t in self.trees:
            train = t.train_path()
            table.add_row(fmt.tree(t.id), t.trained_by.name if t.trained_by else fmt.DASH,
                          train.removeprefix("datasets/").rsplit(".", 1)[0] if train else fmt.DASH,
                          Text(fmt.count(t.labels), justify="right"), Text(str(len(t.used_by)), justify="right"),
                          Text("yes", style="green") if t.here else Text("no", style="dim"), key=t.path)
        paths = [t.path for t in self.trees]
        if keep and keep.path in paths:
            table.move_cursor(row=paths.index(keep.path))
        self.query_one(TopBar).set(f"Trees  {len(self.trees)}", f"{sum(t.here for t in self.trees)} on this machine"
                                                               f" · {self.app.host}")
        self.update_details()

    def selected(self):
        table = self.query_one(DataTable)
        if not table.row_count:
            return None
        path = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
        return next((t for t in getattr(self, "trees", ()) if t.path == path), None)

    def update_details(self):
        t = self.selected()
        details = self.query_one("#tree-details", Static)
        if t is None:
            details.update("No saved trees: runs that train write one (n), or sync outputs/saved_trees/.")
            return
        out = Text()
        out.append(f"{t.id or 'no tree id'}\n{t.path}\n", style="bold")
        out.append("on this machine\n\n" if t.here else f"not on this machine\nmissing: {t.path}/xmodel.pkl\n\n",
                   style="green" if t.here else "yellow")
        if t.trained_by:
            out.append(f"Trained by {t.trained_by.name}\n{t.train_path() or ''}\n\n")
        for r in t.used_by:
            out.append(f"{fmt.stamp(r.started)}  {r.name}  {split_cell(r)} {fmt.metric(acc1(r))}\n")
        details.update(out)

    def on_data_table_row_highlighted(self, _):
        self.update_details()

    def on_data_table_row_selected(self, _):
        self.action_evaluate()

    def action_evaluate(self):
        t = self.selected()
        if t is None:
            return
        if not t.here:
            self.notify(f"tree {fmt.tree(t.id)} not on this machine\nmissing: {t.path}/xmodel.pkl",
                        severity="warning")
            return
        name = f"{t.trained_by.name}_eval" if t.trained_by else f"tree_{fmt.tree(t.id)}"
        self.app.push_screen(SetupScreen({"name": name}, t.path))


class CompareScreen(Screen):
    """Two marked runs, one selected evaluation each (§5.8); deltas only where §11 allows them."""
    BINDINGS = [Binding("x", "swap", "swap"), Binding("escape", "app.pop_screen", "runs")]

    def __init__(self, a_id, b_id, splits=None):
        super().__init__()
        self.ids = [a_id, b_id]
        self.splits = list(splits or compare.default_splits(*self.sides()))

    def sides(self):
        by_id = {r.id: r for r in self.app.runs}
        return by_id.get(self.ids[0]), by_id.get(self.ids[1])

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        for i, (side, run) in enumerate(zip("AB", self.sides())):
            with Horizontal(classes="side"):
                yield Static(Text.assemble((f" {side}  ", "bold"), f"{run.name:<22} tree {fmt.tree(run.tree_id)}"
                                           "   Evaluation "), classes="who")
                if len(run.evals) > 1:
                    yield Select([(e.split, e.split) for e in run.evals], value=self.splits[i], allow_blank=False,
                                 id=f"sel-{i}")
                else:
                    yield Static(self.splits[i] or fmt.DASH)
        with VerticalScroll():
            yield Static(id="compare-text")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        self.query_one(TopBar).set("Compare", self.app.host)
        self.render_compare()

    def on_runs_changed(self, _):
        self.render_compare()

    def on_select_changed(self, event):
        self.splits[int(event.select.id[-1])] = event.value
        self.render_compare()

    def action_swap(self):
        self.app.switch_screen(CompareScreen(*reversed(self.ids), splits=self.splits[::-1]))

    def render_compare(self):
        a, b = self.sides()
        body = self.query_one("#compare-text", Static)
        if a is None or b is None:
            body.update("A compared run directory is gone (esc returns to runs).")
            return
        v = compare.verdict(a, self.splits[0], b, self.splits[1], self.app.runs)
        t = Text()
        for i, r in enumerate(v.reasons):
            t.append(f"{r}\n", style=("green" if v.subset_deltas else "yellow") if i == 0 else "")
        if v.differs:
            t.append("Differs: ", style="bold")
            t.append(", ".join(v.differs) + "\n")
        ea, eb = a.eval(self.splits[0]), b.eval(self.splits[1])
        if ea and ea.metrics and eb and eb.metrics:
            rows = compare.rows(ea.metrics, eb.metrics, v)
            w = 2 + max(len(c) for _, x, y, _ in rows for c in (x, y))
            t.append(f"\n{'':<13}{'A':>{w}}{'B':>{w}}" + ("   B - A" if any(d for *_, d in rows) else "") + "\n",
                     style="bold")
            for k, x, y, d in rows:
                t.append(f"{k:<13}{x:>{w}}{y:>{w}}   ")
                t.append(d, style=delta_style(d, lower_better=k == "ms/query"))
                t.append("\n")
        body.update(t)


def delta_style(d, lower_better=False):
    """Green where B is better, red where worse, dim when equal, yellow for a note (devices differ)."""
    num = d.split()[0] if d else ""
    if num[:1] not in ("+", "-") or not num[1:].replace(".", "").isdigit():
        return "yellow"
    if float(num) == 0:
        return "dim"
    return "green" if (float(num) > 0) != lower_better else "red"
