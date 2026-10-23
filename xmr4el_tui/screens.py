"""Runs and Run screens (tui.md §5.1-5.4, §7)."""
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Input, Select, Static, TabbedContent, TabPane

from xmr4el_tui import compare, fmt
from xmr4el_tui.widgets import (LogView, TopBar, acc1, details_text, elapsed, inputs_text, overview_text,
                                split_cell, state_text, state_word)

SORTS = ("started", "run", "acc@1")
DETAILS_WIDTH = 40  # = HelpPanel width in app.tcss
NARROW = 100  # below this terminal width the details panel sits under the table


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
    BINDINGS = [Binding("enter", "open", "open"), Binding("n", "new", "new run"), Binding("e", "evaluate", "evaluate"),
                Binding("space", "mark", "mark"), Binding("c", "compare", "compare"),
                Binding("slash", "filter", "filter"),
                Binding("o", "sort", "sort", show=False), Binding("ctrl+r", "rescan", "rescan", show=False),
                Binding("escape", "clear_filter", "clear filter", show=False)]

    def __init__(self):
        super().__init__()
        self.filter, self.sort, self.marked, self.cols = "", "started", [], []

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        yield RunTable(id="runs", cursor_type="row")
        yield Input(placeholder="filter: run name, run id, tree id or state", id="filter")
        with VerticalScroll(classes="side-panel"):
            yield Static(id="details")
        yield Footer()

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
            needle in (s or "").lower() for s in (r.name, r.id, r.tree_id, state_word(r)))]
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
                 "state": state_text(run)}
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
        self.app.setup()

    def action_evaluate(self):
        if (run := self.selected()) is not None:
            self.app.evaluate_tree(run)

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


class RunScreen(Screen):
    BINDINGS = [Binding("1", "tab('overview')", "overview"), Binding("2", "tab('log')", "log"),
                Binding("3", "tab('inputs')", "inputs"), Binding("w", "warnings", "warnings"),
                Binding("y", "copy", "copy path"), Binding("e", "evaluate", "evaluate", show=False),
                Binding("escape", "app.pop_screen", "runs")]

    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id
        self.run = next(r for r in self.app.runs if r.id == run_id)
        self.split = self.run.evals[0].split if self.run.evals else None

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
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
        yield Footer()

    def on_mount(self):
        self.render_run()
        self.set_interval(1, self.update_top)

    def on_runs_changed(self, _):
        found = next((r for r in self.app.runs if r.id == self.run_id), None)
        if found is not None:
            self.run = found
            self.render_run()

    def update_top(self):
        right = Text.assemble(state_text(self.run), f" {fmt.duration(elapsed(self.run, datetime.now()))}"
                                                    f" · {self.app.host}")
        self.query_one(TopBar).set(self.run.name, right)

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

    def action_evaluate(self):
        self.app.evaluate_tree(self.run)

    def action_copy(self):
        self.app.copy_to_clipboard(str(self.run.path))
        self.notify(f"Copied {self.run.path}")


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
        yield Footer()

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
