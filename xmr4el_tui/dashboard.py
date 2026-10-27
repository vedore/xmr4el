"""Dashboard, the TUI's only screen: explorer of saved trees and their evaluations, read-only description, active runs.

Read-only: nothing here writes, moves or deletes a file. Data comes from the app's scan (outputs/runs/ and
outputs/saved_trees/, every RESCAN_SECONDS in a worker thread); the explorer redraws only when what it shows changed.
"""
import re
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static, Tree

from xmr4el_tui import explorer, fmt, runs
from xmr4el_tui.compare import RECALL
from xmr4el_tui.progress import event_time, read_events
from xmr4el_tui.runs import elapsed, heartbeat_age, state_word

STATE_STYLE = {"RUNNING": "bold blue", "UNVERIFIED": "yellow", "LOST": "red", "COMPLETED": "green",
               "FAILED": "red", "STOPPED": "yellow", "UNKNOWN": ""}
STATE_SHORT = {"RUNNING": "RUN", "UNVERIFIED": "UNVER", "LOST": "LOST", "COMPLETED": "DONE", "FAILED": "FAIL",
               "STOPPED": "STOP", "UNKNOWN": "?"}
STRING_GROUPS = ("all", "seen, 1 label", "seen, >1 label", "unseen string")
STRING_COLS = ("tree", "dict", "either", "routed")


class RunsChanged(Message):
    """The app rescanned outputs/runs."""


ACTIVE_COLUMNS = (("run", "Run"), ("type", "Type"), ("state", "State"), ("elapsed", "Elapsed"), ("stage", "Stage"),
                  ("model", "Model"), ("started", "Started"), ("device", "Device"), ("id", "Run id"))
KIND = {"train": "Training", "eval": "Evaluation", "unknown": "unknown"}


class TopBar(Static):
    """`XMR4EL  <place>` left, state · host right (§4)."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.left, self.right = "", Text()

    def set(self, left, right):
        self.left, self.right = left, right if isinstance(right, Text) else Text(right)
        self.refresh()

    def render(self):
        left = Text(f"XMR4EL  {self.left}", style="bold")
        gap = max(1, self.size.width - left.cell_len - self.right.cell_len)
        return Text.assemble(left, " " * gap, self.right)


def bold_heads(t, *heads):
    """Bold `heads` where they start a line of `t` (section and field labels)."""
    t.highlight_regex(r"(?m)^(?:" + "|".join(map(re.escape, heads)) + r")", "bold")
    return t


def banner_text(run):
    """§5.4 banner for runs that did not complete, or None."""
    word = state_word(run)
    if word == "FAILED":
        return Text.assemble((f"{fmt.marker('failed', fmt.use_ascii())} {run.error or 'no error recorded'}", "red"),
                             f"\n  recorded in status.json at {fmt.stamp(run.finished, long=True)}    2 full log")
    if word == "STOPPED":
        return Text(f"Stopped by {run.error or 'a signal'} at {fmt.stamp(run.finished, long=True)}. "
                    "Completed stages and saved files are kept.", style="yellow")
    if word == "LOST":
        events, _ = read_events(run.path / "events.jsonl")
        last = next((e for e in reversed(events) if e.get("stage") and event_time(e)), None)
        where = f" Last event: {last['stage']} at {fmt.stamp(event_time(last), long=True)}." if last else ""
        return Text(f"Process {run.identity['pid']} is gone and wrote no final status (killed, e.g. out of "
                    f"memory).{where}", style="red")
    if word == "UNVERIFIED":
        age = heartbeat_age(run)
        beat = f" Heartbeat {fmt.duration(age)} ago." if age is not None else ""
        if run.live and run.live.other_runtime:
            return Text(f"Recorded running on {run.identity.get('host') or 'another host'}, another machine or "
                        f"container.{beat}", style="yellow")
        if run.identity:
            reason = run.live.reason if run.live else "not checked"
            return Text(f"Recorded running here, but the process cannot be inspected ({reason}). "
                        f"Stop unavailable.{beat}", style="yellow")
        return Text("Recorded running, but this run records no process identity (written before the runner "
                    "change), so liveness cannot be checked.", style="yellow")
    if word == "UNKNOWN":
        reason = next((p for p in run.problems if p.startswith("status.json")), "status.json: missing")
        return Text(f"status.json is missing or unreadable: {reason.split(': ', 1)[1]}. "
                    "Showing the files that exist.")
    return None


def metrics_text(m, split):
    """Evaluation block of §5.3 for one metrics dict."""
    if not m:
        return Text(f"{split}: no metrics (not produced yet)")
    search = m.get("search") or {}
    lines = ["Search      " + " · ".join(f"{k} {v}" for k, v in search.items()),
             f"Rows        {fmt.count(m.get('rows'))} / {fmt.count(m.get('rows_total'))} have a gold label in the "
             f"tree ({fmt.pct(m.get('rows'), m.get('rows_total'))}) · {fmt.count(m.get('labels'))} labels", ""]
    cells = [(k, m.get(k)) for k in ("acc@1", "acc@1_all", "MRR", *[k for k in RECALL if k in m], "R@cand")]
    if "hybrid" in m:
        cells += [("hybrid", m.get("hybrid")), ("hybrid_all", m.get("hybrid_all"))]
    row = []
    for k, v in cells:
        row.append(f"{k:<10} {fmt.metric(v):>6}")
        if len(row) == 3:
            lines.append("      ".join(row))
            row = []
    lines += ["      ".join(row)] if row else []
    lines += [f"{fmt.ms(m.get('ms_per_query'))}, batch incl. encoding", ""]
    strings = m.get("strings")
    if strings:
        lines.append(f"{'Mention strings':<16} {'n':>6}" + "".join(f"{k:>10}" for k in STRING_COLS))
        for g in STRING_GROUPS:
            s = strings.get(g, {})
            lines.append(f"{g:<16} {fmt.count(s.get('n')):>6}" + "".join(
                f"{fmt.metric(s.get(k)):>10}" for k in STRING_COLS))
    else:
        lines.append("Mention strings  not recorded (no train_path in this evaluation)")
    t = bold_heads(Text("\n".join(lines)), "Search", "Rows", "Mention strings")
    t.highlight_regex(r"(?m)^acc@1 +[\d.]+", "bold")
    return t


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


def _rows(t, rows):
    for k, v in rows:
        if v is None or v == "":
            continue
        t.append(f"{k:<14}", style="dim")
        t.append_text(v if isinstance(v, Text) else Text(str(v)))
        t.append("\n")


def _config(cfg):
    """{"type": t, "kwargs": {...}} -> `t (k=v, ...)`."""
    if not isinstance(cfg, dict):
        return None if cfg is None else str(cfg)
    kwargs = cfg.get("kwargs") if isinstance(cfg.get("kwargs"), dict) else {}
    return f"{cfg.get('type')}" + (f" ({', '.join(f'{k}={v}' for k, v in kwargs.items())})" if kwargs else "")


def _search(search):
    return " · ".join(f"{k} {v}" for k, v in search.items()) if isinstance(search, dict) and search else None


def _state(word):
    return Text(word, style=STATE_STYLE[word])


def _provenance(t, run):
    git = run.git() if isinstance(run.git(), dict) else {}
    prov = run.provenance() if isinstance(run.provenance(), dict) else {}
    if git or prov:
        t.append("\nProvenance\n", style="bold")
        _rows(t, [("Git", f"{str(git.get('commit'))[:12]} · {'dirty' if git.get('dirty') else 'clean'}"
                   if git.get("commit") else None),
                  ("Python", prov.get("python")), ("Device", prov.get("torch_device")),
                  ("Packages", " · ".join(f"{k} {v}" for k, v in (prov.get("packages") or {}).items()) or None),
                  ("Files", str(run.path / "provenance.json") if prov else None)])


def tree_text(node, root):
    tr, t = node.ref, Text()
    trained = tr.trained_by
    t.append(f"{node.label}\n", style="bold")
    dataset = explorer.dataset_of(tr)
    place = Text("on this machine", style="green") if tr.here else Text(
        f"missing: {tr.path}/xmodel.pkl (moved, deleted or not synced)", style="yellow")
    _rows(t, [("Tree id", tr.id or "none recorded"),
              ("Dataset", dataset or f"{explorer.UNKNOWN_DATASET} (no training run recorded)"),
              ("Saved at", tr.path), ("", place),
              ("Recorded at", f"{tr.recorded_path} (runs' tree.json; found by tree_id at Saved at)"
               if tr.recorded_path else None),
              ("Labels", fmt.count(tr.labels) if tr.labels is not None else None)])
    if trained:
        spec = trained.spec() if isinstance(trained.spec(), dict) else {}
        snapshot = trained.path / "model_config.json"
        try:
            saved = fmt.stamp(datetime.fromtimestamp((trained.path / "tree.json").stat().st_mtime), long=True)
        except OSError:
            saved = None
        _rows(t, [("Training run", f"{trained.name} · {trained.id}"), ("Run state", _state(state_word(trained))),
                  ("Training file", trained.train_path), ("Model config", spec.get("model_config")),
                  ("Config copy", str(snapshot) if snapshot.is_file() else None), ("Saved", saved)])
    else:
        _rows(t, [("Training run", "none recorded (trained outside run_experiment, or its run is not synced)")])
    view, _ = runs.read_json(Path(root, tr.path, "xmodel.json"))
    if isinstance(view, dict):
        t.append("\nModel (xmodel.json)\n", style="bold")
        _rows(t, [("Features", view.get("features")), ("Transformer", _config(view.get("transformer_config"))),
                  ("Vectorizer", _config(view.get("vectorizer_config"))),
                  ("Reduction", _config(view.get("dimension_config"))), ("Depth", view.get("depth")),
                  ("Clustering", _config(view.get("clustering_config"))),
                  ("Matcher", _config(view.get("matcher_config"))), ("Min leaf", view.get("min_leaf_size")),
                  ("Search", _search(view.get("predict_config")))])
    t.append("\nEvaluations\n", style="bold")
    if not node.children:
        t.append("none recorded\n", style="dim")
    for c in node.children:
        run, e = c.ref
        t.append(f"{c.label:<28} acc@1 {fmt.metric((e.metrics or {}).get('acc@1'))}  ")
        t.append_text(Text(STATE_SHORT[c.state], style=STATE_STYLE[c.state]))
        t.append("\n")
    if trained:
        _provenance(t, trained)
    return t


def eval_text(node, tree_node):
    (run, e), t = node.ref, Text()
    t.append(f"{e.split} · {run.name}\n", style="bold")
    metrics = run.path / f"metrics_{e.split}.json"
    _rows(t, [("Run", f"{run.id} · {KIND[run.kind]} run"), ("State", _state(node.state)),
              ("Tree", f"{tree_node.label} · {fmt.tree(tree_node.ref.id)}" if tree_node else None),
              ("Test file", e.test_path), ("Strings from", e.train_path),
              ("Started", fmt.stamp(run.started, long=True)), ("Finished", fmt.stamp(run.finished, long=True)
                                                              if run.finished else None),
              ("Duration", fmt.duration(elapsed(run, datetime.now()))),
              ("Metrics file", str(metrics) if e.metrics else "not written yet")])
    if (banner := banner_text(run)) is not None:
        t.append("\n")
        t.append_text(banner)
        t.append("\n")
    t.append("\n")
    if e.metrics:
        t.append_text(metrics_text(e.metrics, e.split))
        t.append("\n")
    else:
        _rows(t, [("Search", _search(e.overrides) or "the tree's saved defaults")])
    _provenance(t, run)
    return t


def run_text(run, info):
    t = Text()
    t.append(f"{run.name}\n", style="bold")
    spec = run.spec() if isinstance(run.spec(), dict) else {}
    stage, device = info.get(run.id, (None, None))
    _rows(t, [("Run id", run.id), ("Type", KIND[run.kind]), ("State", _state(state_word(run))),
              ("Started", fmt.stamp(run.started, long=True)),
              ("Elapsed", fmt.duration(elapsed(run, datetime.now()))),
              ("Stage", stage or ("not recorded" if run.recorded == "running" else None)), ("Device", device),
              ("Tree", spec.get("xmodel_path") if run.kind == "eval" else "none saved"),
              ("Training file", run.train_path), ("Model config", spec.get("model_config")),
              ("Evaluations", ", ".join(f"{e.split} ({e.test_path})" for e in run.evals) or None),
              ("Log", str(run.path / "run.log"))])
    if (banner := banner_text(run)) is not None:
        t.append("\n")
        t.append_text(banner)
        t.append("\n")
    if run.problems:
        t.append("\nproblems: " + "; ".join(run.problems) + "\n", style="yellow")
    return t


def group_text(node, app):
    t = Text(f"{node.label}\n", style="bold")
    if node.kind == "dataset":
        _rows(t, [("Path", node.key[1] or "no training file recorded for these trees"),
                  ("Trees", len(node.children))])
    elif node.key == ("group", "models"):
        n = sum(len(d.children) for d in node.children)
        _rows(t, [("Trees", n), ("Datasets", len(node.children)), ("Runs", len(app.runs))])
        if not app.runs_found:
            t.append(f"\noutputs/runs/ not found under {app.root}. Start in the repository or pass --root.\n")
        elif not n:
            t.append("\nNo saved trees yet: n starts a training run.\n")
    else:
        _rows(t, [("Runs", len(node.children))])
    t.append("\nenter describes the selected item · left results · right run monitor\n", style="dim")
    return t


class Explorer(Tree):
    """Keyboard tree of explorer.build() nodes; node data = node key. Expanded keys and the cursor survive
    rebuilds; a cursor whose item vanished lands on its nearest surviving ancestor."""
    BINDINGS = [Binding("tab", "cursor_down", "next", show=False),
                Binding("shift+tab", "cursor_up", "previous", show=False),
                Binding("left", "collapse_or_parent", "collapse", show=False),
                Binding("right", "expand", "expand", show=False),
                Binding("enter", "screen.describe", "describe")]

    def __init__(self, **kwargs):
        super().__init__("outputs", **kwargs)
        self.show_root, self.auto_expand = False, False
        self.expanded, self.seen, self.sig, self.index = set(), set(), None, {}

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

    def on_tree_node_collapsed(self, event):
        self.expanded.discard(event.node.data)

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


class Description(VerticalScroll):
    """Read-only text of the selected item; left / right leave for Results / Run monitor."""
    BINDINGS = [Binding("left", "screen.results", "results"), Binding("right", "screen.monitor", "monitor"),
                Binding("escape", "screen.focus_explorer", "explorer")]


class DashboardScreen(Screen):
    BINDINGS = [Binding("ctrl+tab,right_square_bracket", "panel(1)", "next panel", key_display="ctrl+tab/]"),
                Binding("ctrl+shift+tab,left_square_bracket", "panel(-1)", "previous panel", show=False),
                Binding("ctrl+r", "rescan", "rescan", show=False)]

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="dash"):
            yield Explorer(id="explorer")
            with Vertical(id="dash-right"):
                with Description(id="description"):
                    yield Static(id="description-text")
                yield DataTable(id="active-runs", cursor_type="row")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        for wid, title in (("explorer", "Models and evaluations"), ("description", "Description"),
                           ("active-runs", "Active runs")):
            self.query_one(f"#{wid}").border_title = title
        table = self.query_one("#active-runs", DataTable)
        for key, label in ACTIVE_COLUMNS:
            table.add_column(label, key=key)
        self.active_ids = []
        self.refresh_all()
        self.query_one(Explorer).focus()
        self.set_interval(1, self.tick)

    def on_runs_changed(self, _):
        self.refresh_all()

    def refresh_all(self):
        gone = self.query_one(Explorer).fill(explorer.build(self.app.runs, self.app.trees))
        if gone is not None:
            self.notify(f"The selected {gone[0]} is no longer found (moved, renamed or deleted outside the TUI).",
                        severity="warning")
        self.fill_active()
        self.show()

    # --- panels ---------------------------------------------------------------------------------------------
    def active_runs(self):
        return [r for r in self.app.runs if state_word(r) in explorer.ACTIVE]

    def fill_active(self):
        table = self.query_one("#active-runs", DataTable)
        keep = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value if table.row_count else None
        active = self.active_runs()
        table.clear()
        for r in active:
            stage, device = self.app.active.get(r.id, (None, None))
            model = (r.train_path and str(Path(r.train_path).parent).removeprefix("datasets/")) if r.kind == "train" \
                else f"tree {fmt.tree(r.tree_id)}"
            table.add_row(r.name, KIND[r.kind], _state(state_word(r)), fmt.duration(elapsed(r, datetime.now())),
                          stage or fmt.DASH, model or fmt.DASH, fmt.stamp(r.started), device or fmt.DASH, r.id,
                          key=r.id)
        self.active_ids = [r.id for r in active]
        if keep in self.active_ids:
            table.move_cursor(row=self.active_ids.index(keep))
        self.query_one(TopBar).set("Dashboard", f"{len(self.app.trees)} trees · {len(active)} active · {self.app.host}")

    def tick(self):
        table = self.query_one("#active-runs", DataTable)
        for r in self.app.runs:
            if r.id in self.active_ids:
                table.update_cell(r.id, "elapsed", fmt.duration(elapsed(r, datetime.now())))

    def selected(self):
        """(node, parent node) under the explorer cursor, (None, None) without one."""
        tree = self.query_one(Explorer)
        node, path = tree.index.get(tree.selected_key(), (None, ()))
        return node, tree.index[path[-1]][0] if path else None

    def show(self):
        node, parent = self.selected()
        text = self.query_one("#description-text", Static)
        if node is None:
            text.update("Nothing selected.")
        elif node.kind == "tree":
            text.update(tree_text(node, self.app.root))
        elif node.kind == "eval":
            text.update(eval_text(node, parent))
        elif node.kind == "run":
            text.update(run_text(node.ref, self.app.active))
        else:
            text.update(group_text(node, self.app))

    def on_tree_node_highlighted(self, _):
        self.show()

    def on_data_table_row_selected(self, event):
        if event.data_table.id == "active-runs":
            self.open_run(event.row_key.value)

    def action_describe(self):
        self.show()
        self.query_one(Description).focus()

    def action_focus_explorer(self):
        self.query_one(Explorer).focus()

    def action_panel(self, step):
        panels = [self.query_one(Explorer), self.query_one(Description), self.query_one("#active-runs")]
        i = panels.index(self.focused) if self.focused in panels else -step
        panels[(i + step) % len(panels)].focus()

    # --- navigation -------------------------------------------------------------------------------------------
    def run_here(self, run_id):
        return any(r.id == run_id for r in self.app.runs)

    def open_run(self, run_id, split=None):
        """Run monitor of `run_id` (on `split`). The screen is not rebuilt yet: says what it would open."""
        if run_id and self.run_here(run_id):
            self.notify(f"Run monitor: {run_id}" + (f" · {split}" if split else "") + " (screen not built yet)")
        else:
            self.notify("That run directory is no longer found.", severity="warning")

    def open_results(self, filter, select):
        """Results scoped by `filter` (tree or run id), cursor on run `select`. Not rebuilt yet: says what it would
        open."""
        self.notify(f"Results: {filter} · run {select} (screen not built yet)")

    def action_results(self):
        """Results scoped to the selected tree (its evaluations), cursor on the selected run."""
        node, parent = self.selected()
        if node is None or node.kind in ("group", "dataset"):
            self.notify("Select a tree, an evaluation or a run.")
        elif node.kind == "run":
            self.open_results(node.ref.id, node.ref.id)
        else:
            tree = node.ref if node.kind == "tree" else parent.ref
            run = node.ref[0] if node.kind == "eval" else tree.trained_by or (tree.used_by or (None,))[0]
            if run:
                self.open_results(tree.id or run.id, run.id)
            else:
                self.notify("No recorded evaluations for this tree.")

    def action_monitor(self):
        """Run screen of the selected run: the evaluation's run (on its split), a tree's training run."""
        node, _ = self.selected()
        if node is None or node.kind in ("group", "dataset"):
            self.notify("Select a tree, an evaluation or a run.")
        elif node.kind == "eval":
            self.open_run(node.ref[0].id, node.ref[1].split)
        elif node.kind == "run":
            self.open_run(node.ref.id)
        elif node.ref.trained_by:
            self.open_run(node.ref.trained_by.id)
        else:
            self.notify("No recorded training run for this tree (trained outside run_experiment, or not synced).")

    def action_rescan(self):
        self.app.rescan(announce=True)
