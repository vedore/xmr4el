"""Dashboard, the TUI's only screen: explorer of saved trees and their evaluations, read-only description, active runs.

Read-only: nothing here writes, moves or deletes a file. Data comes from the app's scan (outputs/runs/ and
outputs/saved_trees/, every RESCAN_SECONDS in a worker thread); the explorer redraws only when what it shows changed.
"""
import re
from datetime import datetime
from pathlib import Path

from rich.segment import Segment
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.strip import Strip
from textual.widgets import DataTable, Footer, OptionList, Static, Tree
from textual.widgets.option_list import Option

from xmr4el_tui import explorer, fmt, runs
from xmr4el_tui.compare import RECALL
from xmr4el_tui.progress import event_time, read_events
from xmr4el_tui.runs import elapsed, heartbeat_age, state_word

STATE_STYLE = {"RUNNING": "bold #5CC8FF", "UNVERIFIED": "#F5C451", "LOST": "#FF6B6B", "COMPLETED": "#7BD88F",
               "FAILED": "#FF6B6B", "STOPPED": "#F5C451", "UNKNOWN": ""}
STATE_SHORT = {"RUNNING": "RUN", "UNVERIFIED": "UNVER", "LOST": "LOST", "COMPLETED": "DONE", "FAILED": "FAIL",
               "STOPPED": "STOP", "UNKNOWN": "?"}
STRING_GROUPS = ("all", "seen, 1 label", "seen, >1 label", "unseen string")
STRING_COLS = ("tree", "dict", "either", "routed")


class RunsChanged(Message):
    """The app rescanned outputs/runs."""


ACTIVE_COLUMNS = (("run", "Run"), ("type", "Type"), ("model", "Model"), ("state", "State"), ("elapsed", "Elapsed"),
                  ("stage", "Stage"), ("started", "Started"), ("device", "Device"), ("id", "Run id"))
KIND = {"train": "Training", "eval": "Evaluation", "unknown": "unknown"}
TITLE = "XMR4EL"
AREAS = ("Dashboard", "New Experiment", "Run Monitor", "Results")
HEAD = "bold"  # a line wholly in this style is a Description heading: shown, not selectable
EXPLORER_SHARE, EXPLORER_MIN, EXPLORER_MAX_SHARE, RIGHT_MIN, EXPLORER_STEP = 0.22, 18, 0.5, 40, 2


def explorer_width(want, total):
    """Explorer cells on a `total`-cell screen: `want` (None: EXPLORER_SHARE of it), at least EXPLORER_MIN, at most
    half the screen and never leaving the right panels under RIGHT_MIN (the minimum wins on tiny screens)."""
    want = round(total * EXPLORER_SHARE) if want is None else want
    return max(EXPLORER_MIN, min(want, int(total * EXPLORER_MAX_SHARE), total - RIGHT_MIN))


class TopBar(Static):
    """XMR4EL centred on the full width; the areas on the right, `area` highlighted, dropped when they would reach
    the title. Shared by every screen."""
    COMPONENT_CLASSES = {"topbar--title", "topbar--area", "topbar--current"}

    def __init__(self, area, **kwargs):
        super().__init__("", **kwargs)
        self.area = area

    def render(self):
        width, style = self.size.width, self.get_component_rich_style
        # centred on the app, not the bar (the help panel narrows the bar)
        start = max(0, min(width - len(TITLE), (self.app.size.width - len(TITLE)) // 2 - self.region.x))
        nav = Text(" ").join(Text(f" {a} ", style=style("topbar--current" if a == self.area else "topbar--area"))
                             for a in AREAS)
        gap = width - start - len(TITLE) - nav.cell_len - 1
        return Text.assemble(" " * start, (TITLE, style("topbar--title")), *((" " * gap, nav) if gap >= 2 else ()))


class FullRow:
    """Paints the cursor style over the whole row (guides, padding, every column), not only the label or cells."""
    CURSOR = ""

    def render_line(self, y):
        strip = super().render_line(y)
        if not self.cursor_at(y):
            return strip
        return Strip(Segment.apply_style(list(strip), post_style=self.get_component_rich_style(self.CURSOR)),
                     strip.cell_length)


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
        t.append("\nProvenance\n", style=HEAD)
        _rows(t, [("Git", f"{str(git.get('commit'))[:12]} · {'dirty' if git.get('dirty') else 'clean'}"
                   if git.get("commit") else None),
                  ("Python", prov.get("python")), ("Device", prov.get("torch_device")),
                  ("Packages", " · ".join(f"{k} {v}" for k, v in (prov.get("packages") or {}).items()) or None),
                  ("Files", str(run.path / "provenance.json") if prov else None)])


def tree_text(node, root):
    tr, t = node.ref, Text()
    trained = tr.trained_by
    t.append(f"{node.label}\n", style=HEAD)
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
        t.append("\nModel (xmodel.json)\n", style=HEAD)
        _rows(t, [("Features", view.get("features")), ("Transformer", _config(view.get("transformer_config"))),
                  ("Vectorizer", _config(view.get("vectorizer_config"))),
                  ("Reduction", _config(view.get("dimension_config"))), ("Depth", view.get("depth")),
                  ("Clustering", _config(view.get("clustering_config"))),
                  ("Matcher", _config(view.get("matcher_config"))), ("Min leaf", view.get("min_leaf_size")),
                  ("Search", _search(view.get("predict_config")))])
    t.append("\nEvaluations\n", style=HEAD)
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
    t.append(f"{e.split} · {run.name}\n", style=HEAD)
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
    t.append(f"{run.name}\n", style=HEAD)
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
    t = Text()
    t.append(f"{node.label}\n", style=HEAD)
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


class Selectable:
    """Mouse drag selects text in a line-based widget, as Textual does for its text widgets (the app copies it on
    release). Each rendered line is remembered with its scroll x, so the selection is extracted from what was shown.
    ponytail: offsets are cells, so a wide character shifts the range; fine for the ASCII paths, ids and metrics."""
    ALLOW_SELECT = True  # Tree, OptionList and DataTable turn it off

    def __init__(self, *args, **kwargs):
        self.plain = {}  # virtual line -> (scroll x, text shown)
        super().__init__(*args, **kwargs)

    def render_line(self, y):
        strip = super().render_line(y)
        x, y = self.scroll_offset.x, self.scroll_offset.y + y
        self.plain[y] = (x, "".join(seg.text for seg in strip))
        span = self.text_selection.get_span(y) if self.text_selection else None
        if span is not None:
            start, end = max(0, span[0] - x), strip.cell_length if span[1] == -1 else min(strip.cell_length, span[1] - x)
            if start < end:
                style = self.screen.get_component_rich_style("screen--selection")
                strip = Strip.join([strip.crop(0, start), Strip(Segment.apply_style(
                    list(strip.crop(start, end)), post_style=style), end - start), strip.crop(end)])
        return strip.apply_offsets(x, y)

    def get_selection(self, selection):
        text = "\n".join(" " * self.plain[i][0] + self.plain[i][1] if i in self.plain else ""
                         for i in range(max(self.plain, default=-1) + 1))
        return "\n".join(line.rstrip() for line in selection.extract(text).split("\n")), "\n"


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


def selectable(line):
    """A Description line the cursor stops on: not blank, not a heading."""
    head = str(line.style) == HEAD or any(str(sp.style) == HEAD and sp.start == 0 and sp.end >= len(line.plain)
                                          for sp in line.spans)
    return bool(line.plain.strip()) and not head


class Description(Selectable, FullRow, OptionList):
    """Read-only lines of the selected item, one option each: up / down (or a click) select a row, headings and
    blank lines are skipped, nothing is editable. left / right leave for Results / Run monitor. The selected row is
    one colour (a green COMPLETED would vanish on the accent)."""
    CURSOR = "option-list--option-highlighted"
    BINDINGS = [Binding("left", "screen.results", "results"), Binding("right", "screen.monitor", "monitor"),
                Binding("escape", "screen.focus_explorer", "explorer")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subject, self.shown, self.keys = None, [], []

    def show(self, subject, text):
        """Show `text` for `subject` (an explorer key). Unchanged content is not redrawn; a refresh of the same
        subject keeps the row with the same field label selected."""
        lines = text.copy()
        lines.rstrip()
        lines = lines.split("\n", allow_blank=True)
        shown = [(line.plain, tuple(line.spans), str(line.style)) for line in lines]
        if subject == self.subject and shown == self.shown:
            return
        old = self.keys[self.highlighted] if subject == self.subject and self.highlighted is not None else None
        self.clear_options()
        self.add_options([Option(line, disabled=not selectable(line)) for line in lines])
        self.subject, self.shown, self.keys = subject, shown, [line.plain[:14].strip() for line in lines]
        rows = [i for i, line in enumerate(lines) if selectable(line)]
        self.highlighted = next((i for i in rows if self.keys[i] == old), rows[0] if rows else None)

    def cursor_at(self, y):
        try:
            return self.highlighted is not None and self._lines[self.scroll_offset.y + y][0] == self.highlighted
        except IndexError:
            return False

    def text(self):
        return "\n".join(plain for plain, _, _ in self.shown)


class ActiveRuns(Selectable, FullRow, DataTable):
    """Active training and evaluation runs; enter opens the Run monitor."""
    CURSOR = "datatable--cursor"
    BINDINGS = [Binding("enter", "select_cursor", "monitor")]

    def cursor_at(self, y):
        return bool(self.row_count) and y >= 1 and y - 1 + self.scroll_offset.y == self.cursor_row


DESCRIPTION_TITLE = {"tree": "Trained tree", "eval": "Evaluation", "dataset": "Dataset", "group": "Group"}


class DashboardScreen(Screen):
    """Explorer | Description over Active runs. Hidden keys (in the ? panel): ctrl+tab / ctrl+shift+tab (or ] / [)
    cycle panels, ctrl+b hides / shows the explorer, < / > resize it, ctrl+r rescans."""
    BINDINGS = [Binding("ctrl+tab,right_square_bracket", "panel(1)", "next panel", show=False),
                Binding("ctrl+shift+tab,left_square_bracket", "panel(-1)", "previous panel", show=False),
                Binding("ctrl+b", "toggle_explorer", "hide / show explorer", show=False),
                Binding("less_than_sign", f"resize_explorer({-EXPLORER_STEP})", "narrower explorer", show=False),
                Binding("greater_than_sign", f"resize_explorer({EXPLORER_STEP})", "wider explorer", show=False),
                Binding("ctrl+r", "rescan", "rescan", show=False)]

    def compose(self) -> ComposeResult:
        yield TopBar("Dashboard", id="top")
        with Horizontal(id="dash"):
            yield Explorer(id="explorer")
            with Vertical(id="dash-right"):
                yield Description(id="description")
                yield ActiveRuns(id="active-runs", cursor_type="row")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        for wid, title in (("explorer", "Explorer"), ("description", "Description"),
                           ("active-runs", "Active runs")):
            self.query_one(f"#{wid}").border_title = title
        table = self.query_one(ActiveRuns)
        for key, label in ACTIVE_COLUMNS:
            table.add_column(label, key=key)
        self.active_ids, self.explorer_want = [], None  # None: EXPLORER_SHARE of the width
        self.set_explorer_width(None)
        self.refresh_all()
        self.query_one(Explorer).focus()
        self.set_interval(1, self.tick)

    # --- explorer size ----------------------------------------------------------------------------------------
    def set_explorer_width(self, want):
        """Keep `want` cells (clamped) for this session; terminal resizes re-clamp it."""
        self.explorer_want = None if want is None else explorer_width(want, self.size.width)
        self.query_one(Explorer).styles.width = explorer_width(self.explorer_want, self.size.width)

    def on_resize(self, _):
        self.set_explorer_width(self.explorer_want)

    def action_resize_explorer(self, step):
        self.query_one(Explorer).display = True
        self.set_explorer_width(explorer_width(self.explorer_want, self.size.width) + step)

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
        table = self.query_one(ActiveRuns)
        keep = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value if table.row_count else None
        active = self.active_runs()
        table.clear()
        for r in active:
            stage, device = self.app.active.get(r.id, (None, None))
            model = (r.train_path and str(Path(r.train_path).parent).removeprefix("datasets/")) if r.kind == "train" \
                else f"tree {fmt.tree(r.tree_id)}"
            table.add_row(r.name, KIND[r.kind], model or fmt.DASH, _state(state_word(r)),
                          fmt.duration(elapsed(r, datetime.now())), stage or fmt.DASH, fmt.stamp(r.started),
                          device or fmt.DASH, r.id, key=r.id)
        self.active_ids = [r.id for r in active]
        if keep in self.active_ids:
            table.move_cursor(row=self.active_ids.index(keep))

    def tick(self):
        table = self.query_one(ActiveRuns)
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
        desc = self.query_one(Description)
        if node is None:
            text = Text("Nothing selected.")
        elif node.kind == "tree":
            text = tree_text(node, self.app.root)
        elif node.kind == "eval":
            text = eval_text(node, parent)
        elif node.kind == "run":
            text = run_text(node.ref, self.app.active)
        else:
            text = group_text(node, self.app)
        kind = f"{KIND[node.ref.kind]} run" if node and node.kind == "run" else DESCRIPTION_TITLE.get(node and node.kind)
        desc.border_title = f"Description — {kind}" if kind else "Description"
        desc.show(node and node.key, text)

    def on_tree_node_highlighted(self, _):
        self.show()

    def on_data_table_row_selected(self, event):
        if event.data_table.id == "active-runs":
            self.open_run(event.row_key.value)

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
