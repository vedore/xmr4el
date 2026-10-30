"""Results (tui.md §5.12): Current Baseline over the explorer | Comparison Summary over Detailed Analysis.

One screen, one layout: zero to three challengers next to a fixed baseline (an evaluation chosen on the Dashboard).
The explorer lists the baseline and the evaluations it can be compared with (Show incompatible lists the rest too).
enter inspects the explorer's evaluation in place of the right column. esc: closes the inspection, else from a
right panel (or the filter) goes to the explorer, else back to the Dashboard.
Read-only, from saved files only: the scan's metrics, plus the small run files comparison.read_facts reads in a worker.
Each opening is a new session: no challengers.
"""
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Checkbox, ContentSwitcher

from xmr4el_tui import explorer
from xmr4el_tui.screens.results_details import analysis, baseline_text, inspection, summary
from xmr4el_tui.services.comparison import BASELINE, GROUPS, INCOMPATIBLE, MAX_CHALLENGERS, assess, beam, read_facts
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.explorer import ResultsExplorer
from xmr4el_tui.widgets.resize import column_width
from xmr4el_tui.widgets.results_panels import BaselinePanel, ComparisonSummary, DetailedAnalysis, InspectionView
from xmr4el_tui.widgets.top_bar import TopBar

LEFT_SHARE, LEFT_MIN, LEFT_MAX_SHARE, RIGHT_MIN, STEP = 0.30, 24, 0.5, 40, 2
GROUP_NAMES = tuple(GROUPS)


class FactsRead(Message):
    def __init__(self, seq, facts):
        super().__init__()
        self.seq, self.facts = seq, facts


class ResultsScreen(Screen):
    """Hidden keys (in the ? panel): ctrl+tab / ctrl+shift+tab (or ] / [) cycle panels, < / > resize the columns,
    ctrl+r rescans (facts reread too)."""
    BINDINGS = [Binding("escape", "back", "back"),
                *(Binding(str(i + 1), f"group({i})", "metric group" if i == 0 else name, show=i == 0,
                          key_display=f"1-{len(GROUP_NAMES)}") for i, name in enumerate(GROUP_NAMES)),
                Binding("tab,ctrl+tab,right_square_bracket", "panel(1)", "next panel", show=False, priority=True),
                Binding("shift+tab,ctrl+shift+tab,left_square_bracket", "panel(-1)", "previous panel", show=False, priority=True),
                Binding("ctrl+r", "rescan", "rescan", show=False),
                Binding("less_than_sign", f"resize({-STEP})", "narrower left", show=False),
                Binding("greater_than_sign", f"resize({STEP})", "wider left", show=False)]

    def __init__(self, baseline):
        """`baseline`: explorer key ("eval", run id, split) of an evaluation with metrics, in the app's snapshot."""
        super().__init__()
        self.base, self.members = baseline, []  # challengers in selection order
        self.group, self.inspecting, self.left_want, self.show_all, self.roots = 0, None, None, False, ()
        self.facts, self.compat, self.names, self.seq, self.facts_runs = {}, {}, {}, 0, None
        self.loading, self.base_gone = False, False  # facts being read; baseline no longer in the scan

    @property
    def crumbs(self):
        return ("Results", "Inspect") if self.inspecting is not None else ("Results",)

    def back_to_crumb(self, level):
        """Navbar click on Results while inspecting: as esc from the inspection."""
        if self.inspecting is not None:
            self.action_back()

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="res"):
            with Vertical(id="res-left"):
                yield BaselinePanel(id="baseline")
                yield Checkbox("Show incompatible", id="show-all")
                yield ResultsExplorer(id="res-explorer")
            with ContentSwitcher(id="res-right", initial="res-compare"):
                with Vertical(id="res-compare"):
                    yield ComparisonSummary(id="summary")
                    yield DetailedAnalysis(id="analysis")
                yield InspectionView(id="inspect")
        yield ContextFooter()

    def on_mount(self):
        for wid, title in (("baseline", "Current Baseline"), ("res-explorer", "Explorer"),
                           ("summary", "Comparison Summary"), ("inspect", "Inspection")):
            self.query_one(f"#{wid}").border_title = title
        tree = self.query_one(ResultsExplorer)
        self.set_left_width(None)
        self.refresh_tree()
        for key in tree.index.get(self.base, (None, ()))[1]:  # the baseline's tree open, cursor on the baseline
            tree.nodes[key].expand()
        if self.base in tree.nodes and tree.last_line >= 0:  # last_line rebuilds the lines: node.line is current
            tree.move_cursor(tree.nodes[self.base])
        tree.focus()

    # --- data -----------------------------------------------------------------------------------------------
    def on_runs_changed(self, _):
        self.refresh_tree()

    def refresh_tree(self):
        """Explorer = the Dashboard's Models branch (datasets > trees > evaluations), cut to `visible`; cursor,
        expansion and members survive. Challengers whose run vanished are dropped (hidden ones are not); the
        baseline keeps its last facts, marked as no longer found."""
        tree, snap = self.query_one(ResultsExplorer), self.app.snapshot
        self.roots = explorer.build(snap.runs, snap.trees)[0].children
        full = explorer.index(self.roots)
        kept = {self.base: self.names[self.base]} if self.base in self.names else {}  # vanished baseline: last name
        self.names = kept | {key: f"{tree_node.label} · {tree_node.short} · {node.label}"
                             for key, (node, path) in full.items() if node.kind == "eval"
                             for tree_node in (full[path[-1]][0],)}
        if self.base not in full and not self.base_gone:
            self.app.say("Baseline no longer found (run moved or deleted): its last read values stay shown.",
                         "warning", self)
        self.base_gone = self.base not in full
        gone = [m for m in self.members if m not in full]
        if gone:
            self.members = [m for m in self.members if m not in gone]
            self.app.say(f"{len(gone)} challenger(s) no longer found (run moved or deleted): removed.",
                         "warning", self)
        sig = tree.sig
        self.refill()
        if gone or tree.sig is not sig:  # an unchanged rescan redraws nothing
            self.mark()
        if snap.runs != self.facts_runs:
            self.facts_runs = snap.runs
            self.seq += 1
            self.loading = True
            self.load_facts({k: n.ref for k, (n, _) in full.items() if n.kind == "eval"}, snap.runs, self.seq)

    def visible(self, node, _ancestors=()):
        """Shown in the explorer: the baseline, and (Show incompatible off) only evaluations assessed VERIFIED or
        CAUTION; a tree or dataset is shown while one of its evaluations is. Not yet assessed: hidden."""
        if self.show_all or node.key == self.base:
            return True
        c = self.compat.get(node.key)
        return node.kind == "eval" and c is not None and c.level != INCOMPATIBLE

    def refill(self):
        tree = self.query_one(ResultsExplorer)
        tree.fill(self.roots if self.show_all else explorer.prune(self.roots, self.visible)[0])

    def on_checkbox_changed(self, event):
        """Show incompatible: a view filter only (members, baseline, group, inspection unchanged)."""
        self.show_all = event.value
        self.refill()
        self.mark()

    @work(thread=True, exclusive=True, group="facts")
    def load_facts(self, evals, all_runs, seq):
        self.post_message(FactsRead(seq, {k: read_facts(run, e, all_runs) for k, (run, e) in evals.items()}))

    def on_facts_read(self, message):
        if message.seq != self.seq:
            return
        self.loading = False
        facts = message.facts
        if self.base not in facts and self.base in self.facts:  # vanished: compare against what was read
            facts[self.base] = self.facts[self.base]
        self.facts = facts
        base = facts.get(self.base)
        self.compat = {k: assess(base, f) for k, f in facts.items()} if base else {}
        dropped = [m for m in self.members if self.compat.get(m, BASELINE).level == INCOMPATIBLE]
        if dropped:
            self.members = [m for m in self.members if m not in dropped]
            self.app.say(f"{len(dropped)} challenger(s) became incompatible after a rescan: removed.",
                         "warning", self)
        tree = self.query_one(ResultsExplorer)
        tree.compat = self.compat
        self.refill()
        self.mark()

    def mark(self):
        tree = self.query_one(ResultsExplorer)
        tree.marks = {self.base: "base", **{m: str(i + 1) for i, m in enumerate(self.members)}}  # role numbers
        tree.restyle()
        self.show_right()

    # --- panels ---------------------------------------------------------------------------------------------
    def entries(self):
        keys = [self.base, *self.members]
        return [self.facts[k] for k in keys if k in self.facts], [self.compat.get(k, BASELINE) for k in keys
                                                                   if k in self.facts]

    def show_right(self):
        name = self.names.get(self.base)
        run_eval = self.query_one(ResultsExplorer).index.get(self.base, (None,))[0]
        base_beam = beam(run_eval.ref[1]) if run_eval else (self.facts[self.base].beam if self.base in self.facts
                                                            else None)
        self.query_one(BaselinePanel).update(baseline_text(name, base_beam, self.base_gone))
        if self.inspecting is not None:
            self.show_inspection()
            return
        entries, compats = self.entries()
        panel = self.query_one(DetailedAnalysis)
        panel.border_title = "Detailed Analysis"
        panel.border_subtitle = "  ".join(f"[b]{i + 1} {n}[/b]" if i == self.group else f"{i + 1} {n}"
                                          for i, n in enumerate(GROUP_NAMES))
        if not entries or entries[0].key != self.base:
            self.query_one(ComparisonSummary).show("Reading evaluation files…" if self.loading else
                                                   "Baseline no longer found (run moved or deleted): esc returns.")
            panel.show([], [], [])
            return
        box = self.query_one(ComparisonSummary)
        box.show(summary(entries, compats, self.names, box.scrollable_content_region.width or 80))
        panel.show(*analysis(GROUP_NAMES[self.group], entries, compats, self.names))

    def show_inspection(self):
        key = self.inspecting
        f = self.facts.get(key)
        view = self.query_one(InspectionView)
        if f is None:
            view.show("Reading evaluation files…" if self.loading else
                      "This evaluation is no longer found (run moved or deleted): esc closes the inspection.")
        else:
            view.show(inspection(f, self.compat.get(key, BASELINE), self.names.get(key, key[1]), key == self.base))

    # --- actions --------------------------------------------------------------------------------------------
    def selected_eval(self):
        tree = self.query_one(ResultsExplorer)
        node = tree.index.get(tree.selected_key(), (None,))[0]
        return node if node is not None and node.kind == "eval" else None

    def action_toggle_member(self):
        """space: add or remove the explorer's evaluation; on a tree or dataset it expands / collapses it."""
        node = self.selected_eval()
        if node is None:
            self.query_one(ResultsExplorer).action_toggle_node()
        else:
            self.toggle(node.key)

    def on_tree_node_selected(self, event):
        """A click on an evaluation row (a click on a tree / dataset only expands it: Tree's auto_expand)."""
        node = self.query_one(ResultsExplorer).index.get(event.node.data, (None,))[0]
        if node is not None and node.kind == "eval":
            self.toggle(node.key)

    def toggle(self, key):
        c = self.compat.get(key)
        if key == self.base:
            self.app.say("This is the baseline: choose another one on the Dashboard.")
        elif key in self.members:
            self.members.remove(key)
            self.mark()
        elif c is None:
            self.app.say("Checking compatibility with the baseline…")
        elif c.level == INCOMPATIBLE:
            self.app.say(f"Incompatible with the baseline: {c.reason}.", "warning")
        elif len(self.members) >= MAX_CHALLENGERS:
            self.app.say(f"Maximum of {MAX_CHALLENGERS} challengers: remove one (space) to add another.", "warning")
        else:
            self.members.append(key)
            self.mark()

    def action_inspect(self):
        """enter: the explorer's evaluation, read-only, in place of the right column (membership unchanged)."""
        node = self.selected_eval()
        if self.inspecting is not None:
            return
        if node is None:
            self.app.say("Move to an evaluation to inspect it.")
            return
        self.inspecting = node.key
        self.query_one(ResultsExplorer).disabled = True
        self.query_one(ContentSwitcher).current = "inspect"
        self.query_one(TopBar).refresh()
        self.show_inspection()
        view = self.query_one(InspectionView)
        view.scroll_home(animate=False)
        view.focus()

    def action_back(self):
        """esc: close the inspection (explorer, members, group and scroll as they were); from another panel, to the
        explorer; from the explorer, back to the Dashboard (an explorer search is cleared by the explorer first)."""
        tree = self.query_one(ResultsExplorer)
        if self.inspecting is None:
            if self.focused is tree:
                self.dismiss()
            else:
                tree.focus()
            return
        self.inspecting = None
        tree.disabled = False
        self.query_one(ContentSwitcher).current = "res-compare"
        self.query_one(TopBar).refresh()
        self.show_right()
        tree.focus()

    def action_rescan(self):
        """ctrl+r: reread every run file and every evaluation's facts, even when the scan finds the runs equal."""
        self.facts_runs = None
        self.app.rescan(announce=True)

    def action_group(self, i):
        self.group = i
        self.show_right()

    def action_panel(self, step):
        panels = [self.query_one(InspectionView)] if self.inspecting is not None else [
            self.query_one(Checkbox), self.query_one(ResultsExplorer), self.query_one(ComparisonSummary),
            self.query_one("#analysis-table")]
        here = next((p for p in panels if self.focused is p), None)
        i = panels.index(here) if here is not None else -step
        panels[(i + step) % len(panels)].focus()

    def check_action(self, action, parameters):
        return action != "group" or self.inspecting is None

    # --- layout ---------------------------------------------------------------------------------------------
    def set_left_width(self, want):
        """Keep `want` cells (clamped) for this session; terminal resizes re-clamp it (ResultsExplorer border drag)."""
        width = self.size.width
        self.left_want = None if want is None else column_width(want, width, LEFT_SHARE, LEFT_MIN, LEFT_MAX_SHARE,
                                                                RIGHT_MIN)
        self.query_one("#res-left").styles.width = column_width(self.left_want, width, LEFT_SHARE, LEFT_MIN,
                                                                 LEFT_MAX_SHARE, RIGHT_MIN)
        self.call_after_refresh(self.show_right)  # Summary names are cut to the panel's new width

    def action_resize(self, step):
        self.set_left_width(column_width(self.left_want, self.size.width, LEFT_SHARE, LEFT_MIN, LEFT_MAX_SHARE,
                                         RIGHT_MIN) + step)

    def on_resize(self, _):
        self.set_left_width(self.left_want)

