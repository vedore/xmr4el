"""New Experiment, step 3: Review Batch (tui.md §5.15). A compact header (batch name, shared dataset, readiness) over
Execution Queue 30 % | Review Details 70 % (Overview, Differences).

Read-only over the app's draft (`app.draft`): the one thing edited here is the batch name, and the one thing changed
is the execution order (`draft.move`, the same tree objects in a new order). Validation runs on its own when the
screen opens and whenever it is shown again (`review.validate` pieces: trees at once, the shared dataset in a
worker); moving the cursor never re-reads a file. Launch Batch hands a copy of the draft (`review.plan`) to `batch.launch` in a
worker: hashing and the coordinator start never block the UI, a second activation is refused while one runs, and a
failure leaves the draft as it is. Fix Issue opens the existing editor (Model Builder at the slot and
field, Dataset Setup at the mapping) on top of this screen, which stays in the stack, so its selection, view, scroll
and focus are where the user left them.
"""
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Input, Static
from textual.widgets.option_list import Option

from xmr4el_tui import batch, components, review, theme
from xmr4el_tui.screens import review_details
from xmr4el_tui.screens.batch_monitor import BatchMonitorScreen
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.description import Rows
from xmr4el_tui.widgets.top_bar import TopBar

VIEWS = (("overview", "Overview"), ("differences", "Differences"))
STATUS_STYLE = {review.READY_WORD: "success", review.CAUTION_WORD: "warning", review.INVALID_WORD: "error"}


class DatasetChecked(Message):
    def __init__(self, selection, issues):
        super().__init__()
        self.selection, self.issues = selection, issues


class Launched(Message):
    """The launch worker's answer: the started batch (`bdir`, `child`) or why none started (`error`)."""

    def __init__(self, bdir=None, child=None, error=None):
        super().__init__()
        self.bdir, self.child, self.error = bdir, child, error


class Queue(Rows):
    """Execution Queue: up / down or tab select a tree (nothing is edited), enter reviews its details, e edits it in the
    Model Builder, space picks it up to reorder: up / down then carry it, space drops it, esc puts everything back.
    alt+left / alt+right move the selected tree one place without picking it up."""
    BINDINGS = [Binding("enter", "screen.focus_details", "details"), Binding("e", "screen.edit", "edit"),
                Binding("space", "screen.grab", "reorder"),
                Binding("alt+left", "screen.move(-1)", show=False), Binding("alt+right", "screen.move(1)", show=False)]

    def action_cursor_up(self):
        self.screen.carry(-1) if self.screen.grab else super().action_cursor_up()

    def action_cursor_down(self):
        self.screen.carry(1) if self.screen.grab else super().action_cursor_down()


class DetailsList(Rows):
    """The selected tree's lines; enter on a problem is Fix Issue, 1 / 2 or left / right switch the view."""
    BINDINGS = [Binding("enter", "select", "fix issue"), Binding("1", "screen.view('overview')", "overview"),
                Binding("2", "screen.view('differences')", "differences"),
                Binding("left", "screen.view('overview')", show=False),
                Binding("right", "screen.view('differences')", show=False)]


class ViewTabs(Widget, can_focus=True):
    """` 1 Overview  2 Differences `: 1 / 2 or a click choose, enter or space switch to the other view."""
    DEFAULT_CSS = "ViewTabs { height: 1; }"
    BINDINGS = [Binding("1", "screen.view('overview')", "overview"), Binding("2", "screen.view('differences')",
                "differences"), Binding("enter,space", "screen.other_view", "switch view", show=False),
                Binding("left", "screen.view('overview')", show=False),
                Binding("right", "screen.view('differences')", show=False)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.view, self.spans = "overview", []

    def render(self):
        text, x, self.spans = Text(" "), 1, []
        for i, (vid, label) in enumerate(VIEWS):
            chip = f" {i + 1} {label} "
            on = vid == self.view
            style = (f"{theme.color('on-accent')} on {theme.color('accent')} bold" if on and self.has_focus else
                     f"{theme.color('accent')} on {theme.color('cursor-blurred')} bold" if on else
                     theme.style("text-muted"))
            text.append(chip, style)
            text.append("  ")
            self.spans.append((x, x + len(chip), vid))
            x += len(chip) + 2
        return text

    def on_click(self, event):
        hit = next((v for x0, x1, v in self.spans if x0 <= event.x < x1), None)
        if hit:
            self.screen.action_view(hit)

    def on_focus(self):
        self.refresh()

    def on_blur(self):
        self.refresh()


class ReviewBatchScreen(Screen):
    """Review mode: enter on the queue reads a tree's details, e edits it (the Model Builder says "Editing"). Tab moves
    between the panels; esc goes back one, from the queue to the Model Builder."""
    crumbs = ("Review Batch",)
    BINDINGS = [Binding("escape", "back", "back"), Binding("f2", "rename_batch", "rename"),
                Binding("l", "launch", "launch"),
                Binding("alt+left", "move(-1)", show=False),
                Binding("alt+right", "move(1)", show=False),
                Binding("tab", "local(1)", show=False), Binding("shift+tab", "local(-1)", show=False)]

    def __init__(self):
        super().__init__()
        self.selected = None     # id of the tree whose details show (follows the queue cursor)
        self.view = "overview"
        self.tree_issues, self.batch_issues = {}, []
        self.dataset_issues = None   # None while the dataset check runs
        self.reference_id = None     # the reference last shown (to say when it changed)
        self.shown_key = None        # (view, tree id) of the details list
        self.renaming = None         # focus to return to after renaming the batch; None: not renaming
        self.first = True
        self.last_index = 0
        self.launching = False   # a launch is preparing / starting (no second one)
        self.grab = None         # (tree id, order of ids before it was picked up) while reordering

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Vertical(id="rv"):
            with Vertical(id="rv-head"):
                with Horizontal(id="rv-name-row"):
                    yield Static(id="rv-name")
                    yield Input(id="rv-rename", placeholder="Batch name", compact=True)
                yield Static(id="rv-info")
                yield Static(id="rv-state")
            with Horizontal(id="rv-body"):
                with Vertical(id="rv-left"):
                    yield Queue(id="rv-queue")
                    with Horizontal(id="rv-moves"):
                        yield Button("Reorder", id="rv-grab", compact=True)
                        yield Button("Details", id="rv-details-open", compact=True)
                        yield Button("Edit", id="rv-edit", compact=True)
                    yield Static(id="rv-reason")
                    yield Button("Launch Batch", id="rv-launch", variant="primary", disabled=True)
                with Vertical(id="rv-details"):
                    yield ViewTabs(id="rv-tabs")
                    yield DetailsList(id="rv-list")
        yield ContextFooter()

    def on_mount(self):
        self.query_one("#rv-head").border_title = "Review Batch"
        self.query_one("#rv-left").border_title = "Execution Queue"
        self.query_one("#rv-rename").display = False
        for b in ("#rv-grab", "#rv-details-open", "#rv-edit"):
            self.query_one(b).can_focus = False  # mouse only
        self.selected = self.draft.tree if any(t.id == self.draft.tree for t in self.draft.trees) else \
            self.draft.trees[0].id
        self.revalidate()
        self.query_one(Queue).focus()

    def on_screen_resume(self):
        """Shown again (after an editor): everything is checked afresh; selection, view and scroll are kept."""
        if self.first:
            self.first = False
        else:
            self.revalidate()

    @property
    def draft(self):
        return self.app.draft

    def tree(self, tree_id=None):
        tree_id = tree_id or self.selected
        return next((t for t in self.draft.trees if t.id == tree_id), None)

    # --- validation -------------------------------------------------------------------------------------------------
    def revalidate(self):
        """Trees and batch name now (cheap, in memory); the shared dataset in a worker (bounded file reads)."""
        draft = self.draft
        if self.tree() is None:  # the selected tree is gone: its neighbour
            i = min(self.last_index, len(draft.trees) - 1)
            self.selected = draft.trees[i].id
        self.tree_issues, self.batch_issues = review.tree_issues(draft), review.batch_issues(draft)
        ref = draft.reference
        if self.reference_id and ref and ref.id != self.reference_id:
            self.app.say(f"Reference tree changed: {ref.name}", screen=self)
        self.reference_id = ref.id if ref else None
        self.dataset_issues = None
        self.check_dataset(draft.confirmed)
        self.fill()

    @work(thread=True, exclusive=True, group="rv-dataset")
    def check_dataset(self, selection):
        self.post_message(DatasetChecked(selection, review.dataset_issues(self.app.root, selection)))

    def on_dataset_checked(self, message):
        if message.selection == self.draft.confirmed:  # a newer confirmation owns the answer otherwise
            self.dataset_issues = message.issues
            self.fill()

    def shared(self):
        """Issues of the batch itself: its name and the shared dataset (not repeated per tree)."""
        return [*self.batch_issues, *(self.dataset_issues or [])]

    def every(self):
        return [*self.shared(), *(i for t in self.draft.trees for i in self.tree_issues.get(t.id, ()))]

    # --- drawing ------------------------------------------------------------------------------------------------------
    def fill(self):
        self.fill_head()
        self.fill_queue()
        self.fill_details()

    def fill_head(self):
        draft, c = self.draft, self.draft.confirmed
        self.query_one("#rv-name", Static).update(Text.assemble(
            ("Batch  ", theme.style("text-muted")), (draft.name, "bold"), ("   F2 rename", theme.style("text-muted"))))
        n = len(draft.trees)
        if c is None:
            data = "No dataset confirmed"
        else:
            data = (f"{c.name} — {c.variant}" if c.variant else c.name) + \
                f" · Training: {Path(c.train).name if c.train else '—'}" + \
                f" · Evaluation: {', '.join(e.split for e in c.enabled()) or 'none'}"
        self.query_one("#rv-info", Static).update(
            f"{data} · {n} tree{'s' * (n != 1)} · On tree failure: continue")
        found = self.every()
        errors, cautions = len(review.blocking(found)), len(found) - len(review.blocking(found))
        if self.launching:
            word, style = "Preparing the batch: hashing inputs, starting the coordinator…", "text-muted"
        elif self.dataset_issues is None:
            word, style = "Validating the shared dataset…", "text-muted"
        elif errors:
            word, style = f"Not ready · {errors} blocking issue{'s' * (errors != 1)}", "error"
        elif cautions:
            word, style = f"Ready · {cautions} caution{'s' * (cautions != 1)}", "warning"
        else:
            word, style = "Ready", "success"
        self.query_one("#rv-state", Static).update(Text.assemble((word, theme.style(style, "bold"))))
        reason = self.launch_blocker()
        self.query_one("#rv-launch", Button).disabled = reason is not None
        ok = reason is None
        self.query_one("#rv-reason", Static).update(Text(
            "Ready to launch." if ok else reason, style=theme.style("success" if ok else "warning")))

    def launch_blocker(self):
        """Why Launch Batch is unavailable now (None: it is available). Cautions never block."""
        errors = len(review.blocking(self.every()))
        if self.launching:
            return "Preparing batch execution…"
        if self.dataset_issues is None:
            return "Validating the shared dataset…"
        if self.grab is not None:
            return "Drop the moved tree first."
        if errors:
            return f"{errors} configuration issue{'s' * (errors != 1)} must be fixed."
        return None

    def tree_status(self, tree):
        found = self.tree_issues.get(tree.id, [])
        errors = len(review.blocking(found))
        if errors:
            return review.INVALID_WORD, f"Invalid · {errors} issue{'s' * (errors != 1)}"
        if found:
            return review.CAUTION_WORD, f"Caution · {len(found)}"
        return review.READY_WORD, "Ready"

    def queue_row(self, i, tree, width):
        fields = components.fields_of(tree.config, components.slot("hierarchy"))
        summary = f"Depth {fields['depth']} · {fields['n_clusters']} clusters" if fields else "no hierarchy"
        kind, status = self.tree_status(tree)
        here = tree.id == self.selected
        moving = self.grab is not None and self.grab[0] == tree.id
        lines = [Text.assemble(("=" if moving else ">" if here else " ", theme.style("accent", "bold")),
                               f" {i + 1:02d}  ", (tree.name, theme.style("accent", "bold") if moving else
                                                    "bold" if here else "")),
                 Text("      " + summary, theme.style("text-muted")),
                 Text("      " + ("Moving: up / down, space drops" if moving else status),
                      theme.style("accent" if moving else STATUS_STYLE[kind]))]
        for line in lines:
            line.truncate(width, overflow="ellipsis")
        return Text("\n").join(lines)

    def fill_queue(self):
        box = self.query_one(Queue)
        width = box.scrollable_content_region.width or 30
        scroll = box.scroll_y
        box.set_options([Option(self.queue_row(i, t, width), id=t.id) for i, t in enumerate(self.draft.trees)])
        ids = [t.id for t in self.draft.trees]
        box.highlighted = ids.index(self.selected)
        self.last_index = ids.index(self.selected)
        self.call_after_refresh(box.scroll_to, y=scroll, animate=False)

    def fill_details(self):
        """Dataset / batch problems once, then the selected tree's view. The same view of the same tree keeps its
        cursor row and scroll across a refresh."""
        tree, box, tabs = self.tree(), self.query_one(DetailsList), self.query_one(ViewTabs)
        tabs.view = self.view
        tabs.refresh()
        self.query_one("#rv-details").border_title = f"Review Details · {tree.name}"
        issues = self.tree_issues.get(tree.id, [])
        lines = review_details.dataset(self.shared())
        if self.view == "overview":
            lines += review_details.overview(tree, issues)
        else:
            ref = self.draft.reference
            lines += review_details.differences(tree, ref, review.diff(ref.config, tree.config),
                                                box.scrollable_content_region.width or 80)
            fixes = [i for i in issues if i.severity == review.ERROR]
            if fixes:
                lines += [review_details.blank(), review_details.head("Issues"),
                          *(review_details.issue_line(i) for i in fixes)]
        key, old, scroll = (self.view, tree.id), box.highlighted, box.scroll_y
        box.clear_options()
        box.forget_lines()
        box.add_options([Option(line.text, id=f"fix:{line.issue}" if line.issue else None, disabled=line.head)
                         for line in lines])
        rows = [i for i, line in enumerate(lines) if not line.head]
        first_fix = next((i for i in rows if lines[i].issue), None)
        same = key == self.shown_key and old in rows
        box.highlighted = old if same else first_fix if first_fix is not None else rows[0] if rows else None
        self.shown_key = key
        self.call_after_refresh(box.scroll_to, y=scroll if same else 0, animate=False)

    def on_resize(self, _):
        if self.draft.trees:
            self.fill()

    # --- selection and view ---------------------------------------------------------------------------------------------
    def on_option_list_option_highlighted(self, event):
        if event.option_list.id == "rv-queue" and event.option and event.option.id != self.selected:
            self.selected = event.option.id
            self.fill_queue()
            self.fill_details()

    def on_option_list_option_selected(self, event):
        if event.option_list.id == "rv-queue":
            self.action_focus_details()
        elif event.option.id and event.option.id.startswith("fix:"):
            issue = next((i for i in self.every() if i.id == event.option.id[4:]), None)
            if issue is not None:
                self.fix(issue)

    def action_focus_details(self):
        if self.grab is not None:
            self.app.say("Space drops the tree first (esc cancels).", screen=self)
        else:
            self.query_one(DetailsList).focus()

    def action_edit(self):
        if self.grab is None:
            self.open_tree(self.selected)

    def action_view(self, view):
        if view != self.view:
            self.view = view
            self.fill_details()

    def action_other_view(self):
        self.action_view("differences" if self.view == "overview" else "overview")

    # --- queue actions -----------------------------------------------------------------------------------------------------
    def action_grab(self):
        """Space / Reorder: pick the selected tree up (up / down carry it through the queue, esc cancels) or, while carrying,
        drop it here."""
        if self.grab is not None:
            self.end_grab(True)
        elif len(self.draft.trees) > 1:
            self.grab = (self.selected, [t.id for t in self.draft.trees])
            self.query_one("#rv-left").border_title = "Execution Queue · reordering"
            self.query_one("#rv-grab", Button).label = "Drop"
            self.fill_queue()
            self.app.say(f"Moving {self.tree().name}: up / down place it, space drops it, esc cancels.", screen=self)
        elif self.grab is None:
            self.app.say("A single tree has no order to change.", screen=self)

    def carry(self, step):
        self.action_move(step)

    def end_grab(self, keep):
        """Drop (keep the order) or cancel (back to the order before the pick-up)."""
        _, before = self.grab
        self.grab = None
        if not keep:
            self.draft.trees.sort(key=lambda t: before.index(t.id))
        self.query_one("#rv-left").border_title = "Execution Queue"
        self.query_one("#rv-grab", Button).label = "Reorder"
        self.fill_queue()
        self.app.say(f"{self.tree().name} dropped at position {self.draft.index(self.selected) + 1}." if keep else
                     "Reorder cancelled: the order is as it was.", screen=self)

    def action_move(self, step):
        tree = self.tree()
        if self.draft.move(tree.id, step):
            self.fill_queue()
            self.app.say(f"{tree.name} moved {'earlier' if step < 0 else 'later'}: position "
                         f"{self.draft.index(tree.id) + 1} of {len(self.draft.trees)}", screen=self)
        else:
            self.app.say(f"{tree.name} is already {'first' if step < 0 else 'last'} in the queue.", screen=self)

    def on_button_pressed(self, event):
        {"rv-launch": self.action_launch, "rv-grab": self.action_grab, "rv-details-open": self.action_focus_details,
         "rv-edit": self.action_edit}.get(event.button.id, lambda: None)()

    def open_tree(self, tree_id):
        """The tree in the Model Builder (grid), over this screen; esc comes back here."""
        self.app.open_builder(review.Issue("open", review.CAUTION, "tree", "", tree=tree_id))

    def fix(self, issue):
        """Fix Issue: the existing editor that owns the setting, over this screen."""
        if issue.dest == "dataset":
            self.app.open_setup(issue)
        elif issue.dest == "batch-name":
            self.action_rename_batch()
        else:
            self.app.open_builder(issue)

    def action_launch(self):
        """Launch Batch: refused while validating, preparing or invalid; otherwise the footer asks and a second `l` (or
        click) sends a copy of the draft to the launcher in a worker. Tree and batch-name issues are rechecked now, the dataset in the worker."""
        if self.launching:
            self.app.say("A launch is already in progress.", "warning", self)
            return
        if self.grab is not None:
            self.app.say(self.launch_blocker(), "warning", self)
            return
        self.revalidate_trees()
        found = review.blocking(self.every())
        if self.dataset_issues is None:
            self.app.say("Still validating the shared dataset.", "warning", self)
        elif found:
            self.app.say(f"Cannot launch an invalid batch: {len(found)} blocking issue{'s' * (len(found) != 1)}.",
                         "error", self)
        elif not self.app.asked("l"):
            n = len(self.draft.trees)
            self.app.ask(f"Launch {n} tree{'s' * (n != 1)}?", "l", "launch")
        else:
            self.launching = True
            self.fill_head()
            self.start_batch(batch.with_id(review.plan(self.draft), batch.new_id(self.draft.id)))

    def revalidate_trees(self):
        self.tree_issues, self.batch_issues = review.tree_issues(self.draft), review.batch_issues(self.draft)

    @work(thread=True, group="rv-launch")
    def start_batch(self, plan_):
        issues = review.blocking(review.dataset_issues(self.app.root, plan_.dataset))  # the files as they are now
        if issues:
            self.post_message(Launched(error=f"The shared dataset changed: {issues[0].message}"))
            return
        try:
            bdir, child = batch.launch(self.app.root, plan_)
        except batch.BatchError as err:
            self.post_message(Launched(error=str(err)))
        else:
            self.post_message(Launched(bdir, child))

    def on_launched(self, message):
        self.launching = False
        if message.error:
            self.revalidate()
            self.app.say(f"Batch not started: {message.error}", "error", self)
            return
        self.app.coordinators.append(message.child)
        self.fill_head()
        self.app.rescan()
        self.app.say(f"Batch started: {self.draft.name}", "success", self)
        self.app.push_screen(BatchMonitorScreen(message.bdir.name))

    # --- batch name ---------------------------------------------------------------------------------------------------------
    def action_rename_batch(self):
        box = self.query_one("#rv-rename", Input)
        if self.renaming is not None:
            return
        self.renaming = self.focused or self.query_one(Queue)
        box.value = self.draft.name
        box.display = True
        self.query_one("#rv-name").display = False
        box.focus()

    def on_input_submitted(self, event):
        if event.input.id != "rv-rename":
            return
        problem = self.draft.rename_batch(event.value)
        if problem:
            self.app.say(f"Not renamed: {problem}", "error", self)
            return
        self.end_rename()
        self.fill_head()
        self.app.say(f"Batch renamed to {self.draft.name}", "success", self)

    def end_rename(self):
        back, self.renaming = self.renaming, None
        self.query_one("#rv-rename").display = False
        self.query_one("#rv-name").display = True
        (back if back is not None and back.is_attached else self.query_one(Queue)).focus()

    # --- navigation -----------------------------------------------------------------------------------------------------------
    def action_back(self):
        """Rename -> cancelled (the name is kept); Review Details -> queue; queue -> the previous screen."""
        if self.grab is not None:
            self.end_grab(False)
        elif self.renaming is not None:
            self.end_rename()
            self.app.say("Rename cancelled: the batch name is unchanged.", screen=self)
        elif self.focused in (self.query_one(DetailsList), self.query_one(ViewTabs)):
            self.query_one(Queue).focus()
        elif self.app.leave(self, "Review Batch"):
            self.app.pop_screen()

    def action_local(self, step):
        """Tab / shift+tab cycle inside the focused panel: the queue and Launch Batch, or Review Details' tabs and list.
        Enter and esc are the only ways between the panels; the Reorder / Details / Edit buttons are for the mouse."""
        if self.renaming is not None or self.grab is not None:
            return
        queue, launch = self.query_one(Queue), self.query_one("#rv-launch", Button)
        if self.focused in (queue, launch):
            (queue if self.focused is launch or launch.disabled else launch).focus()
            return
        members = [self.query_one(ViewTabs), self.query_one(DetailsList)]
        if self.focused in members:
            members[(members.index(self.focused) + step) % len(members)].focus()
