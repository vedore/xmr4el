"""New Experiment, step 1: Dataset Setup (tui.md §5.4). Datasets (30 %) | Dataset Configuration (70 %).

Datasets lists what `datasets.discover` recognizes plus the user's registered datasets; moving the cursor changes
nothing, enter (or a click) shows one in the configuration. Every change there is an edit kept in the app's draft
(`app.draft`, per dataset, for the session); only Confirm Dataset (ctrl+s) validates it and makes it the draft's
confirmed dataset, which later New Experiment screens read. Leaving or switching datasets discards nothing.
Writes only the registry file (outputs/tui/datasets.json), and only when a registered dataset is confirmed.
"""
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.widgets.option_list import Option

from xmr4el_tui import datasets, theme
from xmr4el_tui.datasets import READY, SPLITS, Dataset, Selection
from xmr4el_tui.screens.file_picker import FilePicker
from xmr4el_tui.screens.model_builder import ModelBuilderScreen
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.description import Rows
from xmr4el_tui.widgets.top_bar import TopBar

NEW = "new"  # key of the dataset being registered (a registry id is given when it is confirmed)
STATUS_STYLE = {READY: "success", datasets.MISSING: "error", datasets.INVALID: "error", datasets.REVIEW: "warning"}
FILE_WORD = {None: "Ready", "missing": "Not found", "invalid": "Invalid"}
LABEL = 13  # cells of the split / field label column


class StatusesChecked(Message):
    def __init__(self, statuses):
        super().__init__()
        self.statuses = statuses


class DatasetList(Rows):
    """Datasets: up / down move the cursor only; enter or a click configures the dataset under it."""
    BINDINGS = [Binding("enter", "select", "configure")]


class ConfigRows(Rows):
    """The selected dataset's files: tab / shift+tab (or up / down) move between controls, enter activates (a file
    opens the picker), space selects a training variant or toggles an evaluation split. A click on [x] toggles."""
    BINDINGS = [Binding("enter", "select", "choose"), Binding("space", "screen.toggle", "toggle"),
                Binding("tab", "step(1)", show=False), Binding("shift+tab", "step(-1)", show=False)]

    def action_step(self, step):
        """Next / previous control; past the last (first) one, focus leaves the panel as tab does elsewhere."""
        rows = [i for i, o in enumerate(self.options) if not o.disabled]
        here = self.highlighted if self.highlighted in rows else None
        nxt = [i for i in rows if (i > here if step > 0 else i < here)] if here is not None else rows
        if not nxt:
            self.screen.focus_next() if step > 0 else self.screen.focus_previous()
        else:
            self.highlighted = nxt[0] if step > 0 else nxt[-1]

    async def _on_click(self, event):
        index = event.style.meta.get("option")
        option = self.options[index] if index is not None else None
        if option is not None and option.id and option.id.startswith("eval:") and event.x < 4:
            self.highlighted = index
            self.screen.action_toggle()
            return
        await super()._on_click(event)


class DatasetSetupScreen(Screen):
    """Hidden keys (in the ? panel): ctrl+tab / ctrl+shift+tab (or ] / [) cycle panels, ctrl+r rereads datasets."""
    crumbs = ("New Experiment",)
    BINDINGS = [Binding("escape", "back", "back"), Binding("ctrl+s", "confirm", "confirm"),
                Binding("u", "discard", "discard"), Binding("m", "builder", "builder"),
                Binding("ctrl+tab,right_square_bracket", "panel(1)", "next panel", show=False),
                Binding("ctrl+shift+tab,left_square_bracket", "panel(-1)", "previous panel", show=False),
                Binding("ctrl+r", "reload", "reread datasets", show=False)]

    def __init__(self, fix=None):
        super().__init__()
        self.fix = fix  # review.Issue (Review Batch's Fix Issue): this screen opened over Review Batch to correct it
        self.datasets, self.statuses, self.problem, self.variants = [], {}, None, []

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="ds"):
            yield DatasetList(id="ds-list")
            with Vertical(id="ds-config"):
                with Horizontal(id="ds-name-row"):
                    yield Static("Name", id="ds-name-label")
                    yield Input(placeholder="Dataset name", id="ds-name")
                yield ConfigRows(id="ds-rows")
        yield ContextFooter()

    def on_mount(self):
        self.query_one("#ds-list").border_title = "Datasets"
        self.query_one("#ds-config").border_title = "Dataset Configuration"
        self.reload()
        listed = self.query_one(DatasetList)
        if self.draft.selected in self.keys():
            listed.highlighted = self.keys().index(self.draft.selected)
        listed.focus()
        if self.fix is not None:
            self.start_fix()

    def start_fix(self):
        """Show the confirmed dataset with the focus on the mapping that needs correcting and say what is wrong. The
        correction goes through the normal picker and Confirm Dataset; esc returns to Review Batch."""
        c = self.draft.confirmed
        if c is not None and c.key in self.keys():
            self.configure(c.key)
            if self.fix.target:
                self.go_to_problem(self.fix.target)
        self.app.say(self.fix.message, "warning", self)

    @property
    def draft(self):
        return self.app.draft

    # --- datasets ---------------------------------------------------------------------------------------------
    def keys(self):
        return [d.key for d in self.datasets] + [NEW]

    def dataset(self, key):
        if key == NEW:
            return Dataset(NEW, "", registered=True)
        return next((d for d in self.datasets if d.key == key), None)

    def reload(self):
        """Datasets reread (specs, directories, registry); statuses checked in a worker (bounded file reads)."""
        root = self.app.root
        local, self.problem = datasets.registered(root)
        self.datasets = datasets.discover(root) + local
        if self.problem:
            self.app.say(self.problem, "warning", self)
        if self.draft.selected not in self.keys():
            self.draft.selected = None
        self.fill_list()
        self.fill_config()
        self.check_all([(d, self.draft.current(d), self.taken(d)) for d in self.datasets])

    @work(thread=True, exclusive=True, group="ds-status")
    def check_all(self, items):
        root = self.app.root
        self.post_message(StatusesChecked({d.key: datasets.status(datasets.validate(root, d, sel, taken))
                                           for d, sel, taken in items}))

    def on_statuses_checked(self, message):
        self.statuses.update(message.statuses)
        self.fill_list()

    def taken(self, dataset):
        return [d.name for d in self.datasets if d.registered and d.key != dataset.key]

    # --- left panel -------------------------------------------------------------------------------------------
    def fill_list(self):
        box = self.query_one(DatasetList)
        keep = box.highlighted
        width = box.scrollable_content_region.width or 40
        options = [Option(self.list_row(d, width), id=d.key) for d in self.datasets]
        register = Text("Register local dataset", theme.style("accent"))
        register.truncate(width, overflow="ellipsis")
        options.append(Option(register, id=NEW))
        box.set_options(options)
        box.highlighted = min(keep, len(options) - 1) if keep is not None else 0

    def list_row(self, d, width):
        """Name and variant; then status, `confirmed` and `edited`. `>` marks the dataset being configured."""
        configured = d.key == self.draft.selected
        status = self.statuses.get(d.key)
        line = Text.assemble((">" if configured else " ", theme.style("accent", "bold")), " ",
                             (d.name, "bold" if configured else ""),
                             (f" — {d.variant}" if d.variant else "", theme.style("text-muted")),
                             ("  local" if d.registered else "", theme.style("text-muted")), "\n   ",
                             (status or "Checking…", theme.style(STATUS_STYLE.get(status, "text-muted"))))
        if self.draft.confirmed and self.draft.confirmed.key == d.key:
            line.append(" · confirmed", theme.style("accent"))
        if self.draft.changed(d.key):
            line.append(" · edited", theme.style("warning"))
        parts = line.split("\n")
        for part in parts:  # OptionList wraps prompts: each line cut to the panel
            part.truncate(width, overflow="ellipsis")
        return Text("\n").join(parts)

    def on_option_list_option_selected(self, event):
        if event.option_list.id == "ds-list":
            self.configure(event.option.id)
        elif event.option_list.id == "ds-rows":
            self.activate(event.option.id)

    def configure(self, key):
        """Show `key` in Dataset Configuration (the draft's confirmed dataset is not changed)."""
        self.draft.selected = key
        self.fill_config()
        self.fill_list()
        self.query_one(ConfigRows).highlighted = None
        self.query_one("#ds-name" if key == NEW else ConfigRows).focus()
        if key != NEW:
            self.query_one(ConfigRows).action_step(1)

    # --- right panel ------------------------------------------------------------------------------------------
    def shown(self):
        """(dataset, its current selection) being configured, or (None, None)."""
        d = self.dataset(self.draft.selected) if self.draft.selected else None
        return (d, self.draft.current(d)) if d else (None, None)

    def fill_config(self):
        d, sel = self.shown()
        box, name_row = self.query_one(ConfigRows), self.query_one("#ds-name-row")
        name_row.display = bool(d and d.registered)
        if d is None:
            box.set_options([Option(Text("Select a dataset: enter on it in Datasets.", theme.style("text-muted")),
                                    disabled=True)])
            return
        name = self.query_one("#ds-name", Input)
        if d.registered and name.value != sel.name:
            with name.prevent(Input.Changed):
                name.value = sel.name
        problems = datasets.validate(self.app.root, d, sel, self.taken(d))
        self.statuses[d.key] = datasets.status(problems)
        width = max(30, box.scrollable_content_region.width or 80)
        keep = box.highlighted_option.id if box.highlighted_option else None
        opts = self.info(d, sel, problems, width)
        opts += [blank(), head("Training file")]
        variants = [c.path for c in d.train] + ([sel.train] if sel.train and sel.train not in
                                                 [c.path for c in d.train] else [])
        for i, path in enumerate(variants):
            mark = "(x)" if path == sel.train else "( )"
            opts.append(Option(self.file_row(mark, "", path, width), id=f"train:{i}"))
        opts.append(Option(Text("    Choose another file…", theme.style("text-muted")), id="train:other"))
        opts += errors(problems, "train", width)
        opts += [blank(), head("Evaluation files")]
        for split, label in SPLITS.items():
            e = sel.evaluation(split)
            opts.append(Option(self.file_row("[x]" if e.enabled else "[ ]", label, e.path, width), id=f"eval:{split}"))
            opts += errors(problems, split, width)
        opts += errors(problems, "evals", width)
        opts += [blank(), Option(Text("  Confirm Dataset  ", theme.style("accent", "bold")), id="confirm"),
                 Option(Text("  Model Builder  ", theme.style("accent" if self.draft.confirmed else "text-muted")),
                        id="builder")]
        box.set_options(opts)
        ids = [o.id for o in opts]
        if keep in ids:
            box.highlighted = ids.index(keep)
        self.variants = variants

    def info(self, d, sel, problems, width):
        st = datasets.status(problems)
        c = self.draft.confirmed
        state = ("Confirmed · changed since, not confirmed" if self.draft.changed(d.key) else "Confirmed") \
            if c and c.key == d.key else "Not confirmed"
        sources = ["datasets/ scan" if s == "directory" else s for s in d.sources()]
        src = ", ".join(sources[:3]) + (f", +{len(sources) - 3} more" if len(sources) > 3 else "")
        rows = [Option(Text(d.title, "bold"), disabled=True),
                field("Variant", d.variant or "—"), field("Format", "PubTator (the experiment runner's input)"),
                field("Directory", d.key if not d.registered else "registered (outputs/tui/datasets.json)"
                      if d.key != NEW else "—")]
        if sources and not d.registered:
            rows.append(field("Recognized", src))
        rows.append(field("Status", Text.assemble((st, theme.style(STATUS_STYLE[st])), "  ·  ",
                                                  (state, theme.style("accent" if state == "Confirmed" else
                                                                      "warning" if "changed" in state else
                                                                      "text-muted")))))
        return rows + errors(problems, "name", width)

    def file_row(self, mark, label, path, width):
        kind = datasets.inspect(self.app.root, path)[0] if path else None
        word = FILE_WORD[kind] if path else "Not set"
        style = {"Ready": "success", "Not set": "text-muted"}.get(word, "error")
        room = max(10, width - 4 - (LABEL if label else 0) - 12)
        text = path or "—"
        text = text if len(text) <= room else "…" + text[-(room - 1):]  # keep the file end of deep paths
        line = Text.assemble((mark, theme.style("accent")), " ", (f"{label:<{LABEL}}" if label else "", "bold"),
                             (f"{text:<{room}}", "" if path else theme.style("text-muted")), "  ",
                             (word, theme.style(style)))
        line.no_wrap = True
        return line

    def on_resize(self, _):
        self.fill_list()
        self.fill_config()

    # --- editing ----------------------------------------------------------------------------------------------
    def edit(self, sel):
        d, _ = self.shown()
        self.draft.edit(d, sel)
        self.fill_config()
        self.fill_list()

    def on_input_changed(self, event):
        d, sel = self.shown()
        if event.input.id == "ds-name" and d is not None and d.registered:
            self.edit(Selection(sel.key, event.value, sel.variant, sel.train, sel.evals, sel.format))

    def on_input_submitted(self, event):
        if event.input.id == "ds-name":
            self.query_one(ConfigRows).focus()

    def activate(self, rid):
        """enter on a control: a file row opens the picker (training variant rows choose that variant), the
        confirm row confirms."""
        d, sel = self.shown()
        if rid == "confirm":
            self.action_confirm()
        elif rid == "builder":
            self.action_builder()
        elif rid and rid.startswith("train:") and rid != "train:other":
            self.edit(replace_train(sel, self.variants[int(rid[6:])]))
        elif rid == "train:other":
            self.pick("Training file", sel.train, lambda p: self.edit(replace_train(self.shown()[1], p)))
        elif rid and rid.startswith("eval:"):
            split = rid[5:]
            e = sel.evaluation(split)
            current = e.path or (d.candidates(split)[0].path if d.candidates(split) else sel.train)
            self.pick(f"{SPLITS[split]} file", current,
                      lambda p: self.edit(self.shown()[1].with_eval(split, path=p, enabled=True)))

    def pick(self, title, current, done):
        def back(path):
            if path is not None:
                done(path)
                self.app.say(f"{title}: {path}", screen=self)
        self.app.push_screen(FilePicker(self.app.root, title, current), back)

    def action_toggle(self):
        box = self.query_one(ConfigRows)
        rid = box.highlighted_option.id if self.focused is box and box.highlighted_option else None
        _, sel = self.shown()
        if rid and rid.startswith("eval:"):
            split = rid[5:]
            self.edit(sel.with_eval(split, enabled=not sel.evaluation(split).enabled))
        elif rid and rid.startswith("train:") and rid != "train:other":
            self.activate(rid)

    def action_confirm(self):
        """Validate the configured dataset; all checks pass: it becomes the draft's confirmed dataset (a registered
        one is saved to the registry first). Else the footer names the first problem; the panel lists all."""
        d, sel = self.shown()
        if d is None:
            self.app.say("Select a dataset first: enter on it in Datasets.", "warning")
            return
        problems = datasets.validate(self.app.root, d, sel, self.taken(d))
        if problems:
            self.app.say(f"Not confirmed: {problems[0][2]}", "error")
            self.go_to_problem(problems[0][0])
            return
        if d.registered:
            if self.problem:
                self.app.say(self.problem, "error")
                return
            try:
                key = datasets.register(self.app.root, sel)
            except (OSError, ValueError) as err:
                self.app.say(f"Not confirmed: registry not written ({err})", "error")
                return
            sel = Selection(key, sel.name.strip(), "", sel.train, sel.evals, sel.format)
        self.draft.confirm(sel, old_key=d.key)
        if d.registered:
            self.reload()
        self.fill_list()
        self.fill_config()
        self.app.say(f"Dataset confirmed: {sel.name}" + (f" — {sel.variant}" if sel.variant else "") +
                     (". m returns to Review Batch." if self.fix is not None else ". m opens the Model Builder."),
                     "success")

    def action_builder(self):
        """Model Builder (next step) over this screen; esc there comes back here. Needs a confirmed dataset."""
        if self.fix is not None:  # opened from Review Batch: back to it
            self.app.pop_screen()
            return
        if self.draft.confirmed is None:
            self.app.say("Confirm a dataset first (ctrl+s).", "warning")
            return
        d, _ = self.shown()
        if d is not None and d.key != self.draft.confirmed.key:
            self.app.say(f"Confirm {d.title} to continue: this experiment uses {self.draft.confirmed.name}.", "warning")
            return
        if self.draft.edits:
            self.app.say("Unconfirmed dataset changes are not used: the Model Builder uses the confirmed dataset.")
        self.app.push_screen(ModelBuilderScreen())

    def go_to_problem(self, which):
        """Focus the control that corrects `which` (name field, training file, evaluation split)."""
        if which == "name":
            self.query_one("#ds-name").focus()
            return
        box, ids = self.query_one(ConfigRows), [o.id for o in self.query_one(ConfigRows).options]
        rid = {"train": "train:other", "evals": "eval:dev"}.get(which, f"eval:{which}")
        box.focus()
        if rid in ids:
            box.highlighted = ids.index(rid)

    def action_discard(self):
        """Unconfirmed edits of the configured dataset back to its confirmed (or recognized) mapping; asks first."""
        d, _ = self.shown()
        if d is None or not self.draft.changed(d.key):
            self.app.say("No unconfirmed changes.")
        elif self.app.asked("u"):
            self.draft.discard(d.key)
            self.fill_config()
            self.fill_list()
            self.app.say("Changes discarded.")
        else:
            self.app.ask(f"Discard unconfirmed changes to {d.title}?", "u", "discard")

    # --- navigation -------------------------------------------------------------------------------------------
    def panels(self):
        return [w for w in (self.query_one(DatasetList), self.query_one("#ds-name"), self.query_one(ConfigRows))
                if w.display and w.parent.display]

    def action_panel(self, step):
        panels = self.panels()
        i = panels.index(self.focused) if self.focused in panels else -step
        panels[(i + step) % len(panels)].focus()

    def action_back(self):
        """From the configuration: to Datasets. From Datasets: back to the previous screen; unconfirmed edits stay
        in the draft (said on the screen below)."""
        if self.focused is not self.query_one(DatasetList):
            self.query_one(DatasetList).focus()
            return
        if not self.app.leave(self, "Dataset Setup"):
            return
        edited = len(self.draft.edits)
        self.app.pop_screen()
        if edited:
            self.app.say(f"New Experiment: {edited} dataset{'s' * (edited > 1)} with unconfirmed changes kept "
                         "for this session.")

    def action_reload(self):
        self.reload()
        self.app.say(f"Reread datasets: {len(self.datasets)} found")


def replace_train(sel, path):
    return Selection(sel.key, sel.name, sel.variant, path, sel.evals, sel.format)


def blank():
    return Option(Text(""), disabled=True)


def head(text):
    return Option(Text(text, "bold"), disabled=True)


def field(label, value):
    return Option(Text.assemble((f"{label:<{LABEL}}", theme.style("text-muted")), value), disabled=True)


def errors(problems, which, width):
    """The problems of one field, under it: the specific message (missing path, wrong format, ...)."""
    out = []
    for f, kind, message in problems:
        if f == which:
            t = Text("    " + message, theme.style("warning" if kind == "review" else "error"))
            t.truncate(width, overflow="ellipsis")  # OptionList wraps prompts: cut to one row
            out.append(Option(t, disabled=True))
    return out
