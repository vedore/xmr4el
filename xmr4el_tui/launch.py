"""Setup and Review screens (tui.md §5.5-5.7): a form for one input spec, the runner's -check in the background,
Review of the exact validated snapshot, spawn. Files are picked from what exists (configs/*.json, datasets/<dir>);
the runner stays the authority."""
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Footer, Input, Label, OptionList, Select, Static

from xmr4el_tui import fmt, process, runs
from xmr4el_tui.runs import RUNS_DIR, SPLIT_FILE
from xmr4el_tui.widgets import TopBar

SEARCH_TYPES = {"beam_size": int, "leaf_candidates": int, "topk": int, "knn_beta": float}
HELP = {
    "name": "**name**: run name, `[A-Za-z0-9_.-]+`; the run directory is `<timestamp>_<name>`.",
    "model_config": "**model_config**: model configuration JSON from `configs/` (features, clustering, matcher, "
                    "`predict_config`). ctrl+e opens it in `$VISUAL` / `$EDITOR`. ←/→ cycle, enter lists.",
    "dataset": "**dataset**: a directory under `datasets/` with `dev` / `test` files. ←/→ cycle, enter lists.",
    "train_file": "**training file**: the file of the dataset the tree trains on (`train_plus_ctd` = train + CTD "
                  "dictionary names).",
    "split": "**split**: the dataset file this evaluation reads (`<dataset>/<split>.pubtator`); the run writes "
             "`metrics_<split>.json`. ←/→ cycle.",
    "breakdown": "**breakdown**: seen/unseen mention-string breakdown against the tree's training file "
                 "(the evaluation's `train_path`).",
    "beam_size": "**beam_size**: clusters kept per layer during the search.",
    "leaf_candidates": "**leaf_candidates**: labels kept per reached leaf.",
    "topk": "**topk**: labels returned per mention; 0 keeps every candidate.",
    "knn_beta": "**knn_beta**: weight of the kNN score fusion; 0 turns it off.",
}


class Choice(Select, inherit_bindings=False):
    """Select that cycles with ←/→; enter/space open the list; ↑/↓ stay free to move between fields."""
    BINDINGS = [Binding("enter,space", "show_overlay", "list", show=False),
                Binding("left", "cycle(-1)", show=False), Binding("right", "cycle(1)", show=False)]

    def __init__(self, options, value=None, **kwargs):
        options = with_value(options, value)
        super().__init__(options, value=value or (options[0][1] if options else Select.NULL), allow_blank=not options,
                         prompt="none found", compact=True, **kwargs)
        self.values = [v for _, v in options]

    def set_choices(self, options, value):
        self.values = [v for _, v in options]
        self.set_options(options)
        if value in self.values:
            self.value = value

    def action_cycle(self, step):
        if self.values:
            i = self.values.index(self.value) if self.value in self.values else -1
            self.value = self.values[(i + step) % len(self.values)]


def with_value(options, value):
    """Discovered options plus a loaded value discovery did not find (it is never replaced silently)."""
    return options + ([(value, value)] if value and value not in [v for _, v in options] else [])


def split_of(path):
    """`dev` for datasets/x/dev.pubtator (or MedMentions corpus_pubtator_dev.txt), else the file stem."""
    m = SPLIT_FILE.fullmatch(Path(path).name)
    return m[m.lastindex] if m else Path(path).stem


def dataset_name(dataset):
    """datasets/BC5CDR/chemical -> bc5cdr_chemical (default run name)."""
    return "_".join(p for p in Path(dataset).parts[1:] if p != "data").lower()


def differences(a, b, where=""):
    """Paths where two JSON values differ (loaded spec vs form)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in {**a, **b} for d in differences(a.get(k), b.get(k), f"{where}.{k}".strip("."))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{where}[{i}]")]
    return [] if a == b else [where]


def row(label, widget, help_key, side=None):
    """Label, widget, side note; the form error sits under the row."""
    widget.HELP = HELP[help_key]
    yield Horizontal(Label(label), widget, *([side] if side is not None else []), classes="row")
    yield Static("", id=f"err-{widget.id}", classes="error")


class EvalBlock(Vertical):
    """One evaluation of the dataset. `n` keeps ids unique when blocks are removed. A loaded split label that is not
    the file's split (dev_knn0 on dev.pubtator) is kept until the split changes; keys the form does not know stay in
    `extra` so the runner rejects them (§10: never drop them silently)."""

    def __init__(self, n, e, splits, default, can_breakdown):
        super().__init__(classes="eval")
        e, given = dict(e), "test_path" in e or "split" in e
        self.n, self.splits, self.can_breakdown = n, splits, can_breakdown
        self.path = e.pop("test_path", None) or splits.get(default) or next(iter(splits.values()), None)
        label = e.pop("split", None)
        self.label = label if label and self.path and label != split_of(self.path) else None
        self.breakdown = bool(e.pop("train_path", None)) if given else can_breakdown
        self.search = {k: e.pop(k, None) for k in SEARCH_TYPES}
        self.extra = e

    def compose(self) -> ComposeResult:
        with Horizontal(classes="eval-head"):
            yield Label(self.heading(), id=f"head-{self.n}")
            yield Button("Remove", id=f"remove-{self.n}", compact=True)
        yield from row("split", Choice([(k, p) for k, p in self.splits.items()], self.path, id=f"e{self.n}-split"),
                       "split")
        yield from row("breakdown", Checkbox("mention strings vs training file", self.breakdown and self.can_breakdown,
                                             id=f"e{self.n}-breakdown", disabled=not self.can_breakdown,
                                             compact=True), "breakdown")
        for k, v in self.search.items():  # the placeholder shows the value an empty field inherits
            yield from row(k, Input("" if v is None else str(v), id=f"e{self.n}-{k}", compact=True), k)

    def heading(self):
        return f"Evaluation {self.split()}"

    def split(self):
        return self.label or (split_of(self.path) if self.path else "")


class SpecPicker(ModalScreen):
    """ctrl+o: experiment specs, then the recorded specs of runs (newest first); type to filter."""
    BINDINGS = [Binding("escape", "dismiss", "cancel"), Binding("down", "move(1)", show=False),
                Binding("up", "move(-1)", show=False)]

    def __init__(self, entries):
        super().__init__()
        self.entries, self.shown = entries, entries

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label("[b]Load spec[/b]  experiment specs, then recorded run specs")
            yield Input(placeholder="filter", id="pick-filter", compact=True)
            yield OptionList(*[label for label, _ in self.entries], id="pick-list")

    def on_input_changed(self, event):
        needle = event.value.lower()
        self.shown = [e for e in self.entries if needle in e[0].plain.lower()]
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options([label for label, _ in self.shown])
        options.highlighted = 0 if self.shown else None

    def action_move(self, step):
        options = self.query_one(OptionList)
        options.action_cursor_down() if step > 0 else options.action_cursor_up()

    def on_input_submitted(self, _):
        i = self.query_one(OptionList).highlighted
        if i is not None:
            self.dismiss(self.shown[i][1])

    def on_option_list_option_selected(self, event):
        self.dismiss(self.shown[event.option_index][1])


class SetupScreen(Screen):
    """New training run (xmodel_path None) or evaluation of a saved tree (§5.5, §5.6)."""
    BINDINGS = [Binding("ctrl+s", "review", "review", priority=True),
                Binding("ctrl+o", "load", "load spec", priority=True),
                Binding("ctrl+e", "edit_config", "edit model_config", priority=True),
                Binding("down", "app.focus_next", show=False), Binding("up", "app.focus_previous", show=False),
                Binding("escape", "app.pop_screen", "discard")]

    def __init__(self, spec=None, xmodel_path=None, loaded=False, message=""):
        super().__init__()
        spec = dict(spec or {})
        self.loaded = dict(spec) if loaded else None
        self.train, self.xmodel_path = xmodel_path is None, xmodel_path
        self.saved_tree = next((t for t in runs.trees(self.app.root, self.app.runs) if t.path == xmodel_path), None)
        self.name0 = spec.pop("name", None)
        self.config0, self.train0 = spec.pop("model_config", None), spec.pop("train_path", None)
        if not loaded and self.config0 is None:  # a new draft starts from the newest training run's config
            configs = runs.model_configs(self.app.root)
            self.config0 = next((c for r in self.app.runs if r.kind == "train" and isinstance(rs := r.spec(), dict)
                                 and (c := rs.get("model_config")) in configs), None)
        self.evals0 = spec.pop("evals", None) or [{}]
        self.extra = spec  # unknown keys: the runner rejects them
        root = self.app.root
        first_test = next((e.get("test_path") for e in self.evals0 if isinstance(e, dict) and e.get("test_path")), None)
        tree_train = self.saved_tree.train_path() if self.saved_tree else None
        start = self.train0 or first_test or tree_train
        self.dataset = str(Path(start).parent) if start else next(iter(runs.datasets(root)), None)
        self.auto_name = dataset_name(self.dataset) if self.dataset else ""
        self.blocks = 0
        self.revision, self.draft, self.result, self.checking = 0, None, None, None
        self.want_review, self.message, self.hints = False, message, {}

    def compose(self) -> ComposeResult:
        root = self.app.root
        yield TopBar(id="top")
        with VerticalScroll(id="form"):
            if not self.train:
                yield Horizontal(Label("tree"), Static(self.xmodel_path, id="tree"), classes="row")
                yield Static(id="tree-info", classes="note")
                yield Static("", id="err-tree", classes="error")
            yield from row("name", Input(self.name0 or self.auto_name, id="name", compact=True), "name")
            if self.train:
                yield from row("model_config", Choice([(Path(p).name, p) for p in runs.model_configs(root)],
                                                      self.config0, id="model_config"), "model_config",
                               Static("ctrl+e edit", classes="hint"))
            yield from row("dataset", Choice([(d.removeprefix("datasets/"), d) for d in runs.datasets(root)],
                                             self.dataset, id="dataset"), "dataset")
            if self.train:
                yield from row("training file", Choice(self.train_options(self.dataset), self.train0 or
                                                       self.preferred_train(self.dataset), id="train_file"),
                               "train_file")
            for e in self.evals0:
                yield self.new_block(e if isinstance(e, dict) else {})
            yield Button("+ Add evaluation", id="add", compact=True)
            yield Static(id="check-line")
        yield Footer(show_command_palette=False)

    def train_options(self, dataset):
        return [(Path(p).name, p) for p in runs.training_files(self.app.root, dataset)] if dataset else []

    def preferred_train(self, dataset, keep=None):
        """Same file name as before, else train_plus_ctd (what every experiment trains on), else the first."""
        files = runs.training_files(self.app.root, dataset) if dataset else []
        for name in (keep, "train_plus_ctd.pubtator"):
            if name and str(Path(dataset, name)) in files:
                return str(Path(dataset, name))
        return files[0] if files else None

    def new_block(self, e=None):
        self.blocks += 1
        splits = runs.split_files(self.app.root, self.dataset) if self.dataset else {}
        return EvalBlock(self.blocks, e or {}, splits, "dev" if self.train else "test",
                         can_breakdown=self.train or bool(self.saved_tree and self.saved_tree.train_path()))

    def on_mount(self):
        self.query_one(TopBar).set("New training run" if self.train else
                                   f"Evaluate tree {fmt.tree(self.saved_tree.id if self.saved_tree else None)}", self.app.host)
        self.refresh_bindings()
        self.edited()
        if self.loaded is not None and (diff := differences(self.loaded, self.draft[0])):
            self.notify("The form differs from the loaded spec at " + ", ".join(diff), severity="warning",
                        timeout=10)
        if not self.read_draft()[2]:  # a loaded or prefilled draft is checked at once
            self.validate()
        self.query_one("#name").focus()

    def check_action(self, action, parameters):
        return action != "edit_config" or self.train

    # draft -> spec ---------------------------------------------------------------------------------------------

    def value(self, key):
        return self.query_one(f"#{key}", Input).value.strip()

    def chosen(self, key):
        v = self.query_one(f"#{key}", Select).value
        return v if isinstance(v, str) else None

    def training_file(self):
        """The file evaluations break mention strings down against: the training file, or the tree's."""
        if self.train:
            return self.chosen("train_file")
        return self.saved_tree.train_path() if self.saved_tree else None

    def read_draft(self):
        """(spec, xmodel_path, errors {widget id: message}) from the form; empty search fields are left out."""
        errors, root = {}, Path(self.app.root)
        spec = {"name": self.value("name")}
        if not spec["name"]:
            errors["name"] = "required"
        if not self.train and not runs.tree_on_disk(root, self.xmodel_path):
            errors["tree"] = f"not on this machine: {self.xmodel_path}/xmodel.pkl"
        if self.train:
            spec["model_config"] = self.chosen("model_config")
            spec["train_path"] = self.chosen("train_file")
            for key, v in (("model_config", spec["model_config"]), ("train_file", spec["train_path"])):
                if not v or not (root / v).is_file():
                    errors[key] = "no such file" if v else "none found"
        if not self.chosen("dataset"):
            errors["dataset"] = "no dataset under datasets/"
        spec["evals"], seen = [], set()
        for block in self.query(EvalBlock):
            e = {"split": block.split(), "test_path": block.path}
            if not block.path:
                errors[f"e{block.n}-split"] = "no dev / test / train file in this dataset"
            elif e["split"] in seen:
                errors[f"e{block.n}-split"] = f"split {e['split']} is evaluated twice"
            seen.add(e["split"])
            if self.query_one(f"#e{block.n}-breakdown", Checkbox).value and self.training_file():
                e["train_path"] = self.training_file()
            for k, kind in SEARCH_TYPES.items():
                if v := self.value(f"e{block.n}-{k}"):
                    try:
                        e[k] = kind(v)
                    except ValueError:
                        errors[f"e{block.n}-{k}"] = "an integer" if kind is int else "a number"
            spec["evals"].append({**e, **block.extra})
        return {**spec, **self.extra}, self.xmodel_path, errors

    def show_errors(self, errors):
        for err in self.query(".error"):
            msg = errors.get(err.id[len("err-"):], "")
            err.update(msg)
            err.display = bool(msg)

    # validation ------------------------------------------------------------------------------------------------

    def edited(self, message=""):
        """Every change of the draft is a new revision; the previous check no longer counts (§5.5)."""
        spec, xmodel_path, errors = self.read_draft()
        self.show_errors({k: v for k, v in errors.items() if v != "required"})  # empty fields: on validate only
        self.update_notes()
        if (spec, xmodel_path) == self.draft and not message:
            return
        self.draft, self.revision, self.result = (spec, xmodel_path), self.revision + 1, None
        self.message = message or self.message
        self.update_hints()
        self.set_check_line(Text(self.message or "edited · checked when you leave the field", style="dim"))

    def on_input_changed(self, _):
        self.message = ""
        self.edited()

    def on_input_blurred(self, _):
        self.validate()

    def on_select_changed(self, event):
        sid = event.select.id
        if sid == "dataset" and event.value != self.dataset and isinstance(event.value, str):
            self.change_dataset(event.value)
        elif sid.endswith("-split"):
            block = next(b for b in self.query(EvalBlock) if b.n == int(sid[1:].split("-")[0]))
            if event.value != block.path and isinstance(event.value, str):
                block.path, block.label = event.value, None
        self.message = ""
        self.edited()
        self.validate()

    def on_checkbox_changed(self, _):
        self.message = ""
        self.edited()
        self.validate()

    def change_dataset(self, dataset):
        """Same training file name and splits in the new dataset where it has them; the default name follows."""
        self.dataset = dataset
        if self.train:
            keep = Path(self.chosen("train_file") or "").name
            self.query_one("#train_file", Choice).set_choices(self.train_options(dataset),
                                                              self.preferred_train(dataset, keep))
        splits = runs.split_files(self.app.root, dataset)
        for block in self.query(EvalBlock):
            block.splits = splits
            block.path = splits.get(block.split()) or splits.get("dev" if self.train else "test") or \
                next(iter(splits.values()), None)
            block.label = None
            self.query_one(f"#e{block.n}-split", Choice).set_choices(list(splits.items()), block.path)
        name = self.query_one("#name", Input)
        if not name.value.strip() or name.value.strip() == self.auto_name:
            self.auto_name = dataset_name(dataset)
            name.value = self.auto_name

    def validate(self):
        """Start -check for the current revision (once); form errors skip it."""
        spec, xmodel_path, errors = self.read_draft()
        self.show_errors(errors)
        if errors:
            self.set_check_line(Text("fix the marked fields", style="red"))
            return False
        if self.result is None and self.checking != self.revision:
            self.checking = self.revision
            self.set_check_line(Text("checking…", style="dim"))
            self.run_check(spec, xmodel_path, self.revision)
        return True

    @work(thread=True, exclusive=True, group="check")
    def run_check(self, spec, xmodel_path, revision):
        # ponytail: a superseded check runs to the end (subprocess.run); its result is dropped by revision
        result = process.check(self.app.root, spec, xmodel_path, revision)
        self.app.call_from_thread(self.got_check, result)

    def got_check(self, result):
        if result.revision != self.revision:  # stale, success or failure alike
            return
        self.result = result
        if result.ok:
            self.hints = result.search
            self.update_hints()
            self.set_check_line(Text("✓ checks passed", style="green"))
            if self.want_review:
                self.open_review()
        else:
            self.want_review = False
            self.set_check_line(Text(f"✗ {result.error}", style="red"))

    def set_check_line(self, text):
        """The check state, after the reason for revalidating (binding rejected, config edited) until the next edit."""
        if self.message and text.plain != self.message:
            text = Text.assemble((f"{self.message} · ", "yellow"), text)
        self.query_one("#check-line", Static).update(text)

    def inherited_from(self):
        """(label, predict_config) the empty search fields inherit from, read from the JSON files (§5.5)."""
        if self.train:
            config, _ = runs.read_json(Path(self.app.root, self.chosen("model_config") or ""))
            return "config", (config or {}).get("predict_config") if isinstance(config, dict) else None
        view, _ = runs.read_json(Path(self.app.root, self.xmodel_path, "xmodel.json"))
        return "tree", (view or {}).get("predict_config") if isinstance(view, dict) else None

    def update_hints(self):
        label, predict = self.inherited_from()
        for block in self.query(EvalBlock):
            search = self.hints.get(block.split(), {})
            for k in SEARCH_TYPES:
                self.query_one(f"#e{block.n}-{k}", Input).placeholder = "" if k not in search else \
                    f"{label if isinstance(predict, dict) and k in predict else 'default'} {search[k]}"

    def update_notes(self):
        """Evaluation headings, tree facts."""
        for block in self.query(EvalBlock):
            self.query_one(f"#head-{block.n}", Label).update(block.heading())
        if not self.train:
            t = self.saved_tree
            parts = [f"trained by {t.trained_by.name} on {t.train_path()}" if t and t.trained_by else
                     "training run not found: no breakdown",
                     f"{fmt.count(t.labels)} labels" if t and t.labels else None,
                     "on this machine" if runs.tree_on_disk(self.app.root, self.xmodel_path) else
                     "not on this machine"]
            self.query_one("#tree-info", Static).update(" · ".join(p for p in parts if p))

    # actions ---------------------------------------------------------------------------------------------------

    def action_review(self):
        """Review the current draft once its check passed; waits for a check in flight."""
        _, _, errors = self.read_draft()
        if errors:
            first = next(iter(errors))
            if first != "tree":
                self.query_one(f"#{first}").focus()
            self.validate()
            return
        if self.result is not None and not self.result.ok:
            self.notify(self.result.error, severity="error")
            return
        self.want_review = True
        if self.result is not None:
            self.open_review()
        else:
            self.validate()

    def open_review(self):
        self.want_review = False
        spec, xmodel_path = self.draft
        self.app.push_screen(ReviewScreen(spec, xmodel_path, self.result))

    def binding_rejected(self, message):
        """Back from a Start the runner refused (exit 3), or the config was edited: check again."""
        self.edited(message)
        self.validate()

    def on_button_pressed(self, event):
        bid = event.button.id
        if bid == "add":
            self.query_one("#form").mount(self.new_block(), before="#add")
            self.call_after_refresh(self.edited)
        elif bid.startswith("remove-"):
            next(b for b in self.query(EvalBlock) if b.n == int(bid.split("-")[1])).remove()
            self.call_after_refresh(self.edited)

    def action_load(self):
        entries = [(Text(p.removeprefix("configs/")), p) for p in runs.experiment_specs(self.app.root)]
        entries += [(Text.assemble(r.id, (f"  {r.kind} run", "dim")), str(r.path / "spec.json"))
                    for r in self.app.runs if (r.path / "spec.json").is_file()]
        self.app.push_screen(SpecPicker(entries), self.load)

    def load(self, path):
        if path is None:
            return
        obj, problem = runs.read_json(Path(self.app.root, path))
        if not isinstance(obj, dict) or not isinstance(obj.get("evals", []), list):
            self.notify(problem or f"{path}: not a spec", severity="error")
            return
        spec, xmodel_path = runs.normalize_spec(obj)
        self.app.switch_screen(SetupScreen(spec, xmodel_path, loaded=True))

    def action_edit_config(self):
        path = self.chosen("model_config")
        if not path:
            self.notify("no model_config chosen", severity="warning")
            return
        editor = shlex.split(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi")
        with self.app.suspend():
            subprocess.run([*editor, path], cwd=self.app.root)
        self.binding_rejected("model_config edited")  # same file, new contents: a new check


class ReviewScreen(Screen):
    """The exact validated snapshot (§5.7); Start passes it with its model-config hash / tree id."""
    BINDINGS = [Binding("enter", "start", "start"), Binding("j", "json", "json"),
                Binding("a", "save", "save spec as…"), Binding("escape", "back", "back")]

    def __init__(self, spec, xmodel_path, result):
        super().__init__()
        self.spec, self.xmodel_path, self.result = spec, xmodel_path, result
        self.run_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{spec['name']}"
        self.child, self.timer, self.json = None, None, False

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with VerticalScroll():
            yield Static(self.body(), id="review-text")
            yield Static(id="start-line")
            with Horizontal(id="review-buttons"):
                yield Button("Back", id="back", compact=True)
                yield Button("Start run", id="start", variant="primary", compact=True)
            yield Input(placeholder="save the spec as (path); esc cancels", id="save")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        self.query_one("#save").display = False
        self.query_one(TopBar).set(f"Review  {self.spec['name']}", self.app.host)
        self.query_one("#start", Button).focus()
        self.call_after_refresh(self.query_one(VerticalScroll).scroll_home, animate=False)  # summary first

    def body(self):
        splits = ", ".join(e["split"] for e in self.spec["evals"])
        t = Text(f"Trains a new tree, then evaluates {splits}.\n" if self.xmodel_path is None else
                 f"Evaluates tree {fmt.tree(self.result.tree_id)} ({self.xmodel_path}) on {splits}.\n")
        t.append("Run directory  ", style="bold")
        t.append(f"{RUNS_DIR / self.run_id} (new)\n\n")
        if self.json:
            t.append(json.dumps(self.spec, indent=2) + "\n\n")
        if self.xmodel_path is None:
            t.append("Model config      ", style="bold")
            t.append(f"{self.spec['model_config']}\n" + " " * 18 +
                     f"sha256 {self.result.model_config_sha256[:8]}… (bound; full hash in Inputs after start)\n")
            t.append("Training file     ", style="bold")
            t.append(f"{self.spec['train_path']}\n")
        else:
            t.append("Tree              ", style="bold")
            t.append(f"tree_id {self.result.tree_id} (bound)\n")
        for e in self.spec["evals"]:
            s = self.result.search.get(e["split"], {})
            t.append(f"\n{e['split']:<18}", style="bold")
            t.append(e["test_path"] + (" · breakdown" if e.get("train_path") else "") + "\n" + " " * 18)
            t.append(" · ".join(f"{k} {s[k]}" for k in SEARCH_TYPES if k in s) + "\n")
        t.append("\n✓ runner checks passed (keys, names, search, files, model config)\n", style="green")
        t.append("  Corpus contents are checked when the run reads them.\n")
        running = [r.name for r in self.app.runs if r.live is not None and r.live.state == "RUNNING"]
        if running:
            t.append(f"! {', '.join(running)} {'is' if len(running) == 1 else 'are'} running on this host and "
                     "shares the GPU.\n", style="yellow")
        return t

    def on_button_pressed(self, event):
        self.action_start() if event.button.id == "start" else self.action_back()

    def action_json(self):
        self.json = not self.json
        self.query_one("#review-text", Static).update(self.body())

    def action_back(self):
        box = self.query_one("#save", Input)
        if box.display:
            box.display = False
        elif self.child is None:
            self.app.pop_screen()

    def action_start(self):
        if self.child is not None:
            return
        self.child = process.spawn(self.app.root, self.spec, self.run_id, self.xmodel_path,
                                   self.result.model_config_sha256, self.result.tree_id)
        self.app.launched.append(self.child)
        self.query_one("#start-line", Static).update(Text("starting…", style="dim"))
        self.timer = self.set_interval(0.25, self.watch_start)

    def watch_start(self):
        """Open the run once its status.json exists; exit 3 -> back to Setup; another early exit -> console log."""
        if Path(self.app.root, RUNS_DIR, self.run_id, "status.json").is_file():
            self.timer.stop()
            self.app.call_later(self.app.started, self.run_id)  # not from this screen's own handler
            return
        code = self.child.poll()
        if code is None:
            return
        self.timer.stop()
        rejected = process.rejection(self.app.root, self.run_id) if code == 3 else None
        if rejected:
            setup = self.app.screen_stack[-2]
            self.app.pop_screen()
            setup.call_later(setup.binding_rejected, "Model configuration changed; review again"
                                             if rejected["error"] == "model_config_changed"
                                             else "Tree changed; review again")
            return
        console = RUNS_DIR / ".console" / f"{self.run_id}.log"
        try:
            tail = Path(self.app.root, console).read_text().splitlines()[-8:]
        except OSError:
            tail = []
        self.query_one("#start-line", Static).update(
            Text(f"Startup failed (exit {code}); console log {console}\n" + "\n".join(tail), style="red"))
        self.child = None  # esc returns to Setup

    def action_save(self):
        box = self.query_one("#save", Input)
        box.display = True
        box.focus()

    def on_input_submitted(self, event):
        path = Path(self.app.root, event.value.strip())
        try:
            with open(path, "x") as f:  # never overwrites
                f.write(json.dumps(self.spec, indent=2) + "\n")
        except OSError as err:
            self.notify(f"Not saved: {err}", severity="error")
            return
        event.input.display = False
        self.notify(f"Saved {event.value.strip()}" +
                    (f"; evaluate with -xmodel_path {self.xmodel_path}" if self.xmodel_path else ""))
