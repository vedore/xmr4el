"""Setup and Review screens (tui.md §5.5-5.7): a form for one input spec, the runner's -check in the background,
Review of the exact validated snapshot, spawn. The runner stays the authority; the form only checks numbers, paths."""
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
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Static

from xmr4el_tui import fmt, process, runs
from xmr4el_tui.runs import RUNS_DIR
from xmr4el_tui.widgets import TopBar

SEARCH_TYPES = {"beam_size": int, "leaf_candidates": int, "topk": int, "knn_beta": float}
EVAL_FIELDS = ("split", "test_path", "train_path", *SEARCH_TYPES)
HELP = {
    "name": "**name**: run name, `[A-Za-z0-9_.-]+`; the run directory is `<timestamp>_<name>`.",
    "model_config": "**model_config**: model configuration JSON (features, clustering, matcher, "
                    "`predict_config`). ctrl+e opens it in `$VISUAL` / `$EDITOR`.",
    "train_path": "**train_path**: training corpus (PubTator).",
    "tree": "**tree**: saved tree directory (holds `xmodel.pkl`); evaluated, not retrained.",
    "split": "**split**: evaluation name; the run writes `metrics_<split>.json`.",
    "test_path": "**test_path**: evaluation corpus (PubTator).",
    "eval_train_path": "**train_path**: corpus for the seen/unseen mention-string breakdown; empty skips it.",
    "beam_size": "**beam_size**: clusters kept per layer during the search.",
    "leaf_candidates": "**leaf_candidates**: labels kept per reached leaf.",
    "topk": "**topk**: labels returned per mention; 0 keeps every candidate.",
    "knn_beta": "**knn_beta**: weight of the kNN score fusion; 0 turns it off.",
}


def field(key, value, help_key=None, hint=False):
    """One form row: label, input, inherited-value hint; the form error sits under it."""
    box = Input(str(value) if value is not None else "", id=key, compact=True)
    box.HELP = HELP[help_key or key.split("-")[-1]]
    yield Horizontal(Label(key.split("-")[-1]), box, *([Static("", id=f"hint-{key}", classes="hint")] if hint else []),
                     classes="row")
    yield Static("", id=f"err-{key}", classes="error")


class EvalBlock(Vertical):
    """One evaluation; `n` keeps input ids unique when blocks are removed. Keys the form does not know are carried
    in `extra` so the runner rejects them (§10: never drop them silently)."""

    def __init__(self, n, e=None):
        super().__init__(classes="eval")
        self.n, e = n, dict(e or {})
        self.values = {k: e.pop(k, None) for k in EVAL_FIELDS}
        self.extra = e

    def compose(self) -> ComposeResult:
        with Horizontal(classes="eval-head"):
            yield Label(f"Evaluation {self.values['split'] or ''}")
            yield Button("Remove", id=f"remove-{self.n}", compact=True)
        for k in EVAL_FIELDS:
            yield from field(f"e{self.n}-{k}", self.values[k], "eval_train_path" if k == "train_path" else None,
                             hint=k in SEARCH_TYPES)


class SetupScreen(Screen):
    """New training run (xmodel_path None) or evaluation of a saved tree (§5.5, §5.6)."""
    BINDINGS = [Binding("ctrl+s", "review", "review", priority=True),
                Binding("ctrl+o", "load", "load spec", priority=True),
                Binding("ctrl+e", "edit_config", "edit model_config", priority=True),
                Binding("escape", "discard", "discard")]

    def __init__(self, spec=None, xmodel_path=None, message=""):
        super().__init__()
        spec = dict(spec or {"evals": [{"split": "dev"}]})
        self.train = xmodel_path is None
        self.top = {**({} if self.train else {"tree": xmodel_path}), "name": spec.pop("name", None)}
        if self.train:
            self.top |= {"model_config": spec.pop("model_config", None), "train_path": spec.pop("train_path", None)}
        self.evals = spec.pop("evals", None) or [{}]
        self.extra = spec  # unknown keys: the runner rejects them
        self.blocks = 0
        self.revision, self.draft, self.result, self.checking = 0, None, None, None
        self.want_review, self.message, self.hints = False, message, {}

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with VerticalScroll(id="form"):
            for k, v in self.top.items():
                yield from field(k, v)
                if k == "tree":
                    yield Static(id="tree-info", classes="note")
            for e in self.evals:
                yield self.new_block(e)
            yield Button("+ Add evaluation", id="add", compact=True)
            yield Static(id="check-line")
            yield Input(placeholder="spec file to load (input spec or a run's spec.json); esc cancels", id="load")
        yield Footer()

    def new_block(self, e=None):
        self.blocks += 1
        return EvalBlock(self.blocks, e)

    def on_mount(self):
        self.query_one("#load").display = False
        self.query_one(TopBar).set("New training run" if self.train else f"Evaluate tree {self.tree_short()}",
                                   self.app.host)
        self.refresh_bindings()
        self.edited()
        if not self.read_draft()[2]:  # a loaded or prefilled draft is checked at once
            self.validate()
        self.query(Input).first().focus()

    def check_action(self, action, parameters):
        return action != "edit_config" or self.train

    def tree_short(self):
        view, _ = runs.read_json(Path(self.app.root, self.top["tree"] or "", "xmodel.json"))
        return fmt.tree((view or {}).get("tree_id")) if isinstance(view, dict) else fmt.DASH

    # draft -> spec ---------------------------------------------------------------------------------------------

    def value(self, key):
        return self.query_one(f"#{key}", Input).value.strip()

    def read_draft(self):
        """(spec, xmodel_path, errors {input id: message}) from the form; empty search fields are left out."""
        errors, root = {}, Path(self.app.root)

        def path(key, required=True, tree=False):
            v = self.value(key)
            if not v:
                if required:
                    errors[key] = "required"
            elif not (root / v / "xmodel.pkl" if tree else root / v).is_file():
                errors[key] = "no xmodel.pkl in this directory" if tree else "no such file"
            return v

        spec = {"name": self.value("name")}
        if not spec["name"]:
            errors["name"] = "required"
        xmodel_path = None if self.train else path("tree", tree=True)
        if self.train:
            spec |= {"model_config": path("model_config"), "train_path": path("train_path")}
        spec["evals"] = []
        for block in self.query(EvalBlock):
            key = lambda k: f"e{block.n}-{k}"
            e = {"split": self.value(key("split")), "test_path": path(key("test_path"))}
            if not e["split"]:
                errors[key("split")] = "required"
            if v := path(key("train_path"), required=False):
                e["train_path"] = v
            for k, kind in SEARCH_TYPES.items():
                if v := self.value(key(k)):
                    try:
                        e[k] = kind(v)
                    except ValueError:
                        errors[key(k)] = "an integer" if kind is int else "a number"
            spec["evals"].append({**e, **block.extra})
        return {**spec, **self.extra}, xmodel_path, errors

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
        self.update_tree_info()
        if (spec, xmodel_path) == self.draft and not message:
            return
        self.draft, self.revision, self.result = (spec, xmodel_path), self.revision + 1, None
        self.message = message or self.message
        self.update_hints()
        self.set_check_line(Text(self.message or "edited · checked when you leave the field", style="dim"))

    def on_input_changed(self, event):
        if event.input.id != "load":
            self.message = ""
            self.edited()

    def on_input_blurred(self, event):
        if event.input.id != "load":
            self.validate()

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
            config, _ = runs.read_json(Path(self.app.root, self.value("model_config")))
            return "config", (config or {}).get("predict_config") if isinstance(config, dict) else None
        view, _ = runs.read_json(Path(self.app.root, self.value("tree"), "xmodel.json"))
        return "tree", (view or {}).get("predict_config") if isinstance(view, dict) else None

    def update_hints(self):
        label, predict = self.inherited_from()
        for block in self.query(EvalBlock):
            search = self.hints.get(self.value(f"e{block.n}-split"), {})
            for k in SEARCH_TYPES:
                typed = self.value(f"e{block.n}-{k}")
                hint = "" if typed or k not in search else \
                    f"{label if isinstance(predict, dict) and k in predict else 'default'} {search[k]}"
                self.query_one(f"#hint-e{block.n}-{k}", Static).update(hint)

    def update_tree_info(self):
        if self.train:
            return
        tree = self.value("tree")
        view, _ = runs.read_json(Path(self.app.root, tree, "xmodel.json"))
        view = view if isinstance(view, dict) else {}
        trained = runs.training_run(self.app.runs, view.get("tree_id"))
        labels = (view.get("initial_labels") or {}).get("len")
        here = "on this machine" if runs.tree_on_disk(self.app.root, tree) else "not on this machine"
        parts = [f"trained by {trained.name}" if trained else None,
                 f"{fmt.count(labels)} labels" if isinstance(labels, int) else None, here]
        self.query_one("#tree-info", Static).update(" · ".join(p for p in parts if p))

    # actions ---------------------------------------------------------------------------------------------------

    def action_review(self):
        """Review the current draft once its check passed; waits for a check in flight."""
        _, _, errors = self.read_draft()
        if errors:
            self.query_one(f"#{next(iter(errors))}", Input).focus()
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
        """Back from a Start the runner refused (exit 3): revalidate before another Review."""
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
        box = self.query_one("#load", Input)
        box.display = True
        box.focus()

    def on_input_submitted(self, event):
        if event.input.id != "load":
            return
        path = Path(self.app.root, event.value.strip())
        obj, problem = runs.read_json(path)
        if not isinstance(obj, dict) or not isinstance(obj.get("evals", []), list):
            self.notify(problem or f"{path}: not a spec", severity="error")
            return
        self.app.switch_screen(SetupScreen(*runs.normalize_spec(obj)))

    def action_edit_config(self):
        path = self.value("model_config")
        if not path:
            self.notify("model_config is empty", severity="warning")
            return
        editor = shlex.split(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi")
        with self.app.suspend():
            subprocess.run([*editor, path], cwd=self.app.root)
        self.binding_rejected("model_config edited")  # same file, new contents: a new check

    def action_discard(self):
        box = self.query_one("#load", Input)
        if box.display:
            box.display = False
            return
        self.app.pop_screen()


class ReviewScreen(Screen):
    """The exact validated snapshot (§5.7); Start passes it with its model-config hash / tree id."""
    BINDINGS = [Binding("enter", "start", "start"), Binding("a", "save", "save spec as…"),
                Binding("escape", "back", "back")]

    def __init__(self, spec, xmodel_path, result):
        super().__init__()
        self.spec, self.xmodel_path, self.result = spec, xmodel_path, result
        self.run_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{spec['name']}"
        self.child, self.timer = None, None

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with VerticalScroll():
            yield Static(self.body(), id="review-text")
            yield Static(id="start-line")
            with Horizontal(id="review-buttons"):
                yield Button("Back", id="back", compact=True)
                yield Button("Start run", id="start", variant="primary", compact=True)
            yield Input(placeholder="save the spec as (path); esc cancels", id="save")
        yield Footer()

    def on_mount(self):
        self.query_one("#save").display = False
        self.query_one(TopBar).set(f"Review  {self.spec['name']}", self.app.host)
        self.query_one("#start", Button).focus()

    def body(self):
        splits = ", ".join(e["split"] for e in self.spec["evals"])
        t = Text(f"Trains a new tree, then evaluates {splits}.\n" if self.xmodel_path is None else
                 f"Evaluates tree {fmt.tree(self.result.tree_id)} ({self.xmodel_path}) on {splits}.\n")
        t.append("Run directory  ", style="bold")
        t.append(f"{RUNS_DIR / self.run_id} (new)\n\n")
        t.append(json.dumps(self.spec, indent=2) + "\n\n")
        for i, (split, s) in enumerate(self.result.search.items()):
            t.append("Effective search  " if i == 0 else " " * 18, style="bold")
            t.append(f"{split}: " + " · ".join(f"{k} {s[k]}" for k in SEARCH_TYPES if k in s) + "\n")
        if self.xmodel_path is None:
            t.append("Model config      ", style="bold")
            t.append(f"sha256 {self.result.model_config_sha256[:8]}… (bound; full hash in Inputs after start)\n")
        else:
            t.append("Tree              ", style="bold")
            t.append(f"tree_id {self.result.tree_id} (bound)\n")
        t.append("\n✓ runner checks passed (keys, names, search, files, model config)\n", style="green")
        t.append("  Corpus contents are checked when the run reads them.\n")
        running = [r.name for r in self.app.runs if r.live is not None and r.live.state == "RUNNING"]
        if running:
            t.append(f"! {', '.join(running)} {'is' if len(running) == 1 else 'are'} running on this host and "
                     "shares the GPU.\n", style="yellow")
        return t

    def on_button_pressed(self, event):
        self.action_start() if event.button.id == "start" else self.action_back()

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
