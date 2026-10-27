"""Setup -> -check -> Review -> spawn (tui.md §5.5-5.7) against a fake runner: dataset/split pickers and arrow keys,
stale checks never enable Review, form errors stop at the field, Start opens the new run, exit 3 returns to Setup,
ctrl+o picker loads a spec unchanged, Trees `e` evaluates a tree on this machine only."""
import json
from pathlib import Path

import pytest

pytest.importorskip("textual")

from fake_runs import make_run  # noqa: E402
from test_screens import run_app, text  # noqa: E402
from textual.widgets import Input, Static  # noqa: E402
from xmr4el_tui import process  # noqa: E402
from xmr4el_tui.launch import ReviewScreen, SetupScreen, SpecPicker  # noqa: E402
from xmr4el_tui.screens import RunScreen, RunsScreen, TreesScreen  # noqa: E402

# -check: effective search over beam 10 / the eval's keys; FAIL_CHECK fails it. Run: exit 3 when REJECT exists,
# else create -run_dir with status.json running (recorded spec + argv) and exit.
RUNNER = """import json, os, sys
spec = json.load(sys.stdin)
if "-check" in sys.argv:
    if os.path.exists("FAIL_CHECK"):
        print(json.dumps({"ok": False, "error": "ValueError: no", "model_config_sha256": None, "tree_id": None,
                          "search": {}}))
        sys.exit(1)
    train = "-xmodel_path" not in sys.argv
    print(json.dumps({"ok": True, "error": None, "model_config_sha256": "c" * 64 if train else None,
                      "tree_id": None if train else "8528b624aaaa",
                      "search": {e["split"]: {"beam_size": e.get("beam_size", 10), "leaf_candidates": 100, "topk": 0,
                                              "knn_beta": 10.0} for e in spec["evals"]}}))
    sys.exit(0)
if os.path.exists("REJECT"):
    print(json.dumps({"error": "model_config_changed", "expected": "c", "actual": "d"}))
    sys.exit(3)
run_dir = sys.argv[sys.argv.index("-run_dir") + 1]
os.makedirs(run_dir)
json.dump({**spec, "xmodel_path": None, "argv": sys.argv[1:]}, open(os.path.join(run_dir, "spec.json"), "w"))
json.dump({"status": "running", "started": "2026-10-23T15:00:00"}, open(os.path.join(run_dir, "status.json"), "w"))
"""


SYN = {"name": "syn", "model_config": "configs/c.json", "train_path": "datasets/x/train_plus_ctd.pubtator",
       "evals": [{"split": "dev_knn0", "test_path": "datasets/x/dev.pubtator",
                  "train_path": "datasets/x/train_plus_ctd.pubtator", "knn_beta": 0}]}


def fake_root(tmp_path):
    make_run(tmp_path, "ncbi_disease", tree_here=True)
    make_run(tmp_path, "gone", stamp="2026-10-19_10-00-00", tree_id="feedbeef")
    files = ["configs/c.json", *[f"datasets/x/{f}.pubtator" for f in ("train", "train_plus_ctd", "dev", "test")],
             "datasets/y/train.pubtator", "datasets/y/dev.pubtator"]
    for f in files:
        Path(tmp_path, f).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_path, f).write_text(json.dumps({"predict_config": {"knn_beta": 10.0}}))
    Path(tmp_path, "configs/experiments").mkdir()
    Path(tmp_path, "configs/experiments/syn.json").write_text(json.dumps(SYN))
    Path(tmp_path, "outputs/saved_trees/xmodel_8528b624aaaa/xmodel.json").write_text(json.dumps(
        {"tree_id": "8528b624aaaa", "predict_config": {"beam_size": 10}, "initial_labels": {"type": "list", "len": 13385}}))
    Path(tmp_path, process.RUNNER).parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_path, process.RUNNER).write_text(RUNNER)
    return tmp_path


async def settle(pilot, screen, cond, tries=200):
    for _ in range(tries):
        await pilot.pause(0.05)
        if cond():
            return
    raise AssertionError(text(screen.query_one("#check-line", Static)))


async def fill(pilot, screen, values):
    for key, value in values.items():
        screen.query_one(f"#{key}", Input).value = value
    await pilot.pause()


def test_setup_review_start(tmp_path):
    async def check(app, pilot):
        await pilot.press("n")
        setup = app.screen
        assert isinstance(setup, SetupScreen) and app.focused.id == "name"
        assert setup.value("name") == "x" and setup.chosen("dataset") == "datasets/x"
        assert setup.chosen("train_file") == "datasets/x/train_plus_ctd.pubtator"
        await pilot.press("down", "down")
        assert app.focused.id == "dataset", "up/down move between fields"
        await pilot.press("right")  # cycle to datasets/y: training file and default name follow
        await pilot.pause()
        assert setup.chosen("dataset") == "datasets/y" and setup.value("name") == "y"
        assert setup.chosen("train_file") == "datasets/y/train.pubtator" and setup.chosen("e1-split") == "datasets/y/dev.pubtator"
        await pilot.press("left")
        await pilot.pause()
        assert setup.chosen("dataset") == "datasets/x" and setup.value("name") == "x"
        await pilot.press("down", "down", "down")  # training file, Remove, split
        assert app.focused.id == "e1-split"
        await pilot.press("right")  # train -> dev -> test
        await pilot.pause()
        assert setup.chosen("e1-split") == "datasets/x/test.pubtator"
        assert "Evaluation test" in text(setup.query_one("#head-1"))
        await fill(pilot, setup, {"name": "", "e1-beam_size": "x"})
        await pilot.press("ctrl+s")
        assert app.focused.id == "name", "ctrl+s focuses the first invalid field"
        assert text(setup.query_one("#err-e1-beam_size", Static)) == "an integer"
        await fill(pilot, setup, {"name": "syn", "e1-beam_size": "5"})
        await pilot.press("ctrl+s")  # waits for the check, then opens Review
        await settle(pilot, setup, lambda: isinstance(app.screen, ReviewScreen))
        assert app.screen.spec == {"name": "syn", "model_config": "configs/c.json",
                                   "train_path": "datasets/x/train.pubtator",
                                   "evals": [{"split": "test", "test_path": "datasets/x/test.pubtator",
                                              "train_path": "datasets/x/train.pubtator", "beam_size": 5}]}
        body = text(app.screen.query_one("#review-text", Static))
        assert "datasets/x/test.pubtator · breakdown\n" + " " * 18 + "beam_size 5 · leaf_candidates 100 · topk 0 · " \
               "knn_beta 10.0" in body and "sha256 cccccccc…" in body and '"evals"' not in body
        await pilot.press("j")
        assert '"evals"' in text(app.screen.query_one("#review-text", Static)), "j shows the spec JSON"
        await pilot.press("escape")
        assert app.screen is setup
        assert setup.query_one("#e1-knn_beta", Input).placeholder == "config 10.0"
        assert setup.query_one("#e1-leaf_candidates", Input).placeholder == "default 100"
        await pilot.press("ctrl+s")
        await settle(pilot, setup, lambda: isinstance(app.screen, ReviewScreen))
        run_id = app.screen.run_id
        await pilot.press("enter")
        await settle(pilot, setup, lambda: isinstance(app.screen, RunScreen))
        assert app.screen.run_id == run_id and [type(s) for s in app.screen_stack[1:]] == [RunsScreen, RunScreen]
        argv = json.loads((tmp_path / "outputs/runs" / run_id / "spec.json").read_text())["argv"]
        assert argv[argv.index("-model_config_sha256") + 1] == "c" * 64
    run_app(fake_root(tmp_path), check)


def test_stale_check_and_rejection(tmp_path):
    async def check(app, pilot):
        app.push_screen(SetupScreen(SYN, loaded=True))
        await pilot.pause()
        setup = app.screen
        await settle(pilot, setup, lambda: setup.result is not None)
        assert setup.result.ok
        await fill(pilot, setup, {"e1-beam_size": "3"})  # new revision: the old success no longer counts
        assert setup.result is None
        old = process.CheckResult(setup.revision - 1, True, None, "c" * 64, None, {"dev": {}})
        setup.got_check(old)
        assert setup.result is None, "a stale success cannot enable Review"
        (tmp_path / "FAIL_CHECK").write_text("")
        setup.validate()
        await settle(pilot, setup, lambda: setup.result is not None)
        assert not setup.result.ok and "ValueError: no" in text(setup.query_one("#check-line", Static))
        await pilot.press("ctrl+s")
        assert app.screen is setup
        (tmp_path / "FAIL_CHECK").unlink()
        await fill(pilot, setup, {"e1-beam_size": "4"})
        (tmp_path / "REJECT").write_text("")
        await pilot.press("ctrl+s")
        await settle(pilot, setup, lambda: isinstance(app.screen, ReviewScreen))
        await pilot.press("enter")
        await settle(pilot, setup, lambda: app.screen is setup and setup.result is not None)
        assert text(setup.query_one("#check-line", Static)) == \
            "Model configuration changed; review again · ✓ checks passed"
        assert not list((tmp_path / "outputs/runs").glob("*_syn"))
    run_app(fake_root(tmp_path), check)


def test_load_picker(tmp_path):
    async def check(app, pilot):
        await pilot.press("n", "ctrl+o")
        assert isinstance(app.screen, SpecPicker)
        await pilot.press(*"syn", "enter")
        await pilot.pause()
        setup = app.screen
        assert isinstance(setup, SetupScreen) and setup.draft[0] == SYN, "a loaded spec round-trips unchanged"
        assert "Evaluation dev_knn0" in text(setup.query_one("#head-1"))
        await pilot.press("ctrl+o", *"ncbi_disease", "down", "enter")  # the recorded spec of the eval run
        await pilot.pause()
        assert app.screen.xmodel_path is None and app.screen.chosen("train_file") == "datasets/x/train.pubtator"
    run_app(fake_root(tmp_path), check)


def test_trees_evaluate(tmp_path):
    async def check(app, pilot):
        await pilot.press("t")
        trees = app.screen
        assert isinstance(trees, TreesScreen) and [t.id for t in trees.trees] == ["8528b624aaaa", "feedbeef"]
        await pilot.press("down", "e")
        assert app.screen is trees, "a tree not on this machine is not evaluated"
        await pilot.press("up", "e")
        setup = app.screen
        assert isinstance(setup, SetupScreen) and setup.xmodel_path == "outputs/saved_trees/xmodel_8528b624aaaa"
        assert text(setup.query_one("#tree-info", Static)) == \
            "trained by ncbi_disease on datasets/x/train.pubtator · 13,385 labels · on this machine"
        assert "Evaluate tree 8528b624" in text(setup.query_one("#top"))
        assert setup.chosen("e1-split") == "datasets/x/test.pubtator" and setup.value("name") == "ncbi_disease_eval"
        await settle(pilot, setup, lambda: setup.result is not None)
        assert setup.query_one("#e1-beam_size", Input).placeholder == "tree 10"
        await pilot.press("ctrl+s")
        await settle(pilot, setup, lambda: isinstance(app.screen, ReviewScreen))
        assert app.screen.spec["evals"] == [{"split": "test", "test_path": "datasets/x/test.pubtator",
                                             "train_path": "datasets/x/train.pubtator"}]
        assert "Evaluates tree 8528b624 (outputs/saved_trees/xmodel_8528b624aaaa) on test." in \
            text(app.screen.query_one("#review-text", Static))
    run_app(fake_root(tmp_path), check)
