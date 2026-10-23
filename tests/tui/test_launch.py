"""Setup -> -check -> Review -> spawn (tui.md §5.5-5.7) against a fake runner: stale checks never enable Review,
form errors stop at the field, Start opens the new run, exit 3 returns to Setup to revalidate."""
import json
from pathlib import Path

import pytest

pytest.importorskip("textual")

from fake_runs import make_run  # noqa: E402
from test_screens import run_app, text  # noqa: E402
from textual.widgets import Input, Static  # noqa: E402
from xmr4el_tui import process  # noqa: E402
from xmr4el_tui.launch import ReviewScreen, SetupScreen  # noqa: E402
from xmr4el_tui.screens import RunScreen, RunsScreen  # noqa: E402

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


def fake_root(tmp_path):
    make_run(tmp_path, "ncbi_disease", tree_here=True)
    for f in ("configs/c.json", "datasets/x/train.pubtator", "datasets/x/dev.pubtator"):
        Path(tmp_path, f).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_path, f).write_text(json.dumps({"predict_config": {"knn_beta": 10.0}}))
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
        assert isinstance(setup, SetupScreen)
        await fill(pilot, setup, {"name": "syn", "model_config": "configs/c.json", "train_path": "nope",
                                  "e1-test_path": "datasets/x/dev.pubtator", "e1-beam_size": "x"})
        await pilot.press("ctrl+s")
        assert app.focused.id == "train_path", "ctrl+s focuses the first invalid field"
        assert text(setup.query_one("#err-e1-beam_size", Static)) == "an integer"
        await fill(pilot, setup, {"train_path": "datasets/x/train.pubtator", "e1-beam_size": "5"})
        await pilot.press("ctrl+s")  # waits for the check, then opens Review
        await settle(pilot, setup, lambda: isinstance(app.screen, ReviewScreen))
        review = app.screen
        body = text(review.query_one("#review-text", Static))
        assert '"beam_size": 5' in body and "dev: beam_size 5 · leaf_candidates 100 · topk 0 · knn_beta 10.0" in body
        assert "sha256 cccccccc…" in body
        await pilot.press("escape")
        assert app.screen is setup
        assert text(setup.query_one("#hint-e1-knn_beta", Static)) == "config 10.0"
        assert text(setup.query_one("#hint-e1-leaf_candidates", Static)) == "default 100"
        assert text(setup.query_one("#hint-e1-beam_size", Static)) == "", "a typed value replaces the hint"
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
        app.setup({"name": "syn", "model_config": "configs/c.json", "train_path": "datasets/x/train.pubtator",
                   "evals": [{"split": "dev", "test_path": "datasets/x/dev.pubtator"}]})
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


def test_evaluate_tree_prefilled(tmp_path):
    async def check(app, pilot):
        await pilot.press("e")
        setup = app.screen
        assert isinstance(setup, SetupScreen) and not setup.train
        assert setup.query_one("#tree", Input).value == "outputs/saved_trees/xmodel_8528b624aaaa"
        assert setup.query_one("#e1-test_path", Input).value == "datasets/x/dev.pubtator"
        assert text(setup.query_one("#tree-info", Static)) == "trained by ncbi_disease · 13,385 labels · on this machine"
        assert "Evaluate tree 8528b624" in text(setup.query_one("#top"))
        await settle(pilot, setup, lambda: setup.result is not None)
        assert text(setup.query_one("#hint-e1-beam_size", Static)) == "tree 10"
        await pilot.press("ctrl+s")
        await settle(pilot, setup, lambda: isinstance(app.screen, ReviewScreen))
        assert "Evaluates tree 8528b624 (outputs/saved_trees/xmodel_8528b624aaaa) on dev." in \
            text(app.screen.query_one("#review-text", Static))
        await pilot.press("escape")
        await pilot.press("escape")
        assert isinstance(app.screen, RunsScreen)
    run_app(fake_root(tmp_path), check)
