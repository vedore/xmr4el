"""Dashboard: explorer model (trees by dataset, evaluations, runs without a tree), keyboard, file-system sync without
restart, that nothing under the root is written; app keys (help, rescan, quit, theme)."""
import asyncio
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("textual")

from fake_runs import make_run  # noqa: E402
from textual.widgets import DataTable, Static  # noqa: E402
from xmr4el_tui import explorer, runs  # noqa: E402
from xmr4el_tui.app import XMR4ELApp, settings_path  # noqa: E402
from xmr4el_tui.dashboard import DashboardScreen, Description, Explorer  # noqa: E402

TREE = "8528b624aaaa"


def text(widget):
    return str(widget.render())


def run_app(root, test, size=(120, 36), opened=None):
    """`opened`: list collecting what Results / Run monitor would open (their screens are not built yet)."""
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            if opened is not None:
                app.screen.open_results = lambda f, sel: opened.append(("results", f, sel))
                app.screen.open_run = lambda rid, split=None: opened.append(("monitor", rid, split))
            await test(app, pilot)
    asyncio.run(go())


def set_train(run, train_path):
    spec = json.loads((run / "spec.json").read_text())
    (run / "spec.json").write_text(json.dumps({**spec, "train_path": train_path}))


def model(root):
    found = runs.scan(root)
    return explorer.index(explorer.build(found, runs.trees(root, found)))


def fake_root(tmp_path):
    set_train(make_run(tmp_path, "alpha", tree_id=TREE, tree_here=True, splits=("dev", "test")),
              "datasets/My Custom Dataset/train.pubtator")
    make_run(tmp_path, "alpha_eval", stamp="2026-10-21_10-00-00", kind="eval", tree_id=TREE, splits=("test",))
    make_run(tmp_path, "alpha_eval", stamp="2026-10-22_10-00-00", kind="eval", tree_id=TREE, splits=("test",),
             status="failed", metrics={"test": None})
    make_run(tmp_path, "boom", stamp="2026-10-19_10-00-00", status="failed", tree_id=None, metrics={"dev": None})
    make_run(tmp_path, "live", stamp="2026-10-23_14-10-00", status="running", tree_id=None, metrics={"dev": None})
    return tmp_path


def labels(index, key):
    return [c.label for c in index[key][0].children]


def test_model_groups_trees_evaluations_and_loose_runs(tmp_path):
    idx = model(fake_root(tmp_path))
    assert [n.label for n in idx[("group", "models")][0].children] == ["My Custom Dataset"]
    tree = idx[("tree", TREE)][0]
    assert tree.label == "alpha" and tree.state == "COMPLETED" and tree.note == ""
    # the same eval name twice on one tree: told apart by start time, kept as two records
    assert [c.label for c in tree.children] == ["test · alpha_eval · 10-22 10:00", "test · alpha_eval · 10-21 10:00",
                                                "dev", "test"]
    assert [c.state for c in tree.children] == ["FAILED", "COMPLETED", "COMPLETED", "COMPLETED"]
    assert labels(idx, ("group", "active")) == ["live"] and idx[("run", "2026-10-23_14-10-00_live")][0].state \
        == "UNVERIFIED"
    assert labels(idx, ("group", "no-tree")) == ["boom"]


def test_model_edge_cases(tmp_path):
    assert labels(explorer.index(explorer.build([], [])), ("group", "models")) == []
    make_run(tmp_path, "dup", tree_id="aaaa1111")  # trained, tree not on disk; dataset datasets/x
    make_run(tmp_path, "dup", stamp="2026-10-21_10-00-00", tree_id="bbbb2222", tree_here=True)
    make_run(tmp_path, "evalonly", stamp="2026-10-22_10-00-00", kind="eval", tree_id="cccc3333", tree_here=True)
    make_run(tmp_path, "broken", stamp="2026-10-18_10-00-00", tree_id=None, corrupt=("status.json", "spec.json"))
    lone = Path(tmp_path, "outputs/saved_trees/xmodel_lone")
    lone.mkdir(parents=True)
    (lone / "xmodel.json").write_text(json.dumps({"tree_id": "dddd4444", "initial_labels": {"len": 7}}))
    (lone / "xmodel.pkl").write_bytes(b"")
    idx = model(tmp_path)
    assert labels(idx, ("dataset", "datasets/x")) == ["dup · bbbb2222", "dup · aaaa1111"], "duplicate names"
    assert idx[("tree", "aaaa1111")][0].note == "missing"
    unknown = idx[("dataset", None)][0]
    assert unknown.label == explorer.UNKNOWN_DATASET
    assert [c.label for c in unknown.children] == ["xmodel_cccc3333", "xmodel_lone"], "no training run: no dataset"
    assert idx[("tree", "dddd4444")][0].children == (), "a tree without runs has no evaluations"
    broken = idx[("run", "2026-10-18_10-00-00_broken")][0]
    assert broken.state == "UNKNOWN" and idx[broken.key][1] == (("group", "no-tree"),)


def test_moved_tree_found_by_tree_id(tmp_path):
    make_run(tmp_path, "alpha", tree_id=TREE, tree_here=True)
    Path(tmp_path, f"outputs/saved_trees/xmodel_{TREE}/xmodel.json").write_text(json.dumps({"tree_id": TREE}))
    shutil.move(Path(tmp_path, f"outputs/saved_trees/xmodel_{TREE}"), Path(tmp_path, "outputs/saved_trees/renamed"))
    (tree,) = runs.trees(tmp_path, runs.scan(tmp_path))
    assert (tree.path, tree.here, tree.recorded_path) == ("outputs/saved_trees/renamed", True,
                                                          f"outputs/saved_trees/xmodel_{TREE}")
    assert json.loads(Path(tmp_path, "outputs/runs/2026-10-20_13-36-20_alpha/tree.json").read_text())[
        "xmodel_path"] == f"outputs/saved_trees/xmodel_{TREE}", "the recorded path is never rewritten"
    assert model(tmp_path)[("tree", TREE)][0].note == "moved"


def explorer_key(app):
    return app.screen.query_one(Explorer).selected_key()


async def goto(pilot, app, key):
    tree = app.screen.query_one(Explorer)
    for _ in range(50):
        if tree.selected_key() == key:
            return
        await pilot.press("down")
    raise AssertionError(f"{key} not reached")


def test_keyboard(tmp_path):
    async def check(app, pilot):
        screen = app.screen
        assert isinstance(screen, DashboardScreen) and isinstance(app.focused, Explorer)
        assert explorer_key(app) == ("group", "models")
        await pilot.press("tab")
        assert explorer_key(app) == ("dataset", "datasets/My Custom Dataset")
        await pilot.press("tab", "shift+tab")
        assert explorer_key(app) == ("dataset", "datasets/My Custom Dataset")
        await pilot.press("tab")  # the tree: collapsed by default
        node = app.focused.cursor_node
        assert not node.is_expanded
        await pilot.press("right")
        assert node.is_expanded
        await pilot.press("left")
        assert not node.is_expanded
        await pilot.press("left")
        assert explorer_key(app) == ("dataset", "datasets/My Custom Dataset"), "left on a collapsed node: parent"
        await pilot.press("tab", "right", "tab", "tab", "tab")  # third eval: dev of alpha
        assert explorer_key(app) == ("eval", "2026-10-20_13-36-20_alpha", "dev")
        await pilot.press("enter")
        assert isinstance(app.focused, Description)
        assert text(screen.query_one("#description-text", Static)).startswith("dev · alpha\n")
        await pilot.press("ctrl+tab")
        assert app.focused.id == "active-runs"
        await pilot.press("ctrl+tab")
        assert isinstance(app.focused, Explorer)
        await pilot.press("ctrl+shift+tab")
        assert app.focused.id == "active-runs"
        await pilot.press("left_square_bracket", "escape")
        assert isinstance(app.focused, Explorer), "] / [ fall back for terminals without ctrl+tab; esc: explorer"

        await pilot.press("enter", "left", "right")  # Results on the tree / Run monitor on the evaluation's split
        assert opened == [("results", TREE, "2026-10-20_13-36-20_alpha"),
                          ("monitor", "2026-10-20_13-36-20_alpha", "dev")]
        await pilot.press("escape", "up", "up", "up", "enter", "left", "right")  # the tree: its training run
        assert explorer_key(app) == ("tree", TREE)
        assert opened[2:] == [("results", TREE, "2026-10-20_13-36-20_alpha"),
                              ("monitor", "2026-10-20_13-36-20_alpha", None)]
        await pilot.press("ctrl+tab")  # Active runs: enter opens the run
        assert app.focused.id == "active-runs" and app.focused.row_count == 1
        await pilot.press("enter")
        assert opened[-1] == ("monitor", "2026-10-23_14-10-00_live", None)
    opened = []
    run_app(fake_root(tmp_path), check, opened=opened)


def snapshot(root):
    return {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in Path(root).rglob("*") if p.is_file()}


def test_sync_keeps_selection_and_writes_nothing(tmp_path):
    root = fake_root(tmp_path)
    before = snapshot(root)

    async def check(app, pilot):
        screen = app.screen
        tree = screen.query_one(Explorer)

        async def rescan():
            app.apply_scan(app.scan())
            await pilot.pause()
        await goto(pilot, app, ("tree", TREE))
        await pilot.press("right")
        await goto(pilot, app, ("eval", "2026-10-21_10-00-00_alpha_eval", "test"))
        sig = tree.sig
        await rescan()
        assert tree.sig is sig, "nothing changed: no rebuild"

        # new evaluation of the tree, new tree in another dataset, the live run completes
        make_run(root, "alpha_eval2", stamp="2026-10-24_10-00-00", kind="eval", tree_id=TREE, splits=("dev",))
        set_train(make_run(root, "beta", stamp="2026-10-24_11-00-00", tree_id="beef0001", tree_here=True),
                  "datasets/BioRED-v2/train.pubtator")
        live = root / "outputs/runs/2026-10-23_14-10-00_live"
        (live / "status.json").write_text(json.dumps({"status": "completed", "started": "2026-10-23T14:10:00",
                                                      "finished": "2026-10-23T15:00:00"}))
        for _ in range(3):  # bursts of rescans add nothing twice
            await rescan()
        idx = tree.index
        assert explorer_key(app) == ("eval", "2026-10-21_10-00-00_alpha_eval", "test"), "selection kept"
        assert tree.cursor_node.parent.is_expanded, "expanded tree kept"
        assert labels(idx, ("tree", TREE))[0] == "dev · alpha_eval2"
        assert labels(idx, ("group", "models")) == ["BioRED-v2", "My Custom Dataset"]
        assert ("group", "active") not in idx and "2026-10-23_14-10-00_live" not in screen.active_ids
        assert labels(idx, ("group", "no-tree")) == ["live", "boom"]
        assert len(idx) == len({k for k in idx}) and screen.query_one("#active-runs", DataTable).row_count == 0

        # the selected evaluation's run directory is renamed outside the TUI: cursor falls back to its tree
        shutil.move(root / "outputs/runs/2026-10-21_10-00-00_alpha_eval", root / "outputs/runs/moved_eval")
        await rescan()
        assert explorer_key(app) == ("tree", TREE)
        assert sum(x.startswith("test · alpha_eval") for x in labels(tree.index, ("tree", TREE))) == 2, "found again"
        # the tree directory is deleted: the tree stays (runs record it) and says missing
        shutil.rmtree(root / f"outputs/saved_trees/xmodel_{TREE}")
        await rescan()
        assert tree.index[("tree", TREE)][0].note == "missing"
        assert "missing: outputs/saved_trees" in text(screen.query_one("#description-text", Static))
        await pilot.press("enter", "left", "right", "escape")  # still navigable after refresh
        assert app.screen is screen and isinstance(app.focused, Explorer)
        assert [o[0] for o in opened] == ["results", "monitor"]

    opened = []
    run_app(root, check, opened=opened)
    after = snapshot(root)
    changed = {p for p in before if p in after and before[p] != after[p]}
    assert changed == {root / "outputs/runs/2026-10-23_14-10-00_live/status.json"}, "only the test's own write"
    made = ("2026-10-24_10-00-00_alpha_eval2", "2026-10-24_11-00-00_beta", "xmodel_beef0001", "moved_eval")
    assert all(set(p.relative_to(root).parts) & set(made) for p in set(after) - set(before)), "no new files"


def test_empty_workspace_help_rescan_quit_theme(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    async def missing(app, pilot):
        assert "outputs/runs/ not found" in text(app.screen.query_one("#description-text", Static))
    run_app(tmp_path / "nowhere", missing)
    (tmp_path / "empty/outputs/runs").mkdir(parents=True)

    async def empty(app, pilot):
        assert "No saved trees yet" in text(app.screen.query_one("#description-text", Static))
        await pilot.press("question_mark")
        assert app.screen.query("HelpPanel")
        await pilot.press("escape")  # closes help
        assert not app.screen.query("HelpPanel")
        await pilot.press("ctrl+r")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any("Rescanned" in str(n.message) for n in app._notifications)
        await pilot.press("ctrl+q")  # first press warns only
        await pilot.pause()
        assert app.is_running and any("again to quit" in str(n.message) for n in app._notifications)
        app.theme = "nord"
        await pilot.pause()
    run_app(tmp_path / "empty", empty)
    assert '"nord"' in settings_path().read_text()

    async def restored(app, pilot):
        assert app.theme == "nord"
    run_app(tmp_path / "empty", restored)
