"""Dashboard: explorer model (trees by dataset, evaluations, runs without a tree), keyboard, file-system sync without
restart, that nothing under the root is written; app keys (help, rescan, quit, theme)."""
import asyncio
import json
import shutil
import socket
import threading
from pathlib import Path

import pytest

pytest.importorskip("textual")

from fake_runs import bad_fields_run, make_run, patch_json  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from xmr4el_tui import explorer, process, runs  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dashboard_details import banner_text  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import ActiveRuns, DashboardScreen  # noqa: E402
from xmr4el_tui.widgets.description import Description  # noqa: E402
from xmr4el_tui.widgets.explorer import EXPLORER_MIN, Explorer, explorer_width  # noqa: E402
from xmr4el_tui.widgets.top_bar import TopBar  # noqa: E402

TREE = "8528b624aaaa"


def text(widget):
    return str(widget.render())


def run_app(root, test, size=(120, 36), opened=None):
    """`opened`: list collecting what Results / Run monitor would open (their screens are not built yet)."""
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await app.workers.wait_for_complete()  # the first scan starts on mount
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


def test_wrong_field_types_keep_the_dashboard_usable(tmp_path):
    """Every explorer item is described without an exception; the bad run shows its problems."""
    root = fake_root(tmp_path)
    bad_fields_run(root)
    ev = make_run(root, "odd_eval", stamp="2026-10-25_10-00-00", kind="eval", tree_id=TREE, splits=("dev",))
    patch_json(ev / "metrics_dev.json", **{"acc@1": "0.9", "search": [], "strings": {"all": []}, "ms_per_query": "x"})
    patch_json(ev / "provenance.json", packages=["numpy"], inputs=[])
    patch_json(ev / "status.json", finished="2026-10-25T10:05:00+02:00")

    async def check(app, pilot):
        tree, desc, seen = app.screen.query_one(Explorer), app.screen.query_one(Description), {}
        for _ in range(40):
            await pilot.press("right", "down")
            seen[tree.selected_key()] = desc.text()
        assert ("eval", ev.name, "dev") in seen and "acc@1 " in seen[("eval", ev.name, "dev")]
        assert "status is not a string" in seen[("run", "2026-10-20_13-36-20_bad")]
        assert app.is_running
    run_app(root, check)


def test_banners_of_unfinished_runs(tmp_path):
    ident = {"pid": 4242, "host": "h", "runtime": "r", "uid": 1, "start": "s", "pgid": 4242, "sid": 4242}
    run = runs.read_run(make_run(tmp_path, "a", status="running", tree_id=None, metrics={"dev": None},
                                 identity=ident), tmp_path)
    other = runs.replace(run, live=process.Live("UNVERIFIED", "another machine or container", other_runtime=True))
    assert "another machine or container" in banner_text(other).plain
    gone = runs.replace(run, live=process.Live("LOST", "process 4242 is gone"))
    assert "Process 4242 is gone" in banner_text(gone).plain


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
        assert screen.query_one(Description).text().startswith("dev · alpha\n")
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
            app.apply_scan(app.repo.snapshot())
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
        assert "missing: outputs/saved_trees" in screen.query_one(Description).text()
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


def test_scans_never_overlap_and_land_in_order(tmp_path):
    """Scan A is held inside the repository; requests meanwhile do not start a second scan beside it but make A's
    thread scan once more; snapshots land in scan order; a cancelled (quitting) scan publishes nothing."""
    async def check(app, pilot):
        gate, entered = threading.Event(), threading.Event()
        made, applied, running, overlap, real = [], [], [0], [], app.repo.snapshot

        def held():
            running[0] += 1
            overlap.append(running[0])
            if not made:
                entered.set()
                gate.wait(10)
            made.append(real())
            running[0] -= 1
            return made[-1]
        app.repo.snapshot = held
        app.apply_scan = lambda snapshot, announce=False: applied.append(snapshot)
        app.rescan()
        await asyncio.to_thread(entered.wait, 10)
        for _ in range(3):
            await app.rescan().wait()  # returns at once: coalesced into the held scan's thread
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(made) == 2 and applied == made and max(overlap) == 1

        made.clear(), gate.clear(), entered.clear(), applied.clear()
        worker = app.rescan()
        await asyncio.to_thread(entered.wait, 10)
        worker.cancel()
        gate.set()
        assert await asyncio.to_thread(app.scan_lock.acquire, timeout=10), "the scan thread finished"
        app.scan_lock.release()
        await pilot.pause()
        assert len(made) == 1 and applied == []
    run_app(fake_root(tmp_path), check)


def test_empty_workspace_help_rescan_quit_style(tmp_path):
    async def missing(app, pilot):
        assert "outputs/runs/ not found" in app.screen.query_one(Description).text()
    run_app(tmp_path / "nowhere", missing)
    (tmp_path / "empty/outputs/runs").mkdir(parents=True)

    async def empty(app, pilot):
        assert "No saved trees yet" in app.screen.query_one(Description).text()
        await pilot.press("question_mark")
        assert app.screen.query("HelpPanel")
        await pilot.press("escape")  # closes help
        assert not app.screen.query("HelpPanel")
        await pilot.press("ctrl+r")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any("Rescanned" in str(n.message) for n in app._notifications)
        await pilot.press("ctrl+c")  # first press warns only
        await pilot.pause()
        assert app.is_running and any("again to quit" in str(n.message) for n in app._notifications)
        assert app.theme == "xmr4el" and "Theme" not in [c.title for c in app.get_system_commands(app.screen)]
    run_app(tmp_path / "empty", empty, size=(60, 20))


def test_header_centred_without_system_info(tmp_path):
    for size in ((120, 36), (81, 24), (60, 20)):
        async def check(app, pilot, w=size[0]):
            line = app.screen.query_one(TopBar).render().plain
            assert line.index("XMR4EL") == (w - len("XMR4EL")) // 2, "centred on the whole width"
            assert ("Results" in line) == (w >= 100), "areas only where they do not reach the title"
            assert socket.gethostname() not in line and "trees" not in line and "active" not in line
            if w < 100:  # the 40-cell help panel covers the centre
                return
            await pilot.press("question_mark")  # help docks right, the title stays centred on the app
            await pilot.pause()
            assert app.screen.query_one(TopBar).render().plain.index("XMR4EL") == (w - len("XMR4EL")) // 2
        run_app(fake_root(tmp_path / str(size[0])), check, size=size)


def bg(strip):
    """Background of a rendered line's last cell (the far right of the row)."""
    return list(strip)[-1].style.bgcolor.triplet.hex


def test_explorer_resize_collapse_and_narrow(tmp_path):
    root = fake_root(tmp_path)

    async def check(app, pilot):
        screen, tree = app.screen, app.screen.query_one(Explorer)
        assert tree.outer_size.width == 26, "22% of 120"
        await pilot.press("greater_than_sign", "greater_than_sign")
        assert tree.outer_size.width == 30
        await pilot.press(*["less_than_sign"] * 20)
        assert tree.outer_size.width == EXPLORER_MIN
        await pilot.press(*["greater_than_sign"] * 40)
        assert tree.outer_size.width == 60, "at most half the screen"
        await pilot.mouse_down(Explorer, offset=(59, 5))  # drag the right border
        await pilot.hover(None, offset=(39, 5))
        await pilot.mouse_up(None, offset=(39, 5))
        assert tree.outer_size.width == 40 and isinstance(app.focused, Explorer)
        await pilot.press("ctrl+b")  # collapse: focus moves on, panels cycle without it
        assert not tree.display and isinstance(app.focused, Description)
        await pilot.press("ctrl+tab", "ctrl+tab")
        assert isinstance(app.focused, Description)
        await pilot.press("escape")  # esc restores it, same width
        assert tree.display and isinstance(app.focused, Explorer) and tree.outer_size.width == 40
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert tree.outer_size.width == 20 and screen.query_one(Description).outer_size.width >= 40
        await pilot.press("down", "enter", "down")  # still navigable when narrow
        assert isinstance(app.focused, Description)
    run_app(root, check)
    assert [explorer_width(None, w) for w in (40, 60, 80, 120, 200)] == [18, 18, 18, 26, 44]
    assert [explorer_width(999, w) for w in (40, 60, 80, 120)] == [18, 20, 40, 60]


async def goto_row(pilot, desc, key):
    for _ in range(50):
        if desc.keys[desc.highlighted] == key:
            return
        await pilot.press("down")
    raise AssertionError(f"{key} not reached")


def colours(strip):
    return {seg.style.color.triplet.hex for seg in strip if seg.text.strip() and seg.style.color}


def test_branch_guides_stay_lit(tmp_path):
    async def check(app, pilot):
        tree = app.screen.query_one(Explorer)
        await goto(pilot, app, ("tree", TREE))
        await pilot.press("right")
        sibling = tree.cursor_line + 2 - int(tree.scroll_offset.y)  # second evaluation of the tree
        assert "#2cf5c5" in colours(tree.render_line(sibling)), "on the tree: its evaluations' guides lit"
        await pilot.press("down")  # on the first evaluation (a leaf): the branch stays lit
        assert "#2cf5c5" in colours(tree.render_line(sibling))
        await pilot.press("left")  # back on the tree (expanded): its own children lit
        assert explorer_key(app) == ("tree", TREE) and tree.lit is None
        await pilot.press("left")  # collapsed: its dataset's branch is lit
        assert tree.lit is tree.cursor_node.parent and tree.cursor_node.parent._selected
    run_app(fake_root(tmp_path), check)


def test_full_row_selection_and_description_rows(tmp_path):
    root = fake_root(tmp_path)

    async def check(app, pilot):
        screen, tree = app.screen, app.screen.query_one(Explorer)
        desc = screen.query_one(Description)
        await goto(pilot, app, ("tree", TREE))
        focused = bg(tree.render_line(tree.cursor_line - int(tree.scroll_offset.y)))
        assert focused == "#2cf5c5", "focused cursor: accent across the whole row"
        assert bg(tree.render_line(tree.cursor_line + 1)) != focused
        await pilot.press("enter")
        assert isinstance(app.focused, Description) and desc.border_title == "Description — Trained tree"
        rows = [i for i in range(desc.option_count) if not desc.get_option_at_index(i).disabled]
        assert desc.highlighted == rows[0] and desc.keys[rows[0]] == "Tree id"
        blurred = bg(tree.render_line(tree.cursor_line - int(tree.scroll_offset.y)))
        assert blurred not in (focused, bg(tree.render_line(tree.cursor_line + 1))), "selected, not focused"
        assert "Evaluations" in desc.keys and desc.get_option_at_index(desc.keys.index("Evaluations")).disabled
        await pilot.press("down")
        assert desc.keys[desc.highlighted] == "Dataset"
        await goto_row(pilot, desc, "Run state")  # COMPLETED is green elsewhere; on the cursor row one colour
        line = desc.highlighted - int(desc.scroll_offset.y)
        assert {seg.style.color.triplet.hex for seg in desc.render_line(line) if seg.text.strip()} == {"#101418"}
        await pilot.press("home", "down")
        seen = [desc.highlighted]
        for _ in rows[2:]:
            await pilot.press("down")
            seen.append(desc.highlighted)
        assert seen == rows[1:], "every stop is a row, never a heading or a blank line"
        await pilot.press("up", "up")
        keep, text = desc.keys[desc.highlighted], desc.text()
        await pilot.press("x", "backspace", "delete", "space", "enter")  # nothing to edit
        assert desc.text() == text and screen.query_one(Explorer).selected_key() == ("tree", TREE)
        app.apply_scan(app.repo.snapshot())  # refresh of the same subject keeps the row
        await pilot.pause()
        assert desc.keys[desc.highlighted] == keep
        await pilot.press("ctrl+tab")  # Active runs: whole row, every column and the padding after them
        table = screen.query_one(ActiveRuns)
        assert table.row_count == 1 and bg(table.render_line(1)) == focused
        await pilot.press("ctrl+shift+tab")  # back to Description: its selection survived
        assert isinstance(app.focused, Description) and desc.keys[desc.highlighted] == keep
    run_app(root, check)


async def drag(pilot, start, end):
    await pilot.mouse_down(None, offset=start)
    await pilot.hover(None, offset=end)
    await pilot.mouse_up(None, offset=end)
    await pilot.pause()


async def drag_over(pilot, widget, word):
    """Drag from the first to the last cell of `word` where `widget` shows it."""
    vy, (_, shown) = next((y, v) for y, v in sorted(widget.plain.items()) if word in v[1])
    box = widget.content_region
    x, y = box.x + shown.index(word), box.y + vy - int(widget.scroll_offset.y)
    await drag(pilot, (x, y), (x + len(word) - 1, y))


def test_mouse_selection_copies(tmp_path):
    async def check(app, pilot):
        await goto(pilot, app, ("tree", TREE))
        desc, tree = app.screen.query_one(Description), app.screen.query_one(Explorer)
        await drag_over(pilot, desc, TREE)
        assert app.clipboard == TREE and any("Copied" in str(n.message) for n in app._notifications)
        await drag_over(pilot, app.screen.query_one(ActiveRuns), "UNVERIFIED")
        assert app.clipboard == "UNVERIFIED"
        await drag_over(pilot, tree, "My Custom")
        assert app.clipboard == "My Custom"
        await pilot.click(Description, offset=(5, 3))  # a click selects a row, copies nothing
        await drag(pilot, (tree.region.right - 1, 5), (tree.region.right + 9, 5))  # border drag: a resize only
        assert app.clipboard == "My Custom" and tree.outer_size.width == 36
    run_app(fake_root(tmp_path), check)
