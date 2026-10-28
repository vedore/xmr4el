"""Shared explorer: two-row tree entries (title from recorded metadata, date and collision-safe short id), gap rows,
full-item hit area, search; one presentation on the Dashboard and in Results."""
import asyncio
import json

import pytest

pytest.importorskip("textual")

from fake_runs import make_run  # noqa: E402
from xmr4el_tui import explorer, runs  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import DashboardScreen  # noqa: E402
from xmr4el_tui.screens.results_screen import ResultsScreen  # noqa: E402
from xmr4el_tui.widgets.explorer import Explorer, ResultsExplorer  # noqa: E402

T1, T2, T3 = "abcdef1234aaaa", "abcdef1299bbbb", "ffff0003cccc"  # T1 and T2 share 8 characters
A = ("eval", "2026-10-20_10-00-00_alpha", "dev")
B = ("eval", "2026-10-21_10-00-00_alpha", "dev")


def root_with_twins(root):
    make_run(root, "alpha", stamp="2026-10-20_10-00-00", tree_id=T1, tree_here=True,
             model_config={"features": "sapbert_char_context", "depth": 2})
    make_run(root, "alpha", stamp="2026-10-21_10-00-00", tree_id=T2, tree_here=True,
             model_config={"features": "tfidf", "depth": 3})
    make_run(root, "gamma", stamp="2026-10-22_10-00-00", tree_id=T3, tree_here=True, model_config={"other": 1})
    return root


def build(root):
    found = runs.scan(root)
    return explorer.index(explorer.build(found, runs.trees(root, found)))


def test_tree_identity_lines(tmp_path):
    idx = build(root_with_twins(tmp_path))
    t1, t2, t3 = (idx[("tree", t)][0] for t in (T1, T2, T3))
    assert (t1.label, t2.label) == ("alpha", "alpha"), "same names: identity is the key, not the label"
    assert explorer.tree_title(t1) == "alpha · SapBERT D2" and explorer.tree_title(t2) == "alpha · TF-IDF D3"
    assert explorer.tree_title(t3) == "gamma", "nothing recorded: no descriptor invented"
    assert (t1.short, t2.short, t3.short) == ("abcdef123", "abcdef129", "ffff0003"), "collisions grow, others stay 8"
    assert t1.detail == "Oct 20 · abcdef123"
    assert explorer.short_ids(["x" * 12]) == {"x" * 12: "x" * 8}


def test_prune_and_search_text(tmp_path):
    roots = explorer.build(*(lambda f: (f, runs.trees(tmp_path, f)))(runs.scan(root_with_twins(tmp_path))))
    shown, opened = explorer.prune(roots, explorer.matches("tf-idf"))
    idx = explorer.index(shown)
    assert ("tree", T2) in idx and ("tree", T1) not in idx and B in idx, "a matching tree keeps its evaluations"
    assert ("dataset", "datasets/x") in opened and ("tree", T2) not in opened
    assert ("tree", T1) in explorer.index(explorer.prune(roots, explorer.matches("abcdef123 dev"))[0])
    assert explorer.prune(roots, explorer.matches("nothing"))[0] == ()


def run_app(root, test, size=(120, 36)):
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await test(app, pilot)
    asyncio.run(go())


async def settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


def row_text(tree, y):
    return "".join(s.text for s in tree.render_line(y - int(tree.scroll_y)))


def test_two_rows_gaps_and_full_item_clicks(tmp_path):
    async def check(app, pilot):
        dash = app.screen.query_one(Explorer)
        dash_rows = {}
        for t in (T1, T2):
            y = dash.row_of(dash.nodes[("tree", t)])
            dash_rows[t] = (row_text(dash, y), row_text(dash, y + 1))
        assert "alpha · SapBERT D2" in dash_rows[T1][0] and "Oct 20 · abcdef123" in dash_rows[T1][1]
        kinds = [sub for _, sub in dash.rows]
        assert kinds.count(-1) == 2 and kinds.count(1) == 3, "a gap above each later tree; three trees on two rows"
        assert row_text(dash, dash.row_of(dash.nodes[("tree", T2)]) - 1).strip("│ ") == "", "gap row: guides only"

        # a click on a tree's second row selects that tree (the whole item is its hit area)
        dash.nodes[("tree", T2)].collapse()
        await pilot.pause()
        y = dash.row_of(dash.nodes[("tree", T2)]) + 1 - int(dash.scroll_y)
        dx, dy = dash.content_region.x - dash.region.x, dash.content_region.y - dash.region.y
        await pilot.click(Explorer, offset=(dx + 20, dy + y))
        await settle(app, pilot)
        assert dash.selected_key() == ("tree", T2)

        app.push_screen(ResultsScreen(A))
        await settle(app, pilot)
        await settle(app, pilot)
        res = app.screen.query_one(ResultsExplorer)
        y = res.row_of(res.nodes[("tree", T1)])
        assert explorer.tree_title(res.index[("tree", T1)][0]) in row_text(res, y)
        assert row_text(res, y + 1).strip("│ ") == dash_rows[T1][1].strip("│ "), "same id line on both"
        assert isinstance(res, Explorer) and type(dash) is Explorer, "Results uses the Dashboard's widget"
    run_app(root_with_twins(tmp_path), check, size=(200, 40))


def test_search_keeps_expansion_and_members(tmp_path):
    async def check(app, pilot):
        tree = app.screen.query_one(Explorer)
        tree.nodes[("tree", T3)].expand()
        tree.nodes[("tree", T1)].collapse()
        await pilot.pause()
        before = set(tree.expanded)
        await pilot.press("slash", "t", "f", "-", "i", "d", "f")
        assert tree.typing and tree.border_subtitle == "/tf-idf▏"
        assert set(tree.index) == {("group", "models"), ("dataset", "datasets/x"), ("tree", T2), B}
        await pilot.press("r")  # typed, not the Results key
        assert isinstance(app.screen, DashboardScreen) and tree.query == "tf-idfr" and "no match" in tree.border_subtitle
        await pilot.press("backspace", "enter")
        assert not tree.typing and tree.query == "tf-idf"
        await pilot.press("escape")
        assert tree.query == "" and tree.border_subtitle == "" and tree.expanded == before, "expansion as chosen"
        assert ("tree", T3) in tree.index and not tree.nodes[("tree", T1)].is_expanded

        app.push_screen(ResultsScreen(A))
        await settle(app, pilot)
        await settle(app, pilot)
        screen, res = app.screen, app.screen.query_one(ResultsExplorer)
        res.move_cursor(res.nodes[B])
        await pilot.press("space")
        assert screen.members == [B]
        await pilot.press("slash", "g", "a", "m")
        assert B not in res.index and screen.members == [B], "hidden by a search: still compared"
        await pilot.press("escape")
        assert isinstance(app.screen, ResultsScreen) and B in res.index, "esc clears the search first"
        await pilot.press("escape")
        assert isinstance(app.screen, DashboardScreen)
    run_app(root_with_twins(tmp_path), check)


def test_model_config_is_part_of_the_cache_key(tmp_path):
    root = root_with_twins(tmp_path)
    cache = {}
    runs.scan(root, cache)
    path = root / "outputs/runs/2026-10-22_10-00-00_gamma/model_config.json"
    path.write_text(json.dumps({"features": "tfidf", "depth": 1}))
    gamma = next(r for r in runs.scan(root, cache) if r.name == "gamma")
    assert (gamma.features, gamma.depth) == ("tfidf", 1)


def test_search_mounts_only_matches_and_survives_restyle(tmp_path):
    """No match: nothing mounted, nothing drawn; restyle, the filter and a rescan still work (R06)."""
    async def check(app, pilot):
        app.push_screen(ResultsScreen(A))
        await settle(app, pilot)
        await settle(app, pilot)
        screen, res = app.screen, app.screen.query_one(ResultsExplorer)
        await pilot.press("slash", "z", "z", "z")
        assert res.index == {} and res.nodes == {} and not res.root.children
        res.restyle()
        screen.query_one("#show-all").value = True
        await settle(app, pilot)
        app.apply_scan(app.repo.snapshot(fresh=True))
        await settle(app, pilot)
        assert res.nodes == {} and isinstance(app.screen, ResultsScreen)
        await pilot.press("backspace", "backspace", "backspace", "a", "l")
        assert set(res.nodes) == set(res.index) and A in res.nodes
    run_app(root_with_twins(tmp_path), check)


def test_same_minute_evaluations_get_distinct_labels(tmp_path):
    """Two evaluations of one tree by one spec, a second apart: still told apart (R10)."""
    make_run(tmp_path, "alpha", stamp="2026-10-20_10-00-00", tree_id=T1, tree_here=True)
    for stamp in ("2026-10-21_10-00-00", "2026-10-21_10-00-01"):
        make_run(tmp_path, "ev", stamp=stamp, kind="eval", tree_id=T1)
    labels = [n.label for n, _ in build(tmp_path).values() if n.kind == "eval"]
    assert len(labels) == len(set(labels)) == 3
    assert "dev · ev · 2026-10-21_10-00-01_ev" in labels
