"""Results screen: comparison levels and deltas (services.comparison), theme tokens and beam colours, explorer row
states, membership by keyboard and mouse, inspection, Dashboard integration."""
import asyncio
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("textual")

from rich.console import Console  # noqa: E402
from fake_runs import make_run  # noqa: E402
from xmr4el_tui import runs, theme  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import DashboardScreen  # noqa: E402
from xmr4el_tui.screens.results_screen import ResultsScreen  # noqa: E402
from xmr4el_tui.services import comparison as cmp  # noqa: E402
from xmr4el_tui.widgets.explorer import Explorer, ResultsExplorer  # noqa: E402
from xmr4el_tui.widgets.results_panels import InspectionView  # noqa: E402

T1, T2, T3, T4, T5 = "aaaa0001aaaa", "bbbb0002bbbb", "cccc0003cccc", "dddd0004dddd", "eeee0005eeee"
A = ("eval", "2026-10-20_10-00-00_alpha", "dev")       # baseline, B10
B = ("eval", "2026-10-20_11-00-00_beta", "dev")        # B20, VERIFIED (population equal by derivation)
C = ("eval", "2026-10-20_12-00-00_gamma", "dev")       # dirty commit: CAUTION
D = ("eval", "2026-10-20_13-00-00_delta", "dev")       # other test file contents: INCOMPATIBLE
E = ("eval", "2026-10-20_14-00-00_eps", "dev")         # B5
F = ("eval", "2026-10-21_10-00-00_alpha_b15", "dev")   # second evaluation of alpha's tree, B15


def search(beam):
    return {"search": {"beam_size": beam, "topk": 0, "knn_beta": 10.0, "leaf_candidates": 100}}


def results_root(root):
    make_run(root, "alpha", stamp="2026-10-20_10-00-00", tree_id=T1, tree_here=True, metrics={"dev": search(10)})
    make_run(root, "beta", stamp="2026-10-20_11-00-00", tree_id=T2, tree_here=True,
             metrics={"dev": {**search(20), "acc@1_all": 0.9004, "acc@1": 0.9346, "MRR": 0.9500, "labels": 9000}})
    make_run(root, "gamma", stamp="2026-10-20_12-00-00", tree_id=T3, tree_here=True, metrics={"dev": search(10)},
             git={"commit": "b10a922a7734f5844d8e30b9b928b740923a3299", "dirty": True})
    make_run(root, "delta", stamp="2026-10-20_13-00-00", tree_id=T4, tree_here=True, test_sha="sha-other",
             metrics={"dev": search(10)})
    make_run(root, "eps", stamp="2026-10-20_14-00-00", tree_id=T5, tree_here=True, metrics={"dev": search(5)})
    make_run(root, "alpha_b15", stamp="2026-10-21_10-00-00", kind="eval", tree_id=T1, metrics={"dev": search(15)})
    return root


def facts(root):
    found = runs.scan(root)
    return {("eval", r.id, e.split): cmp.read_facts(r, e, found) for r in found for e in r.evals}


# --- comparison service -----------------------------------------------------------------------------------------
def test_levels_and_reasons(tmp_path):
    f = facts(results_root(tmp_path))
    base = f[A]
    assert cmp.assess(base, base) is cmp.BASELINE
    b = cmp.assess(base, f[B])
    assert b.level == cmp.VERIFIED and b.population, "other vocabulary and beam: still comparable"
    c = cmp.assess(base, f[C])
    assert c.level == cmp.CAUTION and not c.population
    assert "clean git commit" in c.reason and "757 loaded rows" in c.matches and "acc@1, MRR" in c.affects
    d = cmp.assess(base, f[D])
    assert d.level == cmp.INCOMPATIBLE and "sha256 differs" in d.reason
    assert cmp.assess(base, f[F]).level == cmp.VERIFIED, "same tree, other beam"


def test_incompatible_and_hash_cases(tmp_path):
    f = facts(results_root(tmp_path))
    base, other = f[A], f[B]

    def with_metrics(x, **over):
        return cmp.Facts(**{**x.__dict__, "metrics": {**x.metrics, **over}})
    assert cmp.assess(base, cmp.Facts(**{**other.__dict__, "test_sha": None})).level == cmp.INCOMPATIBLE
    rows = cmp.assess(base, with_metrics(other, rows_total=758))
    assert rows.level == cmp.INCOMPATIBLE and "757 vs 758" in rows.reason
    assert cmp.assess(base, cmp.Facts(**{**other.__dict__, "metrics": None})).level == cmp.INCOMPATIBLE
    same = cmp.assess(with_metrics(base, population_sha256="p1"), with_metrics(other, population_sha256="p1"))
    assert same.level == cmp.VERIFIED and same.population
    differ = cmp.assess(with_metrics(base, population_sha256="p1"), with_metrics(other, population_sha256="p2"))
    assert differ.level == cmp.VERIFIED and not differ.population and "populations differ" in differ.notes[0]


def test_deltas(tmp_path):
    f = facts(results_root(tmp_path))
    entries = [f[A], f[B], f[C]]
    compats = [cmp.BASELINE, cmp.assess(f[A], f[B]), cmp.assess(f[A], f[C])]
    perf = dict(cmp.rows("Performance", entries, compats))
    assert perf["Acc@1_all"] == [("89.04%", None), ("90.04%", "+1.00"), ("89.04%", "+0.00")], "pp"
    assert perf["Acc@1"][1] == ("93.46%", "+1.00") and perf["Acc@1"][2] == ("92.46%", cmp.NO_DELTA), "Caution: withheld"
    assert perf["Acc@1_all"][2][1] == "+0.00" and perf["Coverage"][2][1] == "+0.00", "Caution keeps unrelated deltas"
    assert perf["MRR"][1] == ("0.9500", "+0.0041"), "MRR on its own scale"
    assert perf["Coverage"][1] == ("96.30%", "+0.00") and perf["Labels"][1] == ("9,000", None)
    gen = dict(cmp.rows("Generalization", entries, compats))
    assert gen == {}, "no breakdown recorded: no rows, never zeros"
    eff = dict(cmp.rows("Efficiency", entries, compats))
    assert all(d is None for cells in eff.values() for _, d in cells), "no timing deltas"
    assert list(eff) == ["Prediction time", "ms/query", "Eval time", "Candidates/query"], "measured only"
    specs = dict(cmp.rows("Specs", entries, compats))
    assert list(specs) == ["beam_size", "topk", "knn_beta", "leaf_candidates", "features"]
    assert specs["beam_size"] == [("10", None), ("20", None), ("10", None)], "settings: no deltas"
    rank_c = dict(cmp.rows("Ranking", entries, compats))
    assert rank_c["R@5"][2] == ("96.71%", cmp.NO_DELTA), "Caution: value kept, delta marked unavailable"
    assert list(cmp.GROUPS) == ["Performance", "Ranking", "Generalization", "Efficiency", "Specs"]
    missing = cmp.Facts(**{**f[B].__dict__, "metrics": {k: v for k, v in f[B].metrics.items() if k != "R@50"}})
    rank = dict(cmp.rows("Ranking", [f[A], missing], [cmp.BASELINE, compats[1]]))
    assert rank["R@50"][1] == ("N/A", None)


# --- theme ------------------------------------------------------------------------------------------------------
def test_beam_colours_and_theme_tokens():
    assert theme.beam_color(10) == theme.beam_color(10) != theme.beam_color(20)
    assert len({theme.beam_color(b) for b in (5, 10, 15, 20)}) == 4
    assert theme.beam_color(None) == theme.beam_color("10") == theme.color("beam-unknown")
    assert theme.color("cursor-row") == theme.mix(theme.color("accent"), theme.color("surface"), 0.4)
    assert all(k.startswith("xmr-") for k in theme.css_variables())
    hexes = re.compile(r"#[0-9A-Fa-f]{6}\b")
    pkg = Path(theme.__file__).parent
    offenders = [p.name for p in pkg.rglob("*") if p.suffix in (".py", ".tcss") and p.name != "theme.py"
                 and hexes.search(p.read_text())]
    assert offenders == [], "every colour lives in theme.py"


# --- screen -----------------------------------------------------------------------------------------------------
def run_app(root, test, size=(140, 40)):
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


async def open_results(app, pilot, key=A):
    app.push_screen(ResultsScreen(key))
    await settle(app, pilot)
    await settle(app, pilot)
    return app.screen, app.screen.query_one(ResultsExplorer)


async def show_incompatible(app, pilot, on=True):
    app.screen.query_one("#show-all").value = on
    await settle(app, pilot)


async def cursor_to(pilot, tree, key):
    for k in tree.index[key][1]:
        tree.nodes[k].expand()
    await pilot.pause()
    tree.move_cursor(tree.nodes[key])
    await pilot.pause()


def bg(tree, key):
    """Background of the row of `key` (a middle cell, past the guides) and of its first cell."""
    strip = tree.render_line(tree.row_of(tree.nodes[key]) - int(tree.scroll_y))
    segs = [s for s in strip if s.text]
    hexed = [(s.style.bgcolor.triplet.hex if s.style and s.style.bgcolor else None, s.text) for s in segs]
    return hexed[len(hexed) // 2][0], hexed[0]


def test_explorer_states_and_membership(tmp_path):
    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        accent, cursor = theme.color("accent").lower(), theme.color("cursor-row").lower()
        assert tree.selected_key() == A and screen.members == [] and tree.has_focus
        assert bg(tree, A)[1][1] == "▌", "cursor on the (selected) baseline: outline bar"
        assert bg(tree, A)[0] == accent
        assert screen.compat[D].level == cmp.INCOMPATIBLE and screen.compat[C].level == cmp.CAUTION
        assert D not in tree.nodes and C in tree.nodes, "Incompatible hidden by default"
        await show_incompatible(app, pilot)

        await cursor_to(pilot, tree, B)
        assert bg(tree, B)[0] == cursor and bg(tree, A)[0] == accent, "cursor: weak; baseline stays accent"
        assert bg(tree, A)[1][1] != "▌"
        assert screen.members == [], "moving the cursor changes nothing"
        await pilot.press("space")
        assert screen.members == [B] and bg(tree, B)[0] == accent and bg(tree, B)[1][1] == "▌"
        await pilot.press("up")
        assert tree.selected_key() != B and bg(tree, B)[0] == accent and bg(tree, B)[1][1] != "▌", "selection kept after the cursor leaves"
        for key in (A, D):
            await cursor_to(pilot, tree, key)
            await pilot.press("space")
        assert screen.members == [B], "baseline and incompatible cannot be added"
        for key in (C, F, E):
            await cursor_to(pilot, tree, key)
            await pilot.press("space")
        assert screen.members == [B, C, F], "three at most, in selection order; E refused"
        assert "⚠" in str(tree.nodes[C].label)
        assert "⚠" in str(tree.nodes[("tree", T3)].label), "the tree carries its evaluation's Caution"
        await cursor_to(pilot, tree, B)
        await pilot.press("space")
        assert screen.members == [C, F]
        await cursor_to(pilot, tree, E)
        await pilot.press("space")
        assert screen.members == [C, F, E], "a freed slot is usable at once"
        assert not any(b.key == "i" for b in screen.BINDINGS + tree.BINDINGS)

        table = screen.query_one("#analysis-table")
        assert len(table.columns) == 5 and table.row_count > 0, "metric + baseline + three challengers"
        await pilot.press("2")
        assert screen.group == 1 and table.get_row_at(0)[0] == "R@5"
        await cursor_to(pilot, tree, E)
        await pilot.press("space")
        assert screen.group == 1 and len(table.columns) == 4, "the group survives membership changes"
        summary = screen.query_one("#summary-text").content
        assert "clean git commit" in summary.renderables[2].columns[1]._cells[0].plain, "Caution reason below the rule"
    run_app(results_root(tmp_path), check)


def test_mouse_toggles_and_info_icon_inspects(tmp_path):
    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        await cursor_to(pilot, tree, B)
        await cursor_to(pilot, tree, A)
        dx, dy = tree.content_region.x - tree.region.x, tree.content_region.y - tree.region.y
        y = tree.row_of(tree.nodes[B]) - int(tree.scroll_y)
        text = "".join(s.text for s in tree.render_line(y))
        await pilot.click(ResultsExplorer, offset=(dx + text.index("dev") + 1, dy + y))
        assert screen.members == [B] and screen.inspecting is None
        await pilot.click(ResultsExplorer, offset=(dx + text.index(" B20"), dy + y))
        assert screen.members == [], "a second click removes"
        await pilot.click(ResultsExplorer, offset=(dx + text.index("ⓘ"), dy + y))
        await pilot.pause()
        assert screen.inspecting == B and screen.members == [], "ⓘ inspects, never toggles"
        tree_key = ("tree", T3)
        await pilot.press("escape")
        expanded = tree.nodes[tree_key].is_expanded
        y = tree.row_of(tree.nodes[tree_key]) - int(tree.scroll_y)
        text = "".join(s.text for s in tree.render_line(y))
        await pilot.click(ResultsExplorer, offset=(dx + text.index("gamma"), dy + y))
        assert tree.nodes[tree_key].is_expanded != expanded and screen.members == [], "a tree row only toggles"
    run_app(results_root(tmp_path), check)


def test_inspection_keeps_state(tmp_path):
    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        await cursor_to(pilot, tree, B)
        await pilot.press("space", "3")
        await cursor_to(pilot, tree, E)  # not a member
        await pilot.press("enter")
        assert screen.inspecting == E and tree.disabled and isinstance(app.focused, InspectionView)
        text = screen.query_one("#inspect-text").content.plain
        assert "Performance" in text and "Efficiency" in text and "VERIFIED" in text
        await pilot.press("space", "1")  # explorer inactive, groups ignored
        assert screen.members == [B] and screen.group == 2
        await pilot.press("escape")
        assert screen.inspecting is None and not tree.disabled and tree.has_focus
        assert tree.selected_key() == E and screen.members == [B] and screen.group == 2
        for key in (A, B, C):  # baseline, member, Caution
            await cursor_to(pilot, tree, key)
            await pilot.press("enter")
            text = screen.query_one("#inspect-text").content.plain
            assert ("the baseline" in text) == (key == A) and ("CAUTION" in text) == (key == C)
            await pilot.press("escape")
        await pilot.press("escape")
        assert isinstance(app.screen, DashboardScreen), "esc from Results: back to the Dashboard"
    run_app(results_root(tmp_path), check)


def test_dashboard_opens_fresh_sessions(tmp_path):
    async def check(app, pilot):
        dash = app.screen
        tree = dash.query_one(Explorer)
        tree.nodes[("tree", T2)].expand()
        await pilot.pause()
        tree.move_cursor(tree.nodes[B])
        await pilot.press("r")
        await settle(app, pilot)
        screen = app.screen
        assert isinstance(screen, ResultsScreen) and screen.base == B
        assert screen.names[B].startswith("beta")
        await cursor_to(pilot, screen.query_one(ResultsExplorer), E)
        await pilot.press("space", "escape")
        assert app.screen is dash and tree.selected_key() == B, "Dashboard selection kept"
        tree.move_cursor(tree.nodes[("tree", T1)])  # two evaluations: no baseline chosen for the user
        await pilot.press("r")
        assert app.screen is dash and tree.nodes[("tree", T1)].is_expanded
        tree.move_cursor(tree.nodes[("tree", T5)])  # one evaluation: that one
        await pilot.press("r")
        await settle(app, pilot)
        assert app.screen.base == E and app.screen.members == [], "a new session: no challengers"
    run_app(results_root(tmp_path), check)


def test_narrow_terminal_and_vanished_challenger(tmp_path):
    root = results_root(tmp_path)

    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        for key in (B, F):
            await cursor_to(pilot, tree, key)
            await pilot.press("space")
        table = screen.query_one("#analysis-table")
        assert table.virtual_size.width > table.size.width, "narrow: the table scrolls, no evaluation hidden"
        assert len(table.columns) == 4
        assert tree.size.width >= 20
        (Path(root) / "outputs/runs" / B[1]).rename(Path(root) / "outputs/runs" / ".gone")
        app.apply_scan(app.repo.snapshot())
        await settle(app, pilot)
        assert screen.members == [F] and len(table.columns) == 3
        status = Path(root) / "outputs/runs" / F[1] / "status.json"
        status.write_text(json.dumps({**json.loads(status.read_text()), "status": "failed"}))
        app.apply_scan(app.repo.snapshot())
        await settle(app, pilot)
        assert screen.members == [F], "a FAILED run keeps its written metrics"
    run_app(root, check, size=(80, 24))


# --- Results V1 refinement ----------------------------------------------------------------------------------------
G_DEV, G_TEST = ("eval", "2026-10-22_10-00-00_mixed", "dev"), ("eval", "2026-10-22_10-00-00_mixed", "test")
N = ("eval", "2026-10-22_11-00-00_nosha", "dev")


def filter_root(root):
    """results_root plus a tree with one comparable and one incompatible evaluation, and one without a test sha."""
    results_root(root)
    make_run(root, "mixed", stamp="2026-10-22_10-00-00", tree_id="ffff0006ffff", tree_here=True,
             splits=("dev", "test"), metrics={"dev": search(10), "test": search(10)})
    make_run(root, "nosha", stamp="2026-10-22_11-00-00", tree_id="ffff0007ffff", tree_here=True, test_sha=None,
             metrics={"dev": search(10)})
    return root


def test_show_incompatible_filter(tmp_path):
    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        box = screen.query_one("#show-all")
        assert box.value is False and screen.compat[N].level == cmp.INCOMPATIBLE
        assert screen.compat[G_TEST].level == cmp.INCOMPATIBLE and screen.compat[G_DEV].level == cmp.VERIFIED
        assert {B, C, E, F, G_DEV} <= set(tree.index), "Verified and Caution shown"
        assert not {D, G_TEST, N} & set(tree.index), "Incompatible and unknown dataset identity hidden"
        assert ("tree", T4) not in tree.index and ("tree", "ffff0007ffff") not in tree.index, "no candidate: no tree"
        assert ("tree", "ffff0006ffff") in tree.index, "per evaluation: the tree stays for its dev"
        await cursor_to(pilot, tree, B)
        await pilot.press("space", "4")
        await cursor_to(pilot, tree, C)
        await pilot.press("space")
        await show_incompatible(app, pilot)
        assert {D, G_TEST, N} <= set(tree.index) and screen.members == [B, C] and screen.group == 3
        assert "✗" in str(tree.nodes[D].label) and "✗" in str(tree.nodes[N].label)
        accent = theme.color("accent").lower()
        await cursor_to(pilot, tree, D)
        assert bg(tree, D)[0] == theme.color("cursor-row").lower() != accent, "cursor only, never the accent"
        await pilot.press("space")
        assert screen.members == [B, C], "refused, no slot taken"
        await pilot.press("enter")
        text = screen.query_one("#inspect-text").content.plain
        assert screen.inspecting == D and "INCOMPATIBLE" in text and "sha256 differs" in text
        await pilot.press("escape")
        assert tree.has_focus and tree.selected_key() == D
        await show_incompatible(app, pilot, False)
        assert D not in tree.index and tree.selected_key() in tree.index, "cursor on a visible fallback"
        assert screen.members == [B, C] and screen.group == 3 and screen.base == A
        await pilot.press("escape")
        app.push_screen(ResultsScreen(A))
        await settle(app, pilot)
        assert app.screen.query_one("#show-all").value is False, "a new session starts with the filter off"
    run_app(filter_root(tmp_path), check)


def test_esc_moves_to_the_explorer_before_leaving(tmp_path):
    async def check(app, pilot):
        dash = app.screen
        dash_tree = dash.query_one(Explorer)
        app.push_screen(ResultsScreen(A))
        await settle(app, pilot)
        await settle(app, pilot)
        screen, tree = app.screen, app.screen.query_one(ResultsExplorer)
        await cursor_to(pilot, tree, B)
        await pilot.press("space", "2")
        for target in ("#summary", "#analysis-table", "#show-all"):
            screen.query_one(target).focus()
            await pilot.pause()
            await pilot.press("escape")
            assert app.screen is screen and tree.has_focus, f"esc from {target}: the explorer"
            assert screen.members == [B] and screen.group == 1 and tree.selected_key() == B
        await pilot.press("ctrl+tab")
        assert app.focused is screen.query_one("#summary")
        await pilot.press("escape", "escape")
        assert app.screen is dash and dash_tree.has_focus
    run_app(results_root(tmp_path), check)


def test_summary_is_compact_and_specs_group(tmp_path):
    async def check(app, pilot):
        screen, tree = await open_results(app, pilot)
        summary_box = screen.query_one("#summary")
        assert summary_box.outer_size.height == 4, "border + header + baseline"
        for key in (B, C):
            await cursor_to(pilot, tree, key)
            await pilot.press("space")
        content = screen.query_one("#summary-text").content
        table, rule, notes = content.renderables
        assert table.columns[2].max_width <= 30 and table.columns[-1].ratio == 1, "metrics beside the names"
        assert type(rule).__name__ == "Rule" and notes.row_count == 2, "one reason + one detail line per Caution"
        assert table.rows[0].style.bgcolor.triplet.hex == theme.color("baseline-row").lower()
        assert summary_box.outer_size.height <= 2 + 1 + 3 + 1 + 4, "header, rows, rule, wrapped notes"
        console = Console(width=100, record=True)
        console.print(content)
        text = console.export_text()
        assert "space adds" not in text and "comparison metric" not in text and "Δ vs" not in text, "no instructions"
        assert "⚠ 2" in text and "clean git commit" in text
        await pilot.press("2")
        table = screen.query_one("#analysis-table")
        r5 = table.get_row_at(0)
        assert r5[0] == "R@5" and r5[3].plain == "96.71% Δ —" and r5[2].plain.endswith("+0.00"), "value + Δ —"
        notes_text = str(screen.query_one("#analysis-notes").render())
        assert "Δ —" in notes_text and "clean git commit" not in notes_text, "legend once; reasons stay in Summary"
        await pilot.press("4")
        assert [table.get_row_at(i)[0] for i in range(table.row_count)] == [
            "Prediction time", "ms/query", "Eval time", "Candidates/query"]
        await pilot.press("5")
        assert screen.group == 4 and "5 Specs" in screen.query_one("#analysis").border_subtitle
        rows_ = {table.get_row_at(i)[0]: table.get_row_at(i)[1:] for i in range(table.row_count)}
        assert list(rows_) == ["beam_size", "topk", "knn_beta", "leaf_candidates", "features"]
        assert [c.plain for c in rows_["beam_size"]] == ["10", "20", "10"]
        assert rows_["beam_size"][1].style.color.triplet.hex == theme.beam_color(20).lower(), "beam colour kept"
        assert rows_["topk"][1].style == "", "equal to the baseline: plain"
    run_app(results_root(tmp_path), check)


def test_specs_marks_values_that_differ(tmp_path):
    from xmr4el_tui.screens.results_details import analysis
    f = facts(results_root(tmp_path))
    other = cmp.Facts(**{**f[B].__dict__, "metrics": {**f[B].metrics, "features": "tfidf",
                                                      "search": {**f[B].metrics["search"], "topk": 5}}})
    _, body, _ = analysis("Specs", [f[A], other], [cmp.BASELINE, cmp.assess(f[A], other)], {})
    cells = dict(body)
    assert cells["features"][1].plain == "tfidf" and cells["features"][1].style == theme.style("spec-differs", "bold")
    assert cells["topk"][1].style == theme.style("spec-differs", "bold") and cells["knn_beta"][1].style == ""


def test_inspection_values_share_one_column(tmp_path):
    from xmr4el_tui.screens.results_details import LABEL_WIDTH, inspection
    f = facts(results_root(tmp_path))
    strings = {g: {c: 0.5 for c in ("n", "tree", "dict", "either", "routed")} for g in cmp.STRING_GROUPS}
    full = cmp.Facts(**{**f[A].__dict__, "metrics": {**f[A].metrics, "strings": strings}})
    labels = {row[0] for group in cmp.GROUPS.values() for row in group}
    lines = [line for line in inspection(full, cmp.BASELINE, "alpha · dev", True).plain.split("\n")
             if line.startswith("  ") and line[2:2 + LABEL_WIDTH].strip() in labels]
    assert any("seen, >1 label · dictionary" in line for line in lines), "longest label shown"
    col = 2 + LABEL_WIDTH
    assert all(line[col - 2:col] == "  " and line[col] != " " for line in lines), "one value column, gap kept"
