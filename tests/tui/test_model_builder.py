"""New Experiment, Model Builder: slot registry, field parsing and omission rules, JSON preview ownership, templates,
tree management in the draft, and the screen (grid, library, editor, Apply / Cancel / esc, trees, dataset)."""
import asyncio
import copy
import json
import shutil
from pathlib import Path

import pytest

from xmr4el_tui import components
from xmr4el_tui.components import OMIT
from xmr4el_tui.datasets import EvalInput, Selection
from xmr4el_tui.draft import ExperimentDraft

REPO = Path(__file__).parents[2]
BASE = json.loads((REPO / "configs/xmr4el_bc5cdr_config.json").read_text())
SLOT_KEYS = {"transformer": "transformer_config", "char_tfidf": "vectorizer_config", "mention_svd": "dimension_config",
             "context_tfidf": "context_vectorizer_config", "context_svd": "context_dimension_config",
             "hierarchy": "clustering_config", "matcher": "matcher_config", "search": "predict_config"}


def field(sid, name):
    s = components.slot(sid)
    return next(f for c in s.components for f in c.fields + s.extras if f.path[-1] == name)


# --- registry -----------------------------------------------------------------------------------------------------
def test_eight_slots_hybrid_only_and_pifa_is_not_a_slot():
    assert {s.id: s.key for s in components.SLOTS} == SLOT_KEYS
    assert "pifa" not in components.BY_ID and components.HYBRID == "sapbert_char_context"
    extras = {f.path[0] for s in components.SLOTS for f in s.extras}
    assert extras == {"abbrev_expansion", "context_window", "depth", "min_leaf_size", "cut_half_cluster",
                      "routing_features"}
    library = {s.id: [c.id for c in s.components] for s in components.SLOTS}
    assert library == {"transformer": ["sapbert", "biobert", "sentencetbiobert", "custom"],
                       "char_tfidf": ["tfidf"], "mention_svd": ["sklearntruncatedsvd"], "context_tfidf": ["tfidf"],
                       "context_svd": ["sklearntruncatedsvd"], "hierarchy": ["balancedkmeans"],
                       "matcher": ["jointlogisticregression"], "search": ["beam"]}


def test_installed_components_and_unsupported_ones():
    cfg = copy.deepcopy(BASE)
    assert {s.id: components.installed(cfg, s).id for s in components.SLOTS}["transformer"] == "sapbert"
    cfg["transformer_config"]["kwargs"]["model_name"] = "org/model"
    assert components.installed(cfg, components.slot("transformer")).id == "custom"  # model_name overrides type
    cfg["clustering_config"]["type"] = "kmeans"
    assert components.installed(cfg, components.slot("hierarchy")) is None
    assert "hierarchy" in components.problems(cfg) and components.summary(cfg, components.slot("hierarchy")) == ""
    assert components.problems({**BASE, "features": "tfidf"})["features"]
    assert not components.problems(BASE)


def test_parse_never_turns_bad_text_into_zero_and_keeps_types():
    n, beta, df, seed = field("hierarchy", "n_clusters"), field("search", "knn_beta"), field("char_tfidf", "max_df"), \
        field("mention_svd", "random_state")
    assert components.parse(n, "12x") == (None, "Whole number >= 1.")
    assert components.parse(n, "0")[1] and components.parse(n, "1.5")[1] and components.parse(n, "")[1] == "Required."
    assert components.parse(n, " 64 ") == (64, None)
    assert components.parse(beta, "abc")[1] and components.parse(beta, "nan")[1] and components.parse(beta, "101")[1]
    v, _ = components.parse(beta, "10")
    assert components.coerce(beta, v) == 10.0 and isinstance(components.coerce(beta, v), float)
    assert components.parse(df, "1.0") == (1.0, None) and components.parse(df, "1") == (1, None)  # share vs count
    assert components.parse(df, "1.5")[1] and components.parse(df, "0")[1]
    assert components.parse(seed, "") == (None, None)  # null = unseeded
    assert components.parse(field("transformer", "max_seq_length"), "") == (OMIT, None)  # left out = default
    assert components.check(field("char_tfidf", "ngram_range"), [4, 2]) and not components.check(
        field("char_tfidf", "ngram_range"), [2, 4])
    assert components.check(field("hierarchy", "cut_half_cluster"), "yes") == "Yes or no."


def test_assign_writes_only_what_differs_from_an_omitted_default():
    slot = components.slot("hierarchy")
    origin = components.part(BASE, slot)
    draft = copy.deepcopy(origin)
    components.assign(draft, origin, slot, field("hierarchy", "routing_features"), True)  # default, absent: stays out
    components.assign(draft, origin, slot, field("hierarchy", "refine_iter"), 20)
    assert "routing_features" not in draft and "refine_iter" not in draft["clustering_config"]["kwargs"]
    components.assign(draft, origin, slot, field("hierarchy", "routing_features"), False)
    components.assign(draft, origin, slot, field("hierarchy", "n_clusters"), 8)  # present in origin: kept explicit
    assert draft["routing_features"] is False and draft["clustering_config"]["kwargs"]["n_clusters"] == 8
    components.assign(draft, origin, slot, field("hierarchy", "depth"), OMIT)
    assert "depth" not in draft and origin["depth"] == 2  # origin untouched


def test_merged_preview_keeps_every_other_key_and_highlights_by_path():
    slot = components.slot("hierarchy")
    draft = components.part(BASE, slot)
    draft["clustering_config"]["kwargs"]["n_clusters"] = 64
    proposed = components.merged(BASE, slot, draft)
    assert BASE["clustering_config"]["kwargs"]["n_clusters"] == 128  # the source is never changed
    assert {k: v for k, v in proposed.items() if k not in slot.keys} == {
        k: v for k, v in BASE.items() if k not in slot.keys}
    lines = components.json_lines(proposed)
    assert "\n".join(line for line, _ in lines) == json.dumps(proposed, indent=2)
    owned = [line for line, key in lines if key in slot.keys]
    assert '"n_clusters": 64,' in " ".join(owned) and any('"depth": 2' in line for line in owned)
    assert not any("beam_size" in line for line in owned)


def test_template_from_the_spec_of_the_training_file_and_tfidf_refused(tmp_path):
    (tmp_path / "configs/experiments").mkdir(parents=True)
    shutil.copy(REPO / "configs/xmr4el_base_config.json", tmp_path / "configs/xmr4el_base_config.json")
    (tmp_path / "configs/model_a.json").write_text(json.dumps({**BASE, "depth": 3}))
    (tmp_path / "configs/plain.json").write_text(json.dumps({**BASE, "features": "tfidf"}))
    sel = Selection("datasets/A", "A", train="datasets/A/train.pubtator")
    spec = {"name": "a", "train_path": sel.train, "evals": []}
    (tmp_path / "configs/experiments/a.json").write_text(json.dumps({**spec, "model_config": "configs/model_a.json"}))
    cfg, source, note = components.template(tmp_path, sel)
    assert (source, cfg["depth"], note) == ("configs/model_a.json", 3, "")
    assert list(cfg)[:2] == ["features", "transformer_config"]
    (tmp_path / "configs/experiments/a.json").write_text(json.dumps({**spec, "model_config": "configs/plain.json"}))
    cfg, source, note = components.template(tmp_path, sel)
    assert source == "configs/xmr4el_base_config.json" and cfg["features"] == components.HYBRID
    assert "configs/plain.json uses features 'tfidf'" in note


# --- draft trees --------------------------------------------------------------------------------------------------
def test_tree_add_duplicate_rename_move_remove():
    d = ExperimentDraft()
    a = d.add_tree(BASE)
    b = d.duplicate(a.id)
    assert (a.name, b.name, [t.id for t in d.trees], d.tree) == ("Tree 1", "Tree 1 copy", [a.id, b.id], b.id)
    b.config["clustering_config"]["kwargs"]["n_clusters"] = 7  # no shared nested dicts
    assert a.config["clustering_config"]["kwargs"]["n_clusters"] == 128 == BASE["clustering_config"]["kwargs"][
        "n_clusters"]
    assert d.rename(b.id, " tree 1 ") == "Another tree is named 'tree 1'." and b.name == "Tree 1 copy"
    assert d.rename(b.id, "x" * 41) and d.rename(b.id, "a\tb") and d.rename(b.id, "  ") == "Name required."
    assert d.rename(b.id, "Wide beam") is None and b.name == "Wide beam" and b.id == d.trees[1].id
    assert d.move(b.id, -1) and [t.id for t in d.trees] == [b.id, a.id] and not d.move(b.id, -1)
    c = d.add_tree(BASE)
    assert c.name == "Tree 2"  # the smallest free number
    assert d.remove(c.id) and d.remove(b.id) and d.tree == a.id and not d.remove(a.id) and d.trees == [a]


# --- screen -------------------------------------------------------------------------------------------------------
textual = pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402
from xmr4el_tui import theme  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import DashboardScreen  # noqa: E402
from xmr4el_tui.screens.dataset_setup import DatasetSetupScreen  # noqa: E402
from xmr4el_tui.screens.model_builder import JsonPreview, Library, ModelBuilderScreen, preview  # noqa: E402
from xmr4el_tui.widgets.pipeline_grid import PipelineGrid, TreeTabs, neighbor  # noqa: E402

SEL = Selection("datasets/A", "NCBI Disease", "", "datasets/A/train_plus_ctd.pubtator",
                (EvalInput("dev", "datasets/A/dev.pubtator", True), EvalInput("test", None, False)))


def make_root(root):
    (root / "outputs/runs").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs/xmr4el_base_config.json").write_text(json.dumps(BASE))
    return root


def footer(app):
    note = app.screen.query_one("#footer-note")
    return str(note.render()) if note.display else ""


def run_builder(root, test, size=(140, 40)):
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await app.workers.wait_for_complete()
            app.draft.confirmed = SEL
            await app.push_screen(ModelBuilderScreen())
            await pilot.pause()
            await test(app, pilot, app.screen)
    asyncio.run(go())


async def edit(pilot, screen, sid, comp_index=None):
    """Grid cursor to `sid`, its library, enter on the installed (or `comp_index`) component."""
    screen.select_slot(sid)
    screen.query_one(PipelineGrid).focus()
    await pilot.press("enter")
    if comp_index is not None:
        screen.query_one(Library).highlighted = comp_index
    await pilot.press("enter")
    await pilot.pause()


async def type_into(pilot, screen, name, text):
    ed = screen.editing
    i = next(i for i, f in enumerate(ed.fields) if f.path[-1] == name)
    box = screen.query_one(f"#mbf-{i}", Input)
    box.focus()
    box.value = text
    await pilot.pause()
    return i


def test_grid_navigation_follows_branches():
    assert neighbor("transformer", 1, 0, "mention_svd") == "hierarchy"  # no second block on that branch
    assert neighbor("char_tfidf", 1, 0, "x") == "mention_svd" and neighbor("mention_svd", 1, 0, "x") == "hierarchy"
    assert neighbor("hierarchy", -1, 0, "context_svd") == "context_svd"  # the last feature slot visited
    assert neighbor("mention_svd", 0, -1, "x") == "transformer" and neighbor("transformer", 0, 1, "x") == "char_tfidf"
    assert neighbor("context_tfidf", 0, 1, "x") == "context_tfidf" and neighbor("matcher", 0, 1, "x") == "matcher"
    assert neighbor("search", 1, 0, "x") == "search" and neighbor("char_tfidf", -1, 0, "x") == "char_tfidf"


def test_preview_highlights_the_edited_keys():
    text, first = preview(BASE, components.slot("search").keys, 80)
    rows = text.split("\n")
    lines = components.json_lines(BASE)
    section = f"{theme.color('text')} on {theme.color('section')}"
    marked = [i for i, row in enumerate(rows) if any(str(s.style) == section for s in row.spans)]
    assert marked == [i for i, (_, key) in enumerate(lines) if key == "predict_config"] and first == marked[0]
    assert all(len(row.plain) == 80 for row in rows)


def test_keyboard_grid_library_and_mouse(tmp_path):
    async def go(app, pilot, screen):
        grid = screen.query_one(PipelineGrid)
        assert app.focused is grid and app.draft.trees[0].name == "Tree 1"
        assert screen.query_one("#mb-top").border_subtitle == \
            "NCBI Disease · Training: train_plus_ctd · Evaluation: dev"
        await pilot.press("down", "right", "right")
        assert app.draft.slot == "hierarchy" and app.draft.feature_slot == "mention_svd"
        await pilot.press("left")
        assert app.draft.slot == "mention_svd"
        await pilot.press("enter")
        library = screen.query_one(Library)
        assert app.focused is library and [o.id for o in library.options] == ["sklearntruncatedsvd"]
        assert "installed" in str(library.options[0].prompt)
        await pilot.press("escape")
        assert app.focused is grid
        region = grid.regions["matcher"]
        await pilot.click(PipelineGrid, offset=(region.x + 2, region.y))
        assert app.draft.slot == "matcher" and [o.id for o in library.options] == ["jointlogisticregression"]
        await pilot.press("enter")
        assert app.focused is library
        await pilot.press("escape")
        assert app.focused is grid
        await pilot.press("escape")  # grid -> Tree Selection, not the previous screen
        assert isinstance(app.focused, TreeTabs) and app.screen is screen
        await pilot.press("escape", "escape")  # the first esc only asks
        assert isinstance(app.screen, DashboardScreen)
    run_builder(make_root(tmp_path), go)


def test_edit_preview_apply_keeps_types_and_other_fields(tmp_path):
    async def go(app, pilot, screen):
        tree = app.draft.trees[0]
        before = copy.deepcopy(tree.config)
        await edit(pilot, screen, "hierarchy")
        assert screen.editing.comp.id == "balancedkmeans" and app.focused.id == "mbf-0"
        assert screen.query_one("#mbf-0", Input).value == "128"  # the applied value
        await type_into(pilot, screen, "n_clusters", "64")
        await type_into(pilot, screen, "min_leaf_size", "3")
        text = str(screen.query_one("#mb-json-text").render())
        assert '"n_clusters": 64' in text and tree.config == before  # preview only
        assert screen.query_one(JsonPreview).border_subtitle == "unapplied changes"
        await pilot.press("ctrl+s")
        assert screen.editing is None and app.focused is screen.query_one(PipelineGrid)
        assert app.draft.slot == "hierarchy" and "Hierarchy: Balanced KMeans applied to Tree 1" in footer(app)
        got = tree.config
        assert got["clustering_config"]["kwargs"] == {"n_clusters": 64, "iter_limit": 20}
        assert type(got["min_leaf_size"]) is int and got["min_leaf_size"] == 3 and got["cut_half_cluster"] is True
        assert {k: v for k, v in got.items() if k not in ("clustering_config", "min_leaf_size")} == {
            k: v for k, v in before.items() if k not in ("clustering_config", "min_leaf_size")}
        await edit(pilot, screen, "search")
        await type_into(pilot, screen, "knn_beta", "5")
        await pilot.press("ctrl+s")
        assert tree.config["predict_config"]["knn_beta"] == 5.0 and isinstance(
            tree.config["predict_config"]["knn_beta"], float)
    run_builder(make_root(tmp_path), go)


def test_invalid_apply_commits_nothing_and_cancel_restores(tmp_path):
    async def go(app, pilot, screen):
        tree = app.draft.trees[0]
        before = copy.deepcopy(tree.config)
        await edit(pilot, screen, "hierarchy")
        await type_into(pilot, screen, "iter_limit", "30")
        i = await type_into(pilot, screen, "n_clusters", "many")
        assert screen.query_one(f"#mbe-{i}").display and "invalid field" in screen.query_one(JsonPreview).border_subtitle
        assert '"n_clusters": 128' in str(screen.query_one("#mb-json-text").render())  # last valid value
        await pilot.press("ctrl+s")
        assert screen.editing is not None and tree.config == before
        assert "Not applied: Clusters per node: Whole number >= 1." in footer(app) and app.focused.id == f"mbf-{i}"
        await pilot.click("#mb-cancel")
        assert screen.editing is None and tree.config == before and app.focused is screen.query_one(Library)
    run_builder(make_root(tmp_path), go)


def test_document_frequency_type_change_requires_apply_or_discard(tmp_path):
    async def go(app, pilot, screen):
        await edit(pilot, screen, "char_tfidf")
        original = screen.tree.config["vectorizer_config"]["kwargs"]["max_df"]
        assert type(original) is float and original == 1.0
        await type_into(pilot, screen, "max_df", "1")
        assert screen.dirty(), "1 document and 100% of documents are different scientific settings"
        await pilot.press("escape")
        assert "Discard unapplied" in footer(app)
        await pilot.press("u")
        assert type(screen.tree.config["vectorizer_config"]["kwargs"]["max_df"]) is float
    run_builder(make_root(tmp_path), go)


def test_esc_with_unapplied_changes_asks(tmp_path):
    async def go(app, pilot, screen):
        tree = app.draft.trees[0]
        before = copy.deepcopy(tree.config)
        await edit(pilot, screen, "matcher")
        await pilot.press("escape")  # clean: straight back to the library
        assert screen.editing is None and app.focused is screen.query_one(Library)
        await pilot.press("enter")
        await type_into(pilot, screen, "C", "2")
        field_ = app.focused
        await pilot.press("escape")
        assert "Discard unapplied changes to Matcher?" in footer(app) and screen.editing is not None
        await pilot.press("escape")  # keep editing: focus back in the field
        assert app.focused is field_ and screen.editing.draft["matcher_config"]["kwargs"]["C"] == 2.0
        await pilot.press("escape", "u")
        assert screen.editing is None and tree.config == before and app.focused is screen.query_one(Library)
        await pilot.press("escape")
        await pilot.press("right")  # matcher -> search: nothing to protect
        assert app.draft.slot == "search"
    run_builder(make_root(tmp_path), go)


def test_alternative_component_starts_from_defaults_until_applied(tmp_path):
    async def go(app, pilot, screen):
        tree = app.draft.trees[0]
        await edit(pilot, screen, "transformer", comp_index=1)  # BioBERT
        assert screen.editing.draft["transformer_config"] == {"type": "biobert", "kwargs": {}}
        assert tree.config["transformer_config"]["type"] == "sapbert"
        await pilot.press("ctrl+s")
        assert tree.config["transformer_config"] == {"type": "biobert", "kwargs": {}}
        await edit(pilot, screen, "transformer", comp_index=3)  # custom: the model id is required
        await pilot.press("ctrl+s")
        assert "Not applied: Model: Required." in footer(app) and tree.config["transformer_config"]["type"] == "biobert"
        await type_into(pilot, screen, "model_name", "org/encoder")
        await pilot.press("ctrl+s")
        assert tree.config["transformer_config"] == {"kwargs": {"model_name": "org/encoder"}}
        assert components.component_label(tree.config, components.slot("transformer")) == "encoder"
    run_builder(make_root(tmp_path), go)


def test_tree_management_keys_and_draft_protection(tmp_path):
    async def go(app, pilot, screen):
        tabs = screen.query_one(TreeTabs)
        first = app.draft.trees[0]
        tabs.focus()
        await pilot.press("d")  # duplicate: applied configuration only
        dup = app.draft.trees[1]
        assert (dup.name, app.draft.tree) == ("Tree 1 copy", dup.id) and dup.config == first.config
        await pilot.press("f2")
        rename = screen.query_one("#mb-rename", Input)
        assert app.focused is rename and rename.value == "Tree 1 copy"
        rename.value = "Tree 1"
        await pilot.press("enter")
        assert "Not renamed: Another tree is named 'Tree 1'." in footer(app) and dup.name == "Tree 1 copy"
        rename.value = "Small k"
        await pilot.press("enter")
        assert dup.name == "Small k" and app.focused is tabs and not rename.display
        await pilot.press("shift+left")
        assert [t.id for t in app.draft.trees] == [dup.id, first.id] and app.draft.tree == dup.id
        await pilot.press("n")
        added = app.draft.trees[-1]
        assert added.name == "Tree 2" and app.draft.tree == added.id
        await pilot.press("1")
        assert app.draft.tree == dup.id
        await edit(pilot, screen, "hierarchy")
        await type_into(pilot, screen, "n_clusters", "16")
        tabs.focus()
        await pilot.press("right")  # moves the highlight only: nothing asked, nothing switched
        assert app.draft.tree == dup.id and screen.editing is not None and "Discard" not in footer(app)
        await pilot.press("enter")  # another tree with unapplied changes: asks
        assert app.draft.tree == dup.id and "Discard unapplied changes to Hierarchy?" in footer(app)
        await pilot.press("u")
        assert app.draft.tree == first.id and screen.editing is None
        assert dup.config["clustering_config"]["kwargs"]["n_clusters"] == 128  # nothing leaked
        assert app.focused is screen.query_one(PipelineGrid)
        await pilot.press("delete")  # not in Tree Selection: nothing
        assert len(app.draft.trees) == 3 and "Remove" not in footer(app)
        await pilot.press("escape", "delete")
        assert "Remove Tree 1?" in footer(app) and len(app.draft.trees) == 3
        await pilot.press("y")
        assert [t.name for t in app.draft.trees] == ["Small k", "Tree 2"] and app.draft.tree == added.id
        await pilot.press("delete", "y", "delete", "y")
        assert len(app.draft.trees) == 1 and "A batch needs at least one tree." in footer(app)
    run_builder(make_root(tmp_path), go)


def test_navigation_away_asks_and_dataset_setup_round_trip(tmp_path):
    root = make_root(tmp_path)
    for f in ("train_plus_ctd.pubtator", "dev.pubtator"):
        path = root / "datasets/A" / f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("11|t|Heart attack\n11|a|Abstract.\n11\t0\t5\tHeart\tDisease\tD006331\n\n")

    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=(120, 36)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.press("m")  # nothing confirmed yet
            assert isinstance(app.screen, DatasetSetupScreen) and "Confirm a dataset first" in footer(app)
            await pilot.press("enter", "ctrl+s", "m")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelBuilderScreen) and screen.crumbs == ("Model Builder",)
            assert screen.query_one("#mb-top").border_subtitle == "A · Training: train_plus_ctd · Evaluation: dev"
            confirmed = app.draft.confirmed
            await edit(pilot, screen, "matcher")
            await type_into(pilot, screen, "max_iter", "50")
            dataset_setup = app.screen_stack[-2]
            app.go_to(dataset_setup, 0)  # breadcrumb click
            await pilot.pause()
            assert app.screen is screen and "Discard unapplied changes to Matcher?" in footer(app)
            await pilot.press("u")
            await pilot.pause()
            assert app.screen is dataset_setup and app.draft.confirmed == confirmed  # dataset untouched
            assert app.draft.trees[0].config["matcher_config"] == BASE["matcher_config"]
            app.draft.confirmed = Selection(confirmed.key, "Renamed", "", confirmed.train, confirmed.evals[:1])
            await pilot.press("m")
            await pilot.pause()
            assert app.screen.query_one("#mb-top").border_subtitle.startswith("Renamed · ")
            assert len(app.draft.trees) == 1  # trees kept across the round trip
    asyncio.run(go())


def test_short_terminal_scrolls_the_grid(tmp_path):
    async def go(app, pilot, screen):
        grid = screen.query_one(PipelineGrid)
        assert grid.layout_[0] == 1 and grid.level == "normal"
        await pilot.press("right", "right", "right")
        await pilot.pause()
        box = screen.query_one("#mb-grid-box")
        assert app.draft.slot == "search" and box.scroll_offset.x > 0
        await edit(pilot, screen, "search")
        assert screen.query_one("#mb-attrs").size.width >= 20 and screen.query_one("#mb-json").size.width > 30
        top, bottom = (screen.query_one(q).outer_size.height for q in ("#mb-top", "#mb-bottom"))
        assert abs(top / (top + bottom) - 0.45) < 0.06
    run_builder(make_root(tmp_path), go, size=(80, 24))


# --- UX refinement: layout, focus, arrows, tree actions, help, zoom ---------------------------------------------------
def test_every_exposed_field_has_help_a_json_path_and_constraints():
    for s in components.SLOTS:
        for c in s.components:
            for f in c.fields + s.extras:
                assert len(f.help) >= 20, (s.id, f.label)
                path = components.json_path(s, f)
                assert path == ".".join(components.path_of(s, f)) and path.split(".")[0] in s.keys
                assert components.constraints(f).endswith(".")
    assert components.json_path(components.slot("hierarchy"), field("hierarchy", "n_clusters")) == \
        "clustering_config.kwargs.n_clusters"
    assert components.json_path(components.slot("search"), field("search", "knn_beta")) == "predict_config.knn_beta"
    assert components.json_path(components.slot("hierarchy"), field("hierarchy", "depth")) == "depth"


def test_layout_45_55_with_40_60_editor_split(tmp_path):
    async def go(app, pilot, screen):
        top, bottom = (screen.query_one(q).outer_size.height for q in ("#mb-top", "#mb-bottom"))
        assert abs(top / (top + bottom) - 0.45) < 0.04
        await edit(pilot, screen, "hierarchy")
        attrs, js = (screen.query_one(q).outer_size.width for q in ("#mb-attrs-panel", "#mb-json"))
        assert abs(attrs / (attrs + js) - 0.40) < 0.03
        assert screen.query_one("#mb-attrs").size.height >= 3 and screen.query_one("#mb-help").size.height >= 3
    run_builder(make_root(tmp_path), go)


def test_tab_stays_in_the_panel_and_the_editor_is_reachable_without_ctrl_tab(tmp_path):
    async def go(app, pilot, screen):
        grid, tabs = screen.query_one(PipelineGrid), screen.query_one(TreeTabs)
        await pilot.press("tab")
        assert app.focused is screen.query_one("#mb-review")  # the Review Batch button ends the panel's stops
        await pilot.press("tab")
        assert app.focused is tabs
        await pilot.press("tab")
        assert app.focused is grid  # wraps inside the Model Builder panel, never leaves it
        await pilot.press("shift+tab")
        assert app.focused is tabs
        grid.focus()
        await pilot.press("ctrl+tab", "f6")  # no longer bound
        assert app.focused is grid
        await edit(pilot, screen, "hierarchy")
        n = len(screen.editing.fields)
        for _ in range(n + 6):  # fields, help, Apply, Cancel, Preview, then around
            await pilot.press("tab")
            assert screen.panel_of(app.focused) == "attrs"
        await pilot.press("shift+tab")
        assert screen.panel_of(app.focused) == "attrs"
        screen.query_one("#mb-preview").focus()
        await pilot.press("enter")  # Preview: the JSON panel by keyboard
        assert app.focused is screen.query_one(JsonPreview)
        await pilot.press("tab")
        assert app.focused is screen.query_one(JsonPreview)
        await pilot.press("escape")  # back to the attributes, the editor stays open
        assert screen.panel_of(app.focused) == "attrs" and screen.editing is not None
        for binding in screen.BINDINGS:
            assert "ctrl+tab" not in binding.key and "f6" not in binding.key
    run_builder(make_root(tmp_path), go)


def test_tree_selection_levels_numbers_and_removal_only_there(tmp_path):
    async def go(app, pilot, screen):
        tabs, grid = screen.query_one(TreeTabs), screen.query_one(PipelineGrid)
        first = app.draft.trees[0]
        app.draft.add_tree(first.config)
        app.draft.add_tree(first.config)
        screen.fill()
        t1, t2, t3 = app.draft.trees
        app.draft.tree = t1.id
        screen.fill()
        text = str(tabs.render())
        assert "[1] Tree 1" in text and "[2] Tree 2" in text and "[3] Tree 3" in text
        await pilot.press("x", "delete")  # grid focused: removal keys are inactive
        assert len(app.draft.trees) == 3 and not footer(app)
        await pilot.press("escape")
        assert app.focused is tabs and tabs.cursor == t1.id
        await pilot.press("right", "right")
        assert tabs.cursor == t3.id and app.draft.tree == t1.id  # highlight only
        await pilot.press("right", "right")  # Add tree is the last stop; nothing beyond it
        assert tabs.cursor == "add"
        await pilot.press("left")
        assert tabs.cursor == t3.id
        await pilot.press("enter")
        assert app.draft.tree == t3.id and app.focused is grid
        await pilot.press("escape", "1")  # jump keeps Tree Selection
        assert app.draft.tree == t1.id and app.focused is tabs
        await pilot.press("shift+right")  # reorder the highlighted tree; numbers follow the positions
        assert [t.id for t in app.draft.trees] == [t2.id, t1.id, t3.id]
        assert str(tabs.render()).index("[2] Tree 1") < str(tabs.render()).index("[3] Tree 3")
        await pilot.press("3")
        assert app.draft.tree == t3.id
        await pilot.press("left", "x")  # highlighted: t1 is under the cursor? the cursor follows the jump
        assert "Remove" in footer(app) and len(app.draft.trees) == 3
        await pilot.press("y")
        assert len(app.draft.trees) == 2 and app.focused is tabs and tabs.cursor in {t.id for t in app.draft.trees}
        assert "[2]" in str(tabs.render()) and "[3]" not in str(tabs.render())
        await pilot.press("escape", "escape")  # still Tree Selection -> previous screen (asks first)
        assert isinstance(app.screen, DashboardScreen)
    run_builder(make_root(tmp_path), go)


def test_digits_type_into_inputs(tmp_path):

    async def go(app, pilot, screen):
        app.draft.add_tree(app.draft.trees[0].config)
        screen.fill()
        await edit(pilot, screen, "hierarchy")
        await type_into(pilot, screen, "n_clusters", "")
        await pilot.press("2", "5")
        assert screen.editing.draft["clustering_config"]["kwargs"]["n_clusters"] == 25
        assert app.draft.tree == app.draft.trees[1].id  # unchanged: digits went to the input
    run_builder(make_root(tmp_path), go)


def test_arrows_move_between_parameters_without_applying(tmp_path):
    async def go(app, pilot, screen):
        await edit(pilot, screen, "hierarchy")
        tree = app.draft.trees[0]
        before = copy.deepcopy(tree.config)
        names = [f.label for f in screen.editing.fields]
        assert app.focused is screen.query_one("#mbf-0")
        await type_into(pilot, screen, "n_clusters", "16")
        await pilot.press("down")
        assert app.focused is screen.query_one("#mbf-1")
        assert "Iteration limit" in str(screen.query_one("#mb-help-text").render())
        await pilot.press("up", "up")
        assert app.focused is screen.query_one("#mbf-0")
        text = str(screen.query_one("#mb-help-text").render())
        assert names[0] in text and "JSON path: clustering_config.kwargs.n_clusters" in text
        assert screen.editing.draft["clustering_config"]["kwargs"]["n_clusters"] == 16  # draft kept
        assert tree.config == before                                                   # nothing applied
        await pilot.press("down", "down", "down", "down")  # onto the Tree depth input
        assert screen.editing.fields[app.focused and screen.field_of(app.focused.id)[0]].label == "Tree depth"
    run_builder(make_root(tmp_path), go)


def test_dropdown_keeps_its_arrows_when_expanded(tmp_path):
    from textual.widgets import Select

    async def go(app, pilot, screen):
        await edit(pilot, screen, "hierarchy")
        i = next(i for i, f in enumerate(screen.editing.fields) if f.path[-1] == "cut_half_cluster")
        select = screen.query_one(f"#mbf-{i}", Select)
        select.focus()
        await pilot.press("up")
        assert app.focused is screen.query_one(f"#mbf-{i - 1}")  # collapsed: moves to the previous parameter
        select.focus()
        await pilot.press("enter")
        assert select.expanded
        was = select.value
        await pilot.press("up", "enter")  # the open dropdown's own arrows pick the other option
        assert select.value == 1 - was and not select.expanded
        assert screen.editing.draft["cut_half_cluster"] is (select.value == 0)
        select.focus()
        await pilot.press("enter", "escape")
        assert not select.expanded and screen.editing is not None  # esc closed the dropdown only
    run_builder(make_root(tmp_path), go)


def ZoomBar_width(screen):
    return screen.query_one("#mb-zoom").size.width


def test_zoom_levels_fit_centering_and_independence_from_the_model(tmp_path):
    from xmr4el_tui.widgets.pipeline_grid import GEOS, ZoomBar

    async def go(app, pilot, screen):
        grid, box = screen.query_one(PipelineGrid), screen.query_one("#mb-grid-box")
        config = copy.deepcopy(app.draft.trees[0].config)
        assert grid.zoom == "normal" and grid.level == "normal"
        assert grid.margin_[0] == (box.scrollable_content_region.width - GEOS["normal"].width) // 2 > 0
        await pilot.press("minus")
        assert (grid.zoom, grid.level) == ("compact", "compact") and grid.styles.width.value == GEOS["compact"].width
        await pilot.press("minus")
        assert grid.level == "compact"  # the lowest level stays
        await pilot.press("plus", "plus")
        assert grid.level == "expanded" and grid.layout_[0] >= 1
        await pilot.press("0")
        assert grid.zoom == "fit" and grid.level == "expanded"  # 220 columns hold the expanded grid
        for sid in components.BY_ID:  # every slot stays reachable at every level
            assert sid in grid.regions
        app.draft.slot = "hierarchy"
        await pilot.click(ZoomBar, offset=(1, 0))  # mouse: zoom out
        assert (grid.zoom, grid.level) == ("normal", "normal")
        await pilot.click(ZoomBar, offset=(ZoomBar_width(screen) - 2, 0))  # mouse: Fit
        assert grid.zoom == "fit"
        assert app.draft.trees[0].config == config and screen.editing is None
        assert app.draft.slot == "hierarchy"
        # zoom survives a tree switch
        screen.query_one(TreeTabs).focus()
        await pilot.press("n")
        assert len(app.draft.trees) == 2 and grid.zoom == "fit" == screen.query_one(ZoomBar).zoom
    run_builder(make_root(tmp_path), go, size=(220, 50))


def test_fit_adapts_to_the_width_and_every_level_keeps_all_slots(tmp_path):
    from xmr4el_tui.widgets.pipeline_grid import GEOS

    async def go(app, pilot, screen):
        grid = screen.query_one(PipelineGrid)
        grid.set_zoom("fit")
        await pilot.pause()
        assert grid.level == "compact"  # 100 columns: only the compact grid is readable, the rest scrolls
        assert grid.margin_[0] == 0 and GEOS["compact"].width > screen.query_one("#mb-grid-box").size.width
        for level in ("compact", "normal", "expanded"):
            grid.set_zoom(level)
            await pilot.pause()
            assert set(grid.regions) == set(SLOT_KEYS) and all(r.width >= 1 and r.height >= 1 for r in grid.regions.values())
            grid.cursor = "search"
            grid.reveal()
    run_builder(make_root(tmp_path), go, size=(100, 36))


def test_enter_on_add_tree_creates_a_tree_and_opens_its_builder(tmp_path):
    async def go(app, pilot, screen):
        tabs, grid = screen.query_one(TreeTabs), screen.query_one(PipelineGrid)
        await pilot.press("escape", "right")
        assert tabs.cursor == "add" and "Add tree" in str(tabs.render())
        await pilot.press("x")  # nothing highlighted to remove
        assert len(app.draft.trees) == 1 and "Highlight a tree first." in footer(app)
        await pilot.press("enter")
        assert len(app.draft.trees) == 2 and app.draft.tree == app.draft.trees[1].id and app.focused is grid
        assert app.draft.trees[1].name == "Tree 2"
    run_builder(make_root(tmp_path), go)


def test_footer_confirmation_is_modal(tmp_path):
    async def go(app, pilot, screen):
        tabs = screen.query_one(TreeTabs)
        app.draft.add_tree(app.draft.trees[0].config)
        screen.fill()
        await pilot.press("escape", "x")
        question, cursor = app.asking, tabs.cursor
        assert question and "Remove" in footer(app)
        await pilot.press("left", "enter", "tab", "1", "a", "right")
        from textual import events  # the driver's events reach App.on_event; pilot.click bypasses it
        mouse = dict(widget=None, x=1, y=1, delta_x=0, delta_y=0, button=1, shift=False, meta=False, ctrl=False)
        assert not app.confirm_allows(events.MouseDown(**mouse)) and not app.confirm_allows(events.MouseUp(**mouse))
        assert not app.confirm_allows(events.MouseScrollDown(**mouse)) and app.confirm_allows(events.MouseMove(**mouse))
        await app.on_event(events.MouseDown(**mouse))
        await app.on_event(events.MouseUp(**mouse))
        assert app.asking == question and tabs.cursor == cursor and app.focused is tabs and len(app.draft.trees) == 2
        assert "Remove" in footer(app)
        await pilot.press("y")  # its own key answers it
        assert app.asking is None and len(app.draft.trees) == 1
        await pilot.press("ctrl+c")  # the quit question is modal too
        assert app.confirming is screen and app.asking is None
        await pilot.press("right", "enter", "a")
        assert app.confirming is screen
        await pilot.press("escape")
        assert app.confirming is None and isinstance(app.screen, ModelBuilderScreen)
    run_builder(make_root(tmp_path), go)


def test_every_tab_stop_in_the_editor_is_visibly_focused(tmp_path):
    async def go(app, pilot, screen):
        await edit(pilot, screen, "transformer")
        stops = []
        for _ in range(len(screen.members("attrs"))):
            await pilot.press("tab")
            stops.append(app.focused)
        help_box = screen.query_one("#mb-help")
        assert help_box in stops and help_box.border_title  # the help panel is a stop, and says so when focused
    run_builder(make_root(tmp_path), go)


# --- usability review repairs --------------------------------------------------------------------------------------
def test_resize_keeps_the_focused_field_visible_and_arrows_move_fields(tmp_path):
    async def go(app, pilot, screen):
        await edit(pilot, screen, "transformer")
        await pilot.press("down", "down", "down")
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        box, here = screen.query_one("#mb-attrs"), app.focused
        view = box.region.shrink(box.styles.gutter)
        assert here.region.y >= view.y and here.region.bottom <= view.bottom, "focused field inside its viewport"
        i = int(here.id.split("-")[1])
        await pilot.press("up")  # an Input's arrow moves to the previous parameter, scrolled or not
        assert app.focused.id != here.id and int(app.focused.id.split("-")[1]) == i - 1
    run_builder(make_root(tmp_path), go)


def test_long_confirmation_keeps_both_answers_at_80_columns(tmp_path):
    async def go(app, pilot, screen):
        app.ask("Remove " + "very long tree name " * 6 + "?", "y", "remove")
        await pilot.pause()
        text = footer(app)
        assert "y remove" in text and "esc cancel" in text and "…" in text
    run_builder(make_root(tmp_path), go, size=(80, 24))


def test_tree_actions_are_described_in_help_and_the_mode_is_named(tmp_path):
    for b in TreeTabs.BINDINGS:
        assert b.description, b.key

    async def go(app, pilot, screen):
        screen.query_one(PipelineGrid).focus()
        await pilot.press("escape")
        assert isinstance(app.focused, TreeTabs)
        assert screen.query_one("#mb-top").border_title == "Model Builder · Tree selection"
    run_builder(make_root(tmp_path), go)
