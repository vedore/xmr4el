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
        await pilot.press("ctrl+tab")
        assert app.focused is library
        await pilot.press("ctrl+tab")
        assert isinstance(app.focused, TreeTabs)
        await pilot.press("escape")
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
        await pilot.press("a")
        added = app.draft.trees[-1]
        assert added.name == "Tree 2" and app.draft.tree == added.id
        await pilot.press("1")
        assert app.draft.tree == dup.id
        await edit(pilot, screen, "hierarchy")
        await type_into(pilot, screen, "n_clusters", "16")
        tabs.focus()
        await pilot.press("right")  # another tree with unapplied changes: asks
        assert app.draft.tree == dup.id and "Discard unapplied changes to Hierarchy?" in footer(app)
        await pilot.press("u")
        assert app.draft.tree == first.id and screen.editing is None
        assert dup.config["clustering_config"]["kwargs"]["n_clusters"] == 128  # nothing leaked
        await pilot.press("delete")
        assert "Remove Tree 1?" in footer(app) and len(app.draft.trees) == 3
        await pilot.press("delete")
        assert [t.name for t in app.draft.trees] == ["Small k", "Tree 2"] and app.draft.tree == added.id
        await pilot.press("delete", "delete", "delete")
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
        assert grid.layout_ == (1, 0, 0)
        await pilot.press("right", "right", "right")
        await pilot.pause()
        box = screen.query_one("#mb-grid-box")
        assert app.draft.slot == "search" and box.scroll_offset.x > 0
        await edit(pilot, screen, "search")
        assert screen.query_one("#mb-attrs").size.width >= 20 and screen.query_one("#mb-json").size.width > 30
        top, bottom = (screen.query_one(q).outer_size.height for q in ("#mb-top", "#mb-bottom"))
        assert abs(top / (top + bottom) - 0.35) < 0.06
    run_builder(make_root(tmp_path), go, size=(80, 24))
