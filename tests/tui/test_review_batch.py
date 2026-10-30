"""New Experiment, Review Batch: configuration differences, validation issues, the draft's batch name / reference /
plan, and the screen (queue, reorder, Overview / Differences, Fix Issue, restoring the review context)."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from xmr4el_tui import components, review
from xmr4el_tui.datasets import EvalInput, Selection
from xmr4el_tui.screens import review_details
from xmr4el_tui.draft import ExperimentDraft

REPO = Path(__file__).parents[2]
BASE = json.loads((REPO / "configs/xmr4el_bc5cdr_config.json").read_text())
PUB = "11|t|Heart attack risk\n11|a|Some abstract.\n11\t0\t5\tHeart\tDisease\tD006331\n\n"


def make_root(root):
    (root / "outputs/runs").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs/xmr4el_base_config.json").write_text(json.dumps(BASE))
    for f in ("train.txt", "dev.pubtator", "test.pubtator"):  # a PubTator file may be .txt
        (root / "datasets/A").mkdir(parents=True, exist_ok=True)
        (root / "datasets/A" / f).write_text(PUB)
    return root


SEL = Selection("datasets/A", "NCBI Disease", "", "datasets/A/train.txt",
                (EvalInput("dev", "datasets/A/dev.pubtator", True), EvalInput("test", "datasets/A/test.pubtator", False)))


def drafted(*configs):
    d = ExperimentDraft()
    d.confirmed = SEL
    for c in configs or (BASE,):
        d.add_tree(c)
    return d


def tweaked(path, value, base=BASE):
    cfg = copy.deepcopy(base)
    node = cfg
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return cfg


def messages(issues):
    return [i.message for i in issues]


# --- differences --------------------------------------------------------------------------------------------------
def test_identical_and_reordered_configs_have_no_differences():
    shuffled = dict(reversed(list(BASE.items())))
    assert review.diff(BASE, copy.deepcopy(BASE)) == [] and review.diff(BASE, shuffled) == []


@pytest.mark.parametrize("key", ["vectorizer_config", "context_vectorizer_config"])
@pytest.mark.parametrize("threshold", ["min_df", "max_df"])
def test_document_frequency_count_and_fraction_are_different(key, threshold):
    path = (key, "kwargs", threshold)
    count, fraction = tweaked(path, 1), tweaked(path, 1.0)
    changes = review.diff(count, fraction)
    assert [c.path for c in changes] == [path]
    assert not any(i.severity == review.CAUTION for issues in review.tree_issues(drafted(count, fraction)).values()
                   for i in issues)


def test_numbers_nested_keys_lists_and_stage_grouping():
    other = tweaked(("predict_config", "beam_size"), 20, tweaked(("clustering_config", "kwargs", "n_clusters"), 64,
                    tweaked(("depth",), 3, tweaked(("vectorizer_config", "kwargs", "ngram_range"), [2, 5]))))
    changes = review.diff(BASE, other)
    by = {c.path: c for c in changes}
    assert by[("depth",)].stage == "Hierarchy" and (by[("depth",)].old, by[("depth",)].new) == (BASE["depth"], 3)
    assert by[("clustering_config", "kwargs", "n_clusters")].label == "Clusters per node"
    assert by[("predict_config", "beam_size")].stage == "Search"
    assert by[("vectorizer_config", "kwargs", "ngram_range")].stage == "Features"
    assert [c.stage for c in changes] == sorted((c.stage for c in changes), key=review.STAGES.index)
    assert review.diff(tweaked(("depth",), 2), tweaked(("depth",), 2.0)) == []  # 2 and 2.0 are the same number
    assert review.diff(tweaked(("cut_half_cluster",), False), tweaked(("cut_half_cluster",), 0))  # bool is not 0


def test_runtime_defaults_missing_keys_and_explicit_nulls():
    omitted = {k: v for k, v in BASE.items() if k != "min_leaf_size"}
    explicit = {**BASE, "min_leaf_size": 20}  # the backend default written out
    assert review.diff(omitted, explicit) == []
    assert review.diff(omitted, {**omitted, "min_leaf_size": 21})[0].old == 20  # the default shows as the old value
    a, b = {**BASE, "extra": None}, {**BASE}
    (change,) = review.diff(a, b)
    assert (change.old, change.new, change.label) == (None, review.MISSING, "extra")  # null is not missing
    assert review.show(review.MISSING) == "not set" and review.show(None) == "null"
    assert review.diff(tweaked(("context_window",), None), tweaked(("context_window",), 10))[0].old is None


def test_different_transformers_are_one_component_change():
    other = tweaked(("transformer_config", "type"), "biobert")
    (change,) = review.diff(BASE, other)
    assert change.label == "Transformer component" and (change.old, change.new) == ("sapbert", "biobert")


def test_duplicate_configurations_are_a_caution_not_an_error():
    d = drafted(BASE, BASE)
    found = review.tree_issues(d)
    assert [i.severity for i in found[d.trees[0].id]] == [review.CAUTION]
    assert "Tree 2" in found[d.trees[0].id][0].message and not review.blocking(found[d.trees[1].id])
    d.apply(d.trees[1].id, tweaked(("depth",), 3))
    assert review.tree_issues(d) == {t.id: [] for t in d.trees}


# --- draft: batch name, reference, plan --------------------------------------------------------------------------------
def test_batch_name_is_display_only_and_validated():
    d = drafted()
    ident = d.id
    assert d.rename_batch("NCBI Hierarchy Comparison") is None and d.name == "NCBI Hierarchy Comparison"
    assert d.rename_batch("   ") == "Name required." and d.rename_batch("x" * 61) and d.rename_batch("a\tb")
    assert d.name == "NCBI Hierarchy Comparison" and d.id == ident and ident.isalnum()


def test_reference_is_the_first_created_tree_by_id():
    d = drafted(BASE, BASE, BASE)
    a, b, c = d.trees
    assert d.reference is a
    d.move(a.id, 1)
    d.rename(a.id, "Renamed")
    assert d.reference is a and d.trees[1] is a  # queue position and name do not matter
    d.remove(a.id)
    assert d.reference is b  # deterministic: the oldest survivor
    d.add_tree(BASE)
    assert d.reference is b


def test_plan_copies_the_draft_and_makes_runner_specs():
    d = drafted(BASE, tweaked(("depth",), 3))
    d.rename_batch("My batch")
    d.move(d.trees[0].id, 1)
    p = review.plan(d)
    assert [t.id for t in p.trees] == [t.id for t in d.trees] and p.name == "My batch" and p.dataset is SEL
    d.apply(d.trees[0].id, BASE)
    assert p.trees[0].config["depth"] == 3  # the plan is a copy: later edits never reach it
    spec = review.run_spec(p, p.trees[0], "cfg.json")
    assert spec == {"name": f"{d.id}_{p.trees[0].id}", "model_config": "cfg.json", "train_path": SEL.train,
                    "evals": [{"split": "dev", "test_path": "datasets/A/dev.pubtator", "train_path": SEL.train}]}
    assert review.RUN_NAME.fullmatch(spec["name"])
    with pytest.raises(ValueError):
        review.plan(ExperimentDraft())


# --- validation ---------------------------------------------------------------------------------------------------
def test_valid_batch_has_no_issues(tmp_path):
    root = make_root(tmp_path)
    assert review.validate(root, drafted(BASE, tweaked(("depth",), 3))) == []  # .txt training file passes


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_malformed_component_kwargs_block_preflight(value):
    for slot in components.SLOTS:
        if slot.id == "search":
            continue
        cfg = tweaked((slot.key, "kwargs"), value)
        draft = drafted(cfg)
        issues = review.tree_issues(draft)[draft.trees[0].id]
        assert any(i.slot == slot.id and i.severity == review.ERROR and "kwargs" in i.message for i in issues)
        assert draft.trees[0].config[slot.key]["kwargs"] == value


@pytest.mark.parametrize("sel, text", [
    (None, "No dataset confirmed."),
    (Selection("k", "n", train="datasets/A/gone.txt", evals=SEL.evals), "Training file not found"),
    (Selection("k", "n", train=SEL.train, evals=(EvalInput("dev", "datasets/A/gone", True),)), "Development file not found"),
    (Selection("k", "n", train=SEL.train, evals=(EvalInput("dev", "datasets/A/dev.pubtator", False),)), "Enable at least one"),
    (Selection("k", "n", train=SEL.train, evals=(EvalInput("dev", "datasets/A/dev.pubtator", True),
                                                  EvalInput("dev", "datasets/A/dev.pubtator", True))), "Duplicate evaluation split"),
    (Selection("k", "n", train=SEL.train, evals=(EvalInput("dev", SEL.train, True),)), "is the training file"),
    (Selection("k", "n", train=SEL.train, evals=SEL.evals, format="csv"), "Format csv is not supported"),
    (Selection("k", "n", train=SEL.train, evals=(EvalInput("a b", "datasets/A/dev.pubtator", True),)), "split name"),
])
def test_dataset_issues_are_shared_blocking_and_shown_once(tmp_path, sel, text):
    root = make_root(tmp_path)
    d = drafted(BASE, BASE)
    d.confirmed = sel
    found = review.dataset_issues(root, sel)
    assert any(text in m for m in messages(found)) and all(i.severity == review.ERROR and i.scope == "dataset"
                                                           for i in found)
    allissues = review.validate(root, d)
    assert [i for i in allissues if i.scope == "dataset"] == found  # once, not per tree
    assert not any(i.scope == "dataset" for t in review.tree_issues(d).values() for i in t)


def test_malformed_input_is_caught_by_the_bounded_inspection(tmp_path):
    root = make_root(tmp_path)
    (root / "datasets/A/train.txt").write_text("not a pubtator file\n")
    assert "is not PubTator" in messages(review.dataset_issues(root, SEL))[0]


def test_tree_issues_carry_path_slot_field_and_destination():
    bad = tweaked(("clustering_config", "kwargs", "n_clusters"), "abc",
                  tweaked(("transformer_config", "kwargs", "batch_size"), -1,
                          tweaked(("matcher_config", "kwargs", "C"), 0,
                                  tweaked(("predict_config", "beam_size"), 0))))
    d = drafted(bad)
    found = review.tree_issues(d)[d.trees[0].id]
    by = {i.path: i for i in found}
    assert set(by) == {"clustering_config.kwargs.n_clusters", "transformer_config.kwargs.batch_size",
                       "matcher_config.kwargs.C", "predict_config.beam_size"}
    n = by["clustering_config.kwargs.n_clusters"]
    assert (n.slot, n.field.label, n.dest, n.severity, n.tree) == ("hierarchy", "Clusters per node", "slot", "error",
                                                                   d.trees[0].id)
    assert n.message.startswith("Clusters per node: Whole number") and by["predict_config.beam_size"].slot == "search"
    assert len({i.id for i in found}) == len(found)  # stable, distinct identities
    assert review.status(found) == "Invalid" and review.status([]) == "Ready"


def test_unmapped_and_naming_issues_fall_back():
    cfg = {**copy.deepcopy(BASE), "clustering_config": {"type": "kmeans"}, "mystery": 1, "features": "tfidf"}
    d = drafted(cfg)
    d.trees[0].name = "a\tb"  # an invalid name (only reachable by editing the draft directly)
    found = {i.id.split(":", 2)[2]: i for i in review.tree_issues(d)[d.trees[0].id]}
    assert found["name"].dest == "rename" and found["key:mystery"].dest == "tree"
    assert found["hierarchy:slot"].dest == "slot" and found["hierarchy:slot"].field is None
    assert found["features"].dest == "tree" and found["features"].slot is None


def test_caution_does_not_block_and_revalidation_reflects_applied_edits(tmp_path):
    root = make_root(tmp_path)
    d = drafted(BASE, BASE)
    assert review.blocking(review.validate(root, d)) == [] and review.status(review.validate(root, d)) == "Caution"
    d.apply(d.trees[1].id, tweaked(("depth",), 0))
    assert review.status(review.validate(root, d)) == "Invalid"
    d.apply(d.trees[1].id, tweaked(("depth",), 4))
    assert review.validate(root, d) == []


# --- screen -------------------------------------------------------------------------------------------------------
textual = pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dataset_setup import DatasetSetupScreen  # noqa: E402
from xmr4el_tui.screens.model_builder import Library, ModelBuilderScreen, preview  # noqa: E402
from xmr4el_tui.screens.review_batch import VIEWS, DetailsList, Queue, ReviewBatchScreen, ViewTabs  # noqa: E402
from xmr4el_tui.widgets.pipeline_grid import PipelineGrid, TreeTabs  # noqa: E402


def footer(app):
    note = app.screen.query_one("#footer-note")
    return str(note.render()) if note.display else ""


def run_review(root, test, configs=(BASE, BASE), size=(140, 40), through_builder=True):
    async def go():
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await app.workers.wait_for_complete()
            app.draft.confirmed = SEL
            for c in configs:
                app.draft.add_tree(c)
            app.draft.tree = app.draft.trees[0].id
            if through_builder:
                await app.push_screen(ModelBuilderScreen())
                await pilot.pause()
                app.screen.query_one("#mb-review").focus()
                await pilot.press("enter")  # the Review Batch button
            else:
                app.open_review()
            await settle(app, pilot)
            await test(app, pilot, app.screen)
    asyncio.run(go())


async def settle(app, pilot):
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def lines(screen):
    return [str(o.prompt) for o in screen.query_one(DetailsList).options]


def test_opens_from_the_builder_with_header_queue_and_two_views_only(tmp_path):
    async def go(app, pilot, screen):
        assert isinstance(screen, ReviewBatchScreen)
        head = str(screen.query_one("#rv-name").render()) + str(screen.query_one("#rv-info").render())
        assert "Experiment batch" in head and "NCBI Disease" in head and "train.txt" in head
        assert "Evaluation: dev" in head and "2 trees" in head and "continue" in head
        queue = [str(o.prompt) for o in screen.query_one(Queue).options]
        assert len(queue) == 2 and "01" in queue[0] and "Tree 1" in queue[0] and "Depth 2" in queue[0]
        assert "Ready" in queue[1] or "Caution" in queue[1]
        assert [v for v, _ in VIEWS] == ["overview", "differences"]
        assert not any("check" in str(w.id or "").lower() for w in screen.query("*"))
        assert not screen.query_one("#rv-launch").disabled and "Ready" in str(screen.query_one("#rv-state").render())
        left, right = screen.query_one("#rv-left").region.width, screen.query_one("#rv-details").region.width
        assert abs(left / (left + right) - 0.30) < 0.04
        await pilot.press("escape")
        assert app.screen is screen and "Press esc again to leave Review Batch." in footer(app)
        await pilot.press("escape")
        assert isinstance(app.screen, ModelBuilderScreen)  # esc: back to the previous New Experiment screen
    run_review(make_root(tmp_path), go)


def test_overview_shows_applied_values_by_stage_and_differences_are_a_second_view(tmp_path):
    async def go(app, pilot, screen):
        text = "\n".join(lines(screen))
        for word in ("FEATURES", "HIERARCHY", "MATCHER", "SEARCH", "SapBERT", "Clusters per node", "Beam size"):
            assert word in text
        assert "MATCHER" in text and "128" in text  # the applied n_clusters, not a default
        screen.query_one(Queue).focus()
        await pilot.press("down")  # Tree 2 selected, nothing edited
        assert screen.selected == app.draft.trees[1].id and app.draft.trees[1].config == BASE
        await pilot.press("right")  # arrows never leave the queue
        assert app.focused is screen.query_one(Queue)
        await pilot.press("enter")  # enter = review the details
        assert app.focused is screen.query_one(DetailsList)
        await pilot.press("right")
        assert screen.view == "differences" and app.focused is screen.query_one(DetailsList)
        await pilot.press("left")
        assert screen.view == "overview"
        await pilot.press("2")
        assert screen.view == "differences" and "No configuration differences from reference." in "\n".join(lines(screen))
        await pilot.press("1")
        assert screen.view == "overview"
        screen.query_one(Queue).focus()
        await pilot.press("tab")  # Tab reaches Launch Batch (the queue's local region), shift+tab comes back
        assert app.focused is screen.query_one("#rv-launch")
        await pilot.press("shift+tab")
        assert app.focused is screen.query_one(Queue)
        assert not any(screen.query_one(b).can_focus for b in ("#rv-grab", "#rv-details-open", "#rv-edit"))
        await pilot.press("enter")  # enter goes to the details, where Tab cycles tabs and list only
        assert app.focused is screen.query_one(DetailsList)
        await pilot.press("tab")
        assert app.focused is screen.query_one(ViewTabs)
        await pilot.press("tab")
        assert app.focused is screen.query_one(DetailsList)
    run_review(make_root(tmp_path), go)


def test_differences_against_the_reference_survive_reordering_and_rename(tmp_path):
    other = tweaked(("predict_config", "beam_size"), 20, tweaked(("depth",), 3))

    async def go(app, pilot, screen):
        first, second = app.draft.trees
        screen.query_one(Queue).focus()
        await pilot.press("down")
        screen.query_one(DetailsList).focus()
        await pilot.press("2")
        text = "\n".join(lines(screen))
        assert "REFERENCE  Tree 1" in text and "SELECTED   Tree 2" in text
        assert "Tree depth" in text and "Beam size" in text and "20" in text and "HIERARCHY" in text
        screen.query_one(Queue).focus()
        await pilot.press("alt+left")  # Tree 2 moves first
        assert [t.id for t in app.draft.trees] == [second.id, first.id]
        app.draft.rename(first.id, "Baseline")
        screen.fill()
        text = "\n".join(lines(screen))
        assert "REFERENCE  Baseline" in text and "SELECTED   Tree 2" in text  # same reference, by id
        assert "Tree depth" in text  # a rename changed no configuration
    run_review(make_root(tmp_path), go, configs=(BASE, other))


def test_reordering_keeps_identities_selection_and_boundaries(tmp_path):
    async def go(app, pilot, screen):
        a, b, c = app.draft.trees
        configs = {t.id: (t.name, t.config) for t in app.draft.trees}
        queue = screen.query_one(Queue)
        queue.focus()
        await pilot.press("alt+right")  # a: later
        assert [t.id for t in app.draft.trees] == [b.id, a.id, c.id] and screen.selected == a.id
        assert queue.highlighted_option.id == a.id and "02" in str(queue.options[1].prompt)
        assert "position 2 of 3" in footer(app)
        await pilot.press("alt+left", "alt+left")  # a: earlier (alt+arrows = [ ]), then at the start
        assert [t.id for t in app.draft.trees] == [a.id, b.id, c.id] and "already first" in footer(app)
        await pilot.press("alt+right")
        assert [t.id for t in app.draft.trees] == [b.id, a.id, c.id]
        await pilot.press("alt+right")
        assert [t.id for t in app.draft.trees] == [b.id, c.id, a.id] and "position 3 of 3" in footer(app)
        await pilot.press("alt+right")
        assert "already last" in footer(app)
        assert {t.id: (t.name, t.config) for t in app.draft.trees} == configs  # nothing but the order changed
        assert queue.highlighted_option.id == a.id
    run_review(make_root(tmp_path), go, configs=(BASE, tweaked(("depth",), 3), tweaked(("depth",), 4)))


def test_batch_rename_confirm_cancel_and_validation(tmp_path):
    async def go(app, pilot, screen):
        box = screen.query_one("#rv-rename", Input)
        await pilot.press("f2")
        assert box.display and app.focused is box and box.value == "Experiment batch"
        box.value = "NCBI Hierarchy Comparison"
        await pilot.press("escape")  # cancelled: the name is kept
        assert not box.display and app.draft.name == "Experiment batch" and app.focused is screen.query_one(Queue)
        await pilot.press("f2")
        box.value = "  "
        await pilot.press("enter")
        assert box.display and "Name required." in footer(app) and app.draft.name == "Experiment batch"
        box.value = "NCBI Hierarchy Comparison"
        await pilot.press("enter")
        assert app.draft.name == "NCBI Hierarchy Comparison" and not box.display
        assert "NCBI Hierarchy Comparison" in str(screen.query_one("#rv-name").render())
    run_review(make_root(tmp_path), go)


def test_invalid_tree_shows_issues_in_the_queue_and_launch_is_refused(tmp_path):
    bad = tweaked(("clustering_config", "kwargs", "n_clusters"), "abc")

    async def go(app, pilot, screen):
        queue = [str(o.prompt) for o in screen.query_one(Queue).options]
        assert "Invalid · 1 issue" in queue[1] and "Ready" in queue[0]
        assert "Not ready · 1 blocking issue" in str(screen.query_one("#rv-state").render())
        assert "1 configuration issue must be fixed." in str(screen.query_one("#rv-reason").render())
        await pilot.press("l")
        assert "Cannot launch an invalid batch" in footer(app)
        await pilot.press("down")
        text = "\n".join(lines(screen))
        assert "Invalid (abc)" in text and "Issue: Clusters per node: Whole number" in text and "[Fix\u00a0Issue]" in text
        assert not any("Issue:" in line for line in lines_for_tree(screen, app.draft.trees[0]))  # the other tree is clean
    run_review(make_root(tmp_path), go, configs=(BASE, bad))


def lines_for_tree(screen, tree):
    return [str(line.text) for line in review_details.overview(tree, screen.tree_issues[tree.id])]


def test_fix_issue_opens_the_field_and_review_context_returns(tmp_path):
    bad = tweaked(("clustering_config", "kwargs", "n_clusters"), "abc")

    async def go(app, pilot, screen):
        screen.query_one(Queue).focus()
        await pilot.press("down")
        screen.query_one(DetailsList).focus()
        await pilot.press("2")  # second tree, Differences view
        screen.query_one(DetailsList).scroll_to(y=0, animate=False)
        await pilot.press("enter")  # the issue row is under the cursor (first fix row)
        await settle(app, pilot)
        builder = app.screen
        assert isinstance(builder, ModelBuilderScreen) and builder.tree.id == app.draft.trees[1].id
        ed = builder.editing
        assert ed is not None and ed.slot.id == "hierarchy" and app.draft.slot == "hierarchy"
        i = next(i for i, f in enumerate(ed.fields) if f.path[-1] == "n_clusters")
        assert app.focused.id == f"mbf-{i}" and "Whole number" in str(builder.query_one(f"#mbe-{i}").render())
        assert builder.focus_path == ("clustering_config", "kwargs", "n_clusters")
        text, first = preview(builder.proposed(), ed.slot.keys, 60, builder.focus_path)
        assert '"n_clusters": "abc"' in text.plain.split("\n")[first]
        assert app.draft.trees[1].config == bad  # nothing applied by opening it
        builder.query_one(f"#mbf-{i}", Input).value = "64"
        await pilot.pause()
        assert app.draft.trees[1].config == bad  # typing alone commits nothing
        await pilot.press("ctrl+s")  # Apply
        assert app.draft.trees[1].config["clustering_config"]["kwargs"]["n_clusters"] == 64
        for _ in range(2):  # Apply left the grid focused: Tree Selection, then the previous screen
            await pilot.press("escape")
        await settle(app, pilot)
        assert app.screen is screen and screen.view == "differences" and screen.selected == app.draft.trees[1].id
        assert not any("Issue:" in line for line in lines(screen)) and "Invalid" not in str(
            screen.query_one(Queue).options[1].prompt)
    run_review(make_root(tmp_path), go, configs=(BASE, bad))


def test_review_edit_cycles_do_not_stack_screens(tmp_path):
    async def go(app, pilot, screen):
        depth = len(app.screen_stack)
        screen.query_one(Queue).focus()
        await pilot.press("down")
        picked = screen.selected
        for _ in range(3):
            await pilot.press("e")  # Edit: the Builder over this Review
            await settle(app, pilot)
            builder = app.screen
            assert isinstance(builder, ModelBuilderScreen) and builder.from_review
            builder.query_one("#mb-review").focus()
            await pilot.press("enter")  # Review Batch: back to the one below, not a new one
            await settle(app, pilot)
            assert app.screen is screen and len(app.screen_stack) == depth and screen.selected == picked
        assert sum(isinstance(s, ReviewBatchScreen) for s in app.screen_stack) == 1
    run_review(make_root(tmp_path), go, configs=(BASE, tweaked(("depth",), 3)), through_builder=True)


def test_fix_issue_is_refused_while_a_tree_is_carried(tmp_path):
    bad = tweaked(("clustering_config", "kwargs", "n_clusters"), "abc")

    async def go(app, pilot, screen):
        before = [t.id for t in app.draft.trees]
        screen.query_one(Queue).focus()
        await pilot.press("down", "space")  # carry the invalid tree
        assert screen.grab
        details = screen.query_one(DetailsList)
        details.focus()  # what a mouse click on the Differences list does
        await pilot.press("2")
        details.scroll_to(y=0, animate=False)
        await pilot.press("enter")  # the Fix Issue row
        await settle(app, pilot)
        assert app.screen is screen and screen.grab and "Space drops the tree first" in footer(app)
        assert [t.id for t in app.draft.trees] == before
        screen.query_one(Queue).focus()
        await pilot.press("escape")  # cancel: no exception, order as before
        assert screen.grab is None and [t.id for t in app.draft.trees] == before
        details.focus()
        await pilot.press("enter")  # outside a reorder the fix opens its editor
        await settle(app, pilot)
        assert isinstance(app.screen, ModelBuilderScreen)
    run_review(make_root(tmp_path), go, configs=(BASE, bad))


def test_cancelled_fix_leaves_the_issue_in_place(tmp_path):
    bad = tweaked(("depth",), 0)

    async def go(app, pilot, screen):
        screen.query_one(Queue).focus()
        await pilot.press("down")
        screen.query_one(DetailsList).focus()
        await pilot.press("enter")
        await settle(app, pilot)
        builder = app.screen
        assert builder.editing is not None and builder.editing.slot.id == "hierarchy"
        await pilot.click("#mb-cancel")
        for _ in range(3):
            await pilot.press("escape")
        await settle(app, pilot)
        assert app.screen is screen and app.draft.trees[1].config == bad
        assert "Invalid · 1 issue" in str(screen.query_one(Queue).options[1].prompt)
    run_review(make_root(tmp_path), go, configs=(BASE, bad))


def test_unsupported_component_opens_its_library_and_tree_name_opens_rename(tmp_path):
    bad = {**copy.deepcopy(BASE), "matcher_config": {"type": "svm"}}

    async def go(app, pilot, screen):
        screen.query_one(Queue).focus()
        await pilot.press("down")
        screen.query_one(DetailsList).focus()
        await pilot.press("enter")
        await settle(app, pilot)
        builder = app.screen
        assert builder.editing is None and app.focused is builder.query_one(Library) and app.draft.slot == "matcher"
        assert "No supported component" in footer(app)
        await pilot.press("escape", "escape", "escape")
        await settle(app, pilot)
        assert app.screen is screen
        app.draft.trees[1].name = "a\tb"
        screen.revalidate()
        await settle(app, pilot)
        issue = next(i for i in screen.tree_issues[app.draft.trees[1].id] if i.dest == "rename")
        screen.fix(issue)
        await settle(app, pilot)
        assert app.focused is app.screen.query_one("#mb-rename")
    run_review(make_root(tmp_path), go, configs=(BASE, bad))


def test_dataset_issue_opens_dataset_setup_on_the_mapping_and_confirm_clears_it(tmp_path):
    root = make_root(tmp_path)

    async def go(app, pilot, screen):
        (root / "datasets/A/dev.pubtator").unlink()
        screen.revalidate()
        await settle(app, pilot)
        text = "\n".join(lines(screen))
        assert text.count("Development file not found") == 1 and "SHARED DATASET" in text
        assert "Not ready" in str(screen.query_one("#rv-state").render())
        screen.query_one(DetailsList).focus()
        await pilot.press("enter")  # Fix Issue on the dataset row
        await settle(app, pilot)
        setup = app.screen
        assert isinstance(setup, DatasetSetupScreen) and setup.fix is not None
        assert setup.draft.selected == SEL.key
        box = setup.query_one("#ds-rows")
        assert box.highlighted_option.id == "eval:dev" and "not found" in footer(app)
        (root / "datasets/A/dev.pubtator").write_text(PUB)  # corrected through the normal flow
        setup.action_confirm()
        assert app.draft.confirmed is not None and "returns to Review Batch" in footer(app)
        await pilot.press("m")  # back to Review Batch
        await settle(app, pilot)
        assert app.screen is screen
        assert "SHARED DATASET" not in "\n".join(lines(screen)) and "Ready" in str(screen.query_one("#rv-state").render())
    run_review(root, go)


def test_moving_the_cursor_does_not_reread_files_or_revalidate(tmp_path, monkeypatch):
    calls = []

    async def go(app, pilot, screen):
        from xmr4el_tui import review as rv
        real = rv.dataset_issues
        monkeypatch.setattr(rv, "dataset_issues", lambda *a: calls.append(a) or real(*a))
        screen.query_one(Queue).focus()
        await pilot.press("down", "up", "down")
        screen.query_one(DetailsList).focus()
        await pilot.press("2", "1")
        await settle(app, pilot)
        assert calls == []
        screen.revalidate()
        await settle(app, pilot)
        assert len(calls) == 1
    run_review(make_root(tmp_path), go)


def test_state_is_validating_until_the_dataset_check_lands(tmp_path):
    async def go(app, pilot, screen):
        screen.dataset_issues = None
        screen.fill_head()
        assert "Validating" in str(screen.query_one("#rv-state").render())
        await pilot.press("l")
        assert "Still validating" in footer(app)
    run_review(make_root(tmp_path), go)


def test_builder_keeps_its_esc_hierarchy_and_numeric_shortcuts_with_review_above(tmp_path):
    async def go(app, pilot, screen):
        screen.query_one(Queue).focus()
        await pilot.press("e")  # edit the tree: grid of Model Builder, parent = Review Batch
        await settle(app, pilot)
        builder = app.screen
        assert isinstance(builder, ModelBuilderScreen) and app.focused is builder.query_one(PipelineGrid)
        assert builder.query_one("#mb-banner").display and "EDITING TREE  Tree 1" in str(builder.query_one("#mb-banner").render())
        assert [type(s) for s in app.screen_stack[-3:]] == [ModelBuilderScreen, ReviewBatchScreen, ModelBuilderScreen]
        first = app.draft.trees[0].id
        await pilot.press("alt+right")  # builder: alt + arrows organise the order, the shown tree stays
        assert app.draft.trees[1].id == first and builder.tree.id == first
        await pilot.press("escape", "escape")  # the grid's esc returns to Review (not Tree Selection), after asking
        assert app.screen is screen
        builder = app.screen_stack[-2]
        assert isinstance(builder, ModelBuilderScreen)  # the first builder keeps its own levels and numbers
        await pilot.press("escape")
        builder.query_one(TreeTabs).focus()
        await pilot.press("2")
        assert builder.tree.id == app.draft.trees[1].id
        assert "ctrl+tab" not in {b.key for b in ReviewBatchScreen.BINDINGS}
    run_review(make_root(tmp_path), go)


def test_narrow_terminal_keeps_both_panels_and_scrolls_independently(tmp_path):
    async def go(app, pilot, screen):
        queue, details = screen.query_one(Queue), screen.query_one(DetailsList)
        assert queue.region.height >= 3 and details.region.height >= 3 and queue.region.width >= 20
        assert details.max_scroll_y > 0 and queue.scroll_y == 0  # the long Overview scrolls without the queue
        details.focus()
        await pilot.press("end")
        assert details.highlighted is not None and queue.scroll_y == 0
    run_review(make_root(tmp_path), go, size=(80, 24))


def test_pick_up_carry_drop_and_cancel_reordering(tmp_path):
    async def go(app, pilot, screen):
        a, b, c = app.draft.trees
        queue = screen.query_one(Queue)
        queue.focus()
        await pilot.press("space")  # pick up the first tree
        assert screen.grab and "Moving Tree 1" in footer(app) and "reordering" in screen.query_one("#rv-left").border_title
        assert "Moving" in str(queue.options[0].prompt)
        await pilot.press("down", "down")  # carried to the end: the arrows move the tree, not the cursor
        assert [t.id for t in app.draft.trees] == [b.id, c.id, a.id] and queue.highlighted_option.id == a.id
        await pilot.press("tab", "e", "right")  # nothing else acts while carrying
        assert screen.grab and app.screen is screen and [t.id for t in app.draft.trees] == [b.id, c.id, a.id]
        await pilot.press("escape")  # cancel: the order before the pick-up is back
        assert screen.grab is None and [t.id for t in app.draft.trees] == [a.id, b.id, c.id] and app.screen is screen
        await pilot.press("space", "down", "enter")  # enter does not drop
        assert screen.grab and app.focused is queue and "Space drops the tree first" in footer(app)
        await pilot.press("space")  # space drops
        assert screen.grab is None and [t.id for t in app.draft.trees] == [b.id, a.id, c.id]
        assert "dropped at position 2" in footer(app) and app.focused is queue
        await pilot.click("#rv-grab")  # the mouse button picks up, and drops
        assert screen.grab and str(screen.query_one("#rv-grab").label) == "Drop"
        await pilot.pause(0.3)
        await pilot.click("#rv-grab")
        assert screen.grab is None and str(screen.query_one("#rv-grab").label) == "Reorder"
    run_review(make_root(tmp_path), go, configs=(BASE, tweaked(("depth",), 3), tweaked(("depth",), 4)))


def test_review_row_keeps_a_gap_after_a_long_label():
    assert str(review_details.row("Max document frequency", "1.0", indent=4).text).startswith("    Max document frequency 1.0")


def test_review_and_builder_controls_fit_in_80_columns(tmp_path):
    from textual.widgets import Footer

    def fits(screen, width=80):
        keys = [k for k in screen.query("FooterKey") if k.display]
        assert keys and all(k.region.right <= width for k in keys), [k.region for k in keys]
        for b in screen.query("#rv-moves Button, #mb-buttons Button, #rv-launch"):
            parent = b.parent.parent if b.parent.id == "mb-buttons" else b.parent
            assert parent.region.contains_region(b.region), b.id
        assert isinstance(screen.query_one(Footer), Footer)

    async def go(app, pilot, screen):
        await pilot.pause()
        fits(screen)
        app.screen.action_edit()
        await pilot.pause()
        fits(app.screen)
    run_review(make_root(tmp_path), go, size=(80, 24))
