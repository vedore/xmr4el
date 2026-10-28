"""New Experiment, Dataset Setup: discovery from specs and dataset directories, registry, cheap PubTator checks,
validation, the draft (edit / confirm / discard), and the screen (keys, panels, file picker, footer confirmation)."""
import asyncio
import json
from pathlib import Path

import pytest

from xmr4el_tui import datasets
from xmr4el_tui.datasets import INVALID, MISSING, READY, REVIEW, EvalInput, Selection
from xmr4el_tui.draft import ExperimentDraft

PUB = "11|t|Heart attack risk\n11|a|Some abstract.\n11\t0\t5\tHeart\tDisease\tD006331\n\n"


def write(root, rel, text=PUB):
    path = Path(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def spec(root, name, body):
    write(root, f"configs/experiments/{name}.json", json.dumps(body))


def make_root(root):
    """Dataset A: train.pubtator (directory only) and train_plus_ctd.pubtator (spec), dev from one spec, test from an
    eval-only spec. B/disease and B/chemical: two variants. C: named by a spec, no files. A model config is skipped."""
    for f in ("train.pubtator", "train_plus_ctd.pubtator", "dev.pubtator", "test.pubtator"):
        write(root, f"datasets/A/{f}")
    for v in ("disease", "chemical"):
        for f in ("train.pubtator", "dev.pubtator"):
            write(root, f"datasets/B/{v}/{f}")
    write(root, "datasets/CTD/ctd_disease.pubtator")  # no train file: not a dataset
    tp = "datasets/A/train_plus_ctd.pubtator"
    spec(root, "a", {"name": "a", "model_config": "m.json", "train_path": tp,
                     "evals": [{"split": "dev", "test_path": "datasets/A/dev.pubtator", "train_path": tp},
                               {"split": "dev_knn0", "test_path": "datasets/A/dev.pubtator", "knn_beta": 0}]})
    spec(root, "a_test", {"name": "a_test", "evals": [{"split": "test", "test_path": "datasets/A/test.pubtator",
                                                       "train_path": tp}]})
    spec(root, "c", {"name": "c", "train_path": "datasets/C/train.pubtator",
                     "evals": [{"split": "dev", "test_path": "datasets/C/dev.pubtator"}]})
    spec(root, "model", {"features": "sapbert_char_context", "depth": 2})
    return root


def by_key(root):
    return {d.key: d for d in datasets.discover(root)}


# --- discovery ---------------------------------------------------------------------------------------------------
def test_discovery_merges_specs_and_keeps_variants(tmp_path):
    found = by_key(make_root(tmp_path))
    assert sorted(found) == ["datasets/A", "datasets/B/chemical", "datasets/B/disease", "datasets/C"]
    a = found["datasets/A"]
    assert [c.path for c in a.train] == ["datasets/A/train.pubtator", "datasets/A/train_plus_ctd.pubtator"]
    assert a.train[0].sources == ("directory",) and "a.json" in a.train[1].sources
    assert [c.path for c in a.dev] == ["datasets/A/dev.pubtator"]  # dev and dev_knn0 name one file
    assert [c.path for c in a.test] == ["datasets/A/test.pubtator"] and "a_test.json" in a.test[0].sources
    sel = a.default()
    assert sel.train == "datasets/A/train_plus_ctd.pubtator"  # the one a spec names, train.pubtator stays selectable
    assert sel.evaluation("dev") == EvalInput("dev", "datasets/A/dev.pubtator", True)
    assert sel.evaluation("test") == EvalInput("test", "datasets/A/test.pubtator", False)  # never enabled by default
    titles = (found["datasets/B/disease"].title, found["datasets/B/chemical"].title)
    assert titles == ("B — Disease", "B — Chemical")


def test_unresolved_paths_are_missing_not_available(tmp_path, monkeypatch):
    root = make_root(tmp_path / "repo")
    monkeypatch.chdir(write(tmp_path, "elsewhere/x").parent)  # relative paths resolve against the root, not the cwd
    found = by_key(root)
    c = found["datasets/C"]
    problems = datasets.validate(root, c, c.default())
    assert datasets.status(problems) == MISSING
    assert ("dev", "missing", "Development file not found: datasets/C/dev.pubtator") in problems
    assert datasets.status(datasets.validate(root, found["datasets/A"], found["datasets/A"].default())) == READY


def test_ambiguous_training_files_need_review(tmp_path):
    for f in ("train.pubtator", "train_plus_ctd.pubtator", "dev.pubtator"):
        write(tmp_path, f"datasets/D/{f}")
    d = by_key(tmp_path)["datasets/D"]  # two variants, no spec names either: none is chosen for the user
    sel = d.default()
    assert sel.train is None
    problems = datasets.validate(tmp_path, d, sel)
    assert datasets.status(problems) == REVIEW and problems[0][2] == "Choose a training file: 2 variants recognized"


def test_registry_round_trip_and_broken_registry(tmp_path):
    write(tmp_path, "mine/train.txt")
    sel = Selection("new", " Mine ", train="mine/train.txt", evals=(EvalInput("dev", None, False),
                                                                     EvalInput("test", "mine/train.txt", False)))
    key = datasets.register(tmp_path, sel)
    (d,), problem = datasets.registered(tmp_path)
    assert problem is None and d.key == key and d.name == "Mine" and d.registered
    assert [c.path for c in d.train] == ["mine/train.txt"] and d.dev == () and [c.path for c in d.test] == [
        "mine/train.txt"]
    assert datasets.register(tmp_path, Selection(key, "Renamed", train="mine/train.txt")) == key  # replaced
    assert [x.name for x in datasets.registered(tmp_path)[0]] == ["Renamed"]
    reg = Path(tmp_path, datasets.REGISTRY)
    reg.write_text("{broken")
    assert datasets.registered(tmp_path)[0] == [] and datasets.registered(tmp_path)[1]
    with pytest.raises(ValueError):
        datasets.register(tmp_path, sel)
    assert reg.read_text() == "{broken"


# --- validation --------------------------------------------------------------------------------------------------
def selection(train="datasets/A/train.pubtator", dev="datasets/A/dev.pubtator", test=None, dev_on=True,
              test_on=False):
    return Selection("datasets/A", "A", train=train, evals=(EvalInput("dev", dev, dev_on),
                                                             EvalInput("test", test, test_on)))


def check(root, sel):
    return datasets.validate(root, by_key(root)["datasets/A"], sel)


def test_validation(tmp_path):
    root = make_root(tmp_path)
    assert check(root, selection()) == []
    assert check(root, selection(train=None)) == [("train", "review", "Choose a training file: 2 variants recognized")]
    assert check(root, selection(train="datasets/A/nope.pubtator")) == [
        ("train", "missing", "Training file not found: datasets/A/nope.pubtator")]
    assert check(root, selection(dev="datasets/A/gone.pubtator")) == [
        ("dev", "missing", "Development file not found: datasets/A/gone.pubtator")]
    assert check(root, selection(dev="datasets/A/gone.pubtator", dev_on=False, test="datasets/A/test.pubtator",
                                 test_on=True)) == []  # a disabled split is not checked
    assert check(root, selection(dev_on=False)) == [("evals", "review", "Enable at least one evaluation split")]
    assert check(root, selection(dev="datasets/A/train.pubtator"))[0][1:] == (
        "invalid", "Development file is the training file: datasets/A/train.pubtator")
    dup = Selection("datasets/A", "A", train="datasets/A/train.pubtator",
                    evals=(EvalInput("dev", "datasets/A/dev.pubtator", True),
                           EvalInput("dev", "datasets/A/test.pubtator", True)))
    assert ("evals", "invalid", "Duplicate evaluation split: dev") in check(root, dup)
    assert datasets.status(check(root, dup)) == INVALID


@pytest.mark.parametrize("text, why", [("label\tsome text\nD001\tother\n", "line 1 is not"),
                                       ("11|a|abstract first\n", "line 1 is not"),
                                       ("11|t|title\njust words\n", "line 2 is not"),
                                       ("11|t|title only\n\n", "no PubTator annotation"),
                                       ("", "empty"), ("\x00\x01binary", "binary")])
def test_malformed_pubtator(tmp_path, text, why):
    write(tmp_path, "x.pubtator", text)
    kind, message = datasets.inspect(tmp_path, "x.pubtator")
    assert kind == "invalid" and why in message


def test_pubtator_in_a_txt_file_and_bounded_read(tmp_path):
    write(tmp_path, "corpus.txt")
    assert datasets.inspect(tmp_path, "corpus.txt") == (None, "ok")
    write(tmp_path, "big.pubtator", PUB * 20000)  # > SAMPLE: only the first SAMPLE bytes are read, cut line ignored
    assert datasets.inspect(tmp_path, "big.pubtator") == (None, "ok")
    write(tmp_path, "big.pubtator", PUB * 20000 + "garbage line\n")  # past the sample: left to launch-time checks
    assert datasets.inspect(tmp_path, "big.pubtator") == (None, "ok")


# --- draft -------------------------------------------------------------------------------------------------------
def test_draft_edit_confirm_discard(tmp_path):
    a = by_key(make_root(tmp_path))["datasets/A"]
    draft = ExperimentDraft()
    draft.selected = a.key
    assert draft.confirmed is None and draft.current(a) == a.default()  # selecting confirms nothing
    edited = a.default().with_eval("test", enabled=True)
    draft.edit(a, edited)
    assert draft.changed(a.key) and draft.confirmed is None
    draft.confirm(edited)
    assert draft.confirmed == edited and not draft.changed(a.key)
    later = edited.with_eval("dev", path="datasets/A/test.pubtator")
    draft.edit(a, later)  # a change after confirmation stays unconfirmed
    assert draft.confirmed == edited and draft.current(a) == later and draft.changed(a.key)
    draft.discard(a.key)
    assert draft.current(a) == edited and not draft.changed(a.key)
    draft.edit(a, later)
    draft.edit(a, edited)  # back to the confirmed values: no edit left
    assert not draft.changed(a.key)
    assert draft.confirmed.train == "datasets/A/train_plus_ctd.pubtator"  # what Model Builder / Review read
    assert [(e.split, e.path) for e in draft.confirmed.enabled()] == [("dev", "datasets/A/dev.pubtator"),
                                                                       ("test", "datasets/A/test.pubtator")]


# --- screen ------------------------------------------------------------------------------------------------------
textual = pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402
from xmr4el_tui.app import XMR4ELApp  # noqa: E402
from xmr4el_tui.screens.dashboard_screen import DashboardScreen  # noqa: E402
from xmr4el_tui.screens.dataset_setup import ConfigRows, DatasetList, DatasetSetupScreen  # noqa: E402
from xmr4el_tui.screens.file_picker import FilePicker  # noqa: E402


def footer(app):
    note = app.screen.query_one("#footer-note")
    return str(note.render()) if note.display else ""


def run_setup(root, test, size=(120, 36)):
    async def go():
        (root / "outputs/runs").mkdir(parents=True, exist_ok=True)
        app = XMR4ELApp(root)
        async with app.run_test(size=size) as pilot:
            await app.workers.wait_for_complete()
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, DatasetSetupScreen)
            await test(app, pilot, app.screen)
    asyncio.run(go())


def row_ids(screen):
    return [o.id for o in screen.query_one(ConfigRows).options]


async def to_row(pilot, screen, rid):
    rows = screen.query_one(ConfigRows)
    rows.focus()
    rows.highlighted = row_ids(screen).index(rid)
    await pilot.pause()


def test_keyboard_select_confirm_and_back(tmp_path):
    root = make_root(tmp_path)

    async def go(app, pilot, screen):
        listed = screen.query_one(DatasetList)
        assert app.focused is listed and "Ready" in str(listed.options[0].prompt)
        await pilot.press("down", "down")  # moving the cursor configures nothing
        assert app.draft.selected is None and listed.highlighted == 2
        await pilot.press("up", "up", "enter")
        assert app.draft.selected == "datasets/A" and app.draft.confirmed is None
        assert isinstance(app.focused, ConfigRows)
        await pilot.press("ctrl+tab")
        assert app.focused is listed
        await pilot.press("ctrl+tab")
        assert isinstance(app.focused, ConfigRows)
        await pilot.press("ctrl+s")
        assert app.draft.confirmed == by_key(root)["datasets/A"].default()
        assert "Dataset confirmed: A" in footer(app) and "confirmed" in str(listed.options[0].prompt)
        await pilot.press("escape")  # configuration -> Datasets
        assert app.focused is listed and isinstance(app.screen, DatasetSetupScreen)
        await pilot.press("escape")  # Datasets -> Dashboard
        assert isinstance(app.screen, DashboardScreen)
        await pilot.press("n")  # the draft is the session's: the same dataset comes back
        await pilot.pause()
        assert app.screen.query_one(DatasetList).highlighted == 0 and app.draft.confirmed.key == "datasets/A"
    run_setup(root, go)


def test_toggle_variant_and_discard_in_the_footer(tmp_path):
    root = make_root(tmp_path)

    async def go(app, pilot, screen):
        await pilot.press("enter", "ctrl+s")
        confirmed = app.draft.confirmed
        await to_row(pilot, screen, "eval:test")
        await pilot.press("space")
        assert app.draft.current(screen.shown()[0]).evaluation("test").enabled and app.draft.confirmed == confirmed
        await to_row(pilot, screen, "train:0")
        await pilot.press("space")
        assert app.draft.edits["datasets/A"].train == "datasets/A/train.pubtator"
        assert "changed since" in " ".join(str(o.prompt) for o in screen.query_one(ConfigRows).options)
        await pilot.press("u")
        assert "Discard unconfirmed changes to A?" in footer(app) and app.draft.changed("datasets/A")
        await pilot.press("escape")  # cancelled
        assert not footer(app) and app.draft.changed("datasets/A")
        await pilot.press("u")
        await pilot.press("ctrl+c")  # the quit question replaces it: a second ctrl+c would quit, nothing discarded
        assert "Quit XMR4EL?" in footer(app) and app.draft.changed("datasets/A")
        await pilot.press("escape", "u", "u")
        assert not app.draft.changed("datasets/A") and app.draft.confirmed == confirmed
        assert app.focused is screen.query_one(ConfigRows) and row_ids(screen)[
            screen.query_one(ConfigRows).highlighted] == "train:0"
    run_setup(root, go)


def test_file_picker_cancel_choose_and_restore(tmp_path):
    root = make_root(tmp_path)
    write(root, "elsewhere/notes/dev_v2.txt")

    async def go(app, pilot, screen):
        await pilot.press("enter")
        await to_row(pilot, screen, "eval:dev")
        before = app.draft.current(screen.shown()[0])
        await pilot.press("enter")
        picker = app.screen
        assert isinstance(picker, FilePicker) and picker.query_one(Input).value == "datasets/A/"
        assert picker.entries[picker.query_one("#fp-list").highlighted].name == "dev.pubtator"  # current file
        await pilot.press("escape")  # cancel: mapping, focus and row unchanged
        assert app.screen is screen and app.draft.current(screen.shown()[0]) == before
        assert app.focused is screen.query_one(ConfigRows) and row_ids(screen)[
            screen.query_one(ConfigRows).highlighted] == "eval:dev"
        await pilot.press("enter")
        await pilot.press(*"/test")  # a filter
        assert [e.name for e in app.screen.entries] == ["test.pubtator"]
        app.screen.query_one(Input).value = ""  # the project root
        await pilot.pause()
        names = [e and e.name for e in app.screen.entries]
        assert names[0] is None and {"datasets", "elsewhere"} <= set(names)  # the machine's tree, not only the repo
    run_setup(root, go)


def test_file_picker_directories_and_typed_path(tmp_path):
    root = make_root(tmp_path)
    write(root, "elsewhere/notes/dev_v2.txt")

    async def go(app, pilot, screen):
        await pilot.press("enter")
        await to_row(pilot, screen, "eval:test")
        await pilot.press("enter")
        field = app.screen.query_one(Input)
        field.value = "elsewhere/"
        await pilot.pause()
        assert [e and e.name for e in app.screen.entries] == [None, "notes"]
        await pilot.press("down", "enter")  # enter on a directory opens it
        assert field.value == "elsewhere/notes/"
        await pilot.press("up", "enter")  # ../
        assert field.value == "elsewhere/"
        field.value = str(root / "elsewhere/notes/dev_v2.txt")  # a pasted full path
        await pilot.press("enter")
        assert app.screen is screen
        test = app.draft.current(screen.shown()[0]).evaluation("test")
        assert test == EvalInput("test", "elsewhere/notes/dev_v2.txt", True)  # chosen = enabled
        assert app.focused is screen.query_one(ConfigRows)
    run_setup(root, go)


def test_confirm_refused_with_the_specific_problem(tmp_path):
    root = make_root(tmp_path)

    async def go(app, pilot, screen):
        screen.query_one(DatasetList).highlighted = [d.key for d in screen.datasets].index("datasets/C")
        await pilot.press("enter", "ctrl+s")
        assert app.draft.confirmed is None
        assert "Not confirmed: Training file not found: datasets/C/train.pubtator" in footer(app)
        assert row_ids(screen)[screen.query_one(ConfigRows).highlighted] == "train:other"
        text = " ".join(str(o.prompt) for o in screen.query_one(ConfigRows).options)
        assert "Development file not found: datasets/C/dev.pubtator" in text and "Missing files" in text
    run_setup(root, go)


def test_register_local_dataset(tmp_path):
    root = make_root(tmp_path)
    write(root, "mine/corpus_train.txt")
    write(root, "mine/corpus_dev.txt")

    async def go(app, pilot, screen):
        listed = screen.query_one(DatasetList)
        listed.highlighted = len(listed.options) - 1  # Register local dataset
        await pilot.press("enter")
        assert app.focused is screen.query_one("#ds-name")
        await pilot.press("ctrl+s")
        assert app.draft.confirmed is None and "Name required" in footer(app)
        assert app.focused is screen.query_one("#ds-name")  # the control that corrects it
        await pilot.press(*"Mine", "enter")
        for rid, path in (("train:other", "mine/corpus_train.txt"), ("eval:dev", "mine/corpus_dev.txt")):
            await to_row(pilot, screen, rid)
            await pilot.press("enter")
            app.screen.query_one(Input).value = path
            await pilot.press("enter")
        await pilot.press("ctrl+s")
        c = app.draft.confirmed
        assert c and c.key.startswith("local:") and c.name == "Mine" and c.train == "mine/corpus_train.txt"
        assert [(e.split, e.path) for e in c.enabled()] == [("dev", "mine/corpus_dev.txt")]
        assert [d.name for d in datasets.registered(root)[0]] == ["Mine"]
        assert c.key in [d.key for d in screen.datasets] and app.draft.selected == c.key
        assert not app.draft.edits
    run_setup(root, go)


def test_dashboard_offers_new_experiment(tmp_path):
    async def go(app, pilot, screen):
        await pilot.press("escape")
        assert any(b.binding.key == "n" for b in app.screen.active_bindings.values() if b.binding.show)
        assert "New Experiment (n)" in app.screen.query_one("Description").text()
    run_setup(make_root(tmp_path), go)
