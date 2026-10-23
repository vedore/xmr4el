from fake_runs import make_run
from xmr4el_tui import runs
from xmr4el_tui.compare import default_splits, rows, verdict

HASHED = {"preprocessing": {"context_window": None, "abbrev_expansion": None}, "population_sha256": "p1"}


def pair(tmp_path, a_kw=None, b_kw=None):
    make_run(tmp_path, "a", **(a_kw or {}))
    make_run(tmp_path, "b", stamp="2026-10-21_00-00-00", **(b_kw or {}))
    found = runs.scan(tmp_path)
    b, a = found
    return a, b, found


def deltas(v, a, b):
    return {k: d for k, _, _, d in rows(a.eval("dev").metrics, b.eval("dev").metrics, v)}


def test_population_hash_equal_and_different(tmp_path):
    a, b, found = pair(tmp_path, {"metrics": {"dev": HASHED}},
                       {"tree_id": "other", "metrics": {"dev": {**HASHED, "acc@1": 0.9, "search": {"beam_size": 5}}}})
    v = verdict(a, "dev", b, "dev", found)
    assert v.level == "hash" and v.subset_deltas and v.timing_delta
    assert "tree" in v.differs and "beam_size 10 -> 5" in v.differs
    d = deltas(v, a, b)
    assert d["acc@1"] == "-0.0246" and d["ms/query"] == "+0.00   same device" and d["rows"] == ""

    # same file and counts, different retained rows: no subset delta, acc@1_all kept
    a, b, found = pair(tmp_path / "2", {"metrics": {"dev": HASHED}},
                       {"metrics": {"dev": {**HASHED, "population_sha256": "p2"}}})
    v = verdict(a, "dev", b, "dev", found)
    assert (v.level, v.subset_deltas, v.all_rows_delta) == ("different", False, True)
    d = deltas(v, a, b)
    assert d["acc@1"] == d["MRR"] == d["R@cand"] == d["ms/query"] == "" and d["acc@1_all"] == "+0.0000"


def test_different_file_or_preprocessing(tmp_path):
    a, b, found = pair(tmp_path, b_kw={"test_sha": "sha-other"})
    v = verdict(a, "dev", b, "dev", found)
    assert v.level == "different" and not (v.subset_deltas or v.all_rows_delta)
    assert "contents differ" in v.reasons[0]
    a, b, found = pair(tmp_path / "2", {"metrics": {"dev": HASHED}},
                       {"metrics": {"dev": {**HASHED, "preprocessing": {"context_window": 5}}}})
    v = verdict(a, "dev", b, "dev", found)
    assert v.level == "different" and "context_window None vs 5" in v.reasons[0]
    assert set(deltas(v, a, b).values()) == {""}


def test_derivation_for_old_runs(tmp_path):
    # train + eval-only run of one tree, no population hash: derived through the training run
    make_run(tmp_path, "t", tree_id="t1")
    make_run(tmp_path, "e", stamp="2026-10-21_00-00-00", kind="eval", tree_id="t1")
    found = runs.scan(tmp_path)
    e, t = found
    v = verdict(t, "dev", e, "dev", found)
    assert v.level == "derived" and v.subset_deltas

    # dirty tree -> unknown
    a, b, found = pair(tmp_path / "2", b_kw={"tree_id": "t2", "git": {"commit": "c", "dirty": True}})
    v = verdict(a, "dev", b, "dev", found)
    assert v.level == "unknown" and not v.subset_deltas and v.all_rows_delta

    # training run not synced -> unknown
    make_run(tmp_path / "3", "t", tree_id="t1")
    make_run(tmp_path / "3", "e", stamp="2026-10-21_00-00-00", kind="eval", tree_id="t9")
    found = runs.scan(tmp_path / "3")
    e, t = found
    v = verdict(t, "dev", e, "dev", found)
    assert v.level == "unknown" and not v.subset_deltas

    # different training files -> populations differ
    a, b, found = pair(tmp_path / "4", b_kw={"tree_id": "t3", "train_sha": "sha-train2"})
    v = verdict(a, "dev", b, "dev", found)
    assert v.level == "different" and not v.subset_deltas and v.all_rows_delta


def test_hybrid_needs_same_breakdown_input_and_devices(tmp_path):
    hyb = {**HASHED, "hybrid": 0.93, "hybrid_all": 0.89}
    a, b, found = pair(tmp_path, {"metrics": {"dev": hyb}},
                       {"metrics": {"dev": hyb}, "eval_train_sha": "sha-dict", "device": "mps"})
    v = verdict(a, "dev", b, "dev", found)
    d = deltas(v, a, b)
    assert v.level == "hash" and d["hybrid"] == d["hybrid_all"] == "" and d["acc@1"] == "+0.0000"
    assert d["ms/query"] == "devices differ: cuda vs mps"


def test_selection(tmp_path):
    make_run(tmp_path, "a", splits=("dev", "test"))
    make_run(tmp_path, "b", stamp="2026-10-21_00-00-00", kind="eval", splits=("test",), metrics={"test": None})
    found = runs.scan(tmp_path)
    b, a = found
    assert default_splits(a, b) == ("test", "test") and default_splits(b, a) == ("test", "test")
    v = verdict(a, "test", b, "test", found)
    assert v.level == "unknown" and v.reasons == ("B: no metrics for test (not produced yet)",)
    make_run(tmp_path, "c", stamp="2026-10-22_00-00-00", splits=("train_dev",))
    c = runs.scan(tmp_path)[0]
    assert default_splits(a, c) == ("dev", "train_dev")
