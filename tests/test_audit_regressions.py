"""Regressions for the 2026-10-08 bug audit: knn fusion before topk, split leakage/aliasing, fitted block
widths, persistence (training-only arrays not saved, stale layers, same-second saves), reader edge cases; second audit (C1);
third audit (2026-10-16: reader headers/offsets, cache write, OOF labels, to_pubtator, split publish); code quality
audit (2026-10-16)."""
from copy import deepcopy
import gzip
import os
import logging
import json
import pickle
from pathlib import Path
import runpy
import subprocess
import sys
import tracemalloc

import torch
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from xmr4el.data import splits
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import filter_labels_and_inputs
from xmr4el.features.reduction import DimensionModel
from xmr4el.features.encoder import TextEncoder
from xmr4el.features.label_embeddings import LabelEmbeddingFactory
from xmr4el.features import transformers
from xmr4el.hierarchy import clusterers
from xmr4el.features.transformers import Transformer, resolve_revision
from xmr4el.learning.classifiers import JointOvRLogistic
from xmr4el.learning.scoring import knn_index, label_max_cos, label_rows
from xmr4el.rerank import CrossEncoderReranker, out_of_fold_top_k
from xmr4el.xmodel import RESEARCH_STATE, XModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(
    vectorizer_config={"type": "tfidf", "kwargs": {"analyzer": "char", "ngram_range": [2, 4]}},
    dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 6, "random_state": 42}},
    clustering_config={"type": "balancedkmeans", "kwargs": {"n_clusters": 2}},
    matcher_config={"type": "jointlogisticregression", "kwargs": {"max_iter": 100}},
    features="tfidf", depth=2, min_leaf_size=2,
)


@pytest.fixture(scope="module")
def xm():
    texts = [[f"concept{j} synonym{i} group{j // 4}" for i in range(8)] for j in range(8)]
    model = XModel(**deepcopy(CONFIG))
    model.train(texts, [f"L{j}" for j in range(8)])
    return model


def test_knn_fused_before_topk(xm):
    queries = ["concept1 synonym9 group0", "concept6 group1", "synonym3"]
    full = xm.predict(queries, beam_size=2, knn_beta=50)
    top1 = xm.predict(queries, beam_size=2, knn_beta=50, topk=1)
    for i in range(len(queries)):
        row = full.getrow(i)
        assert (np.diff(row.data) <= 0).all(), "rows sorted by fused score"
        assert top1.getrow(i).indices.tolist() == [row.indices[0]] == [row.indices[np.argmax(row.data)]]


def test_persistence(xm, tmp_path):
    root = xm.model.hmodel[0][0]
    assert root.fused_scores is None and root.label_embeddings is None, "training-only arrays are not saved"
    (tmp_path / "old").write_text("x")
    with pytest.raises(FileExistsError):  # stale layer_* dirs would load into the tree
        xm.model.save(str(tmp_path))
    with patch("xmr4el.xmodel.datetime") as dt:
        dt.now.return_value.strftime.return_value = "same-second"
        xm.save(str(tmp_path / "trees"))
        xm.save(str(tmp_path / "trees"))
    assert sorted(p.name for p in (tmp_path / "trees").iterdir()) == ["xmodel_same-second", "xmodel_same-second_1"]
    saved = {p.name for p in (tmp_path / "trees").rglob("*")}
    assert not saved & {"fused_scores.npy", "label_embeddings.npy"}, "training-only arrays are not saved"
    assert XModel.load(str(tmp_path / "trees" / "xmodel_same-second")).predict(["concept1"]).nnz, "loads and predicts"


def test_feature_blocks():
    xm = XModel(features="sapbert_char_context")
    xm.text_encoder = TextEncoder(features="sapbert_char_context")
    xm.text_encoder.block_widths = [4, 3, 2]
    assert xm.feature_blocks() == {"mention": slice(0, 4), "char": slice(4, 7), "context": slice(7, 9)}
    assert xm.mention_block() == slice(0, 4)


def test_readers(tmp_path):
    tsv, labels = tmp_path / "train.tsv", tmp_path / "labels.txt"
    tsv.write_text("0\tNA\n1\tNULL\n2\tNaN\n")
    labels.write_text("A\nB\nC\n")
    assert Preprocessor.load_data_labels_from_file(str(tsv), str(labels))["corpus"] == [["NA"], ["NULL"], ["NaN"]]
    assert Preprocessor.context_window("alpha beta gamma", (6, 10), 0) == ""
    pub = tmp_path / "doc.pubtator"
    pub.write_text("1|t|\n1|a|alpha beta gamma\n1\t7\t11\tbeta\tDisease\tD1\n\n")
    out = Preprocessor.load_pubtator_file(str(pub), window=1)
    assert out["corpus"] == ["beta [SEP] alpha gamma"]


def test_split_cli(tmp_path):
    corpus = tmp_path / "corpus_pubtator_train.txt"
    corpus.write_text("1|t|t\n1|a|a\n\n2|t|t\n2|a|a\n\n")
    for name, ids in (("tr", "1\n"), ("dv", "1\n2\n"), ("te", "")):
        (tmp_path / name).write_text(ids)
    script = str(ROOT / "scripts/split_pubtator.py")
    run = lambda *a: subprocess.run([sys.executable, script, "--input", str(corpus), *a],
                                    capture_output=True, text=True, cwd=ROOT)
    r = run("--outdir", str(tmp_path / "out"), "--train_pmids", str(tmp_path / "tr"),
            "--dev_pmids", str(tmp_path / "dv"), "--test_pmids", str(tmp_path / "te"))
    assert r.returncode != 0 and "PMID 1" in r.stderr and not (tmp_path / "out").exists()
    r = run("--outdir", str(tmp_path))
    assert r.returncode != 0 and "outputs exist" in r.stderr and "2|a|a" in corpus.read_text()


# C1 (second audit, 2026-10-08)

def test_candidate_knn_equals_full_knn():
    rng = np.random.default_rng(0)
    Q, B, b_labels = rng.normal(size=(600, 5)), rng.normal(size=(40, 5)), rng.integers(0, 12, 40)
    b_labels[b_labels == 3] = 4  # label 3 has no B row: -1 in both modes
    full = label_max_cos(Q, B, b_labels, 12, chunk=64)
    rows = np.sort(rng.integers(0, 600, 900))
    cols = rng.integers(0, 12, 900)
    cols[:5] = 3
    pairs = label_max_cos(Q, B, b_labels, 12, chunk=64, rows=rows, cols=cols)
    assert np.allclose(pairs, full[rows, cols]) and (pairs[:5] == -1).all()
    one = np.isin(rows, [7, 300, 599])  # a few random queries: chunks with few candidate labels
    few = label_max_cos(Q, B, b_labels, 12, chunk=64, rows=rows[one], cols=cols[one])
    assert np.allclose(few, full[rows[one], cols[one]])
    cached = label_max_cos(Q, None, None, 12, chunk=64, rows=rows, cols=cols, index=knn_index(B, b_labels))
    assert np.array_equal(cached, pairs)  # XModel's cached training side gives the same scores
    Bs = csr_matrix(np.where(np.abs(B) > 0.5, B, 0))  # sparse training side: index stays CSR, same scores as dense
    assert isinstance(knn_index(Bs, b_labels)[0], csr_matrix)
    assert np.allclose(label_max_cos(Q, Bs, b_labels, 12, chunk=64), label_max_cos(Q, Bs.toarray(), b_labels, 12, chunk=64), atol=1e-6)
    assert np.allclose(label_max_cos(Q, Bs, b_labels, 12, chunk=64, rows=rows, cols=cols),
                       label_max_cos(Q, Bs.toarray(), b_labels, 12, chunk=64, rows=rows, cols=cols), atol=1e-6)
    # candidate mode multiplies only the candidate labels' B rows: groups 1 and 3 of rows [0, 2), [2, 5), [5, 6), [6, 9)
    idx, local = label_rows(np.array([0, 2, 5, 6]), 9, np.array([1, 3]))
    assert list(idx) == [2, 3, 4, 6, 7, 8] and list(local) == [0, 3]


def test_predict_config_checks(xm):
    for bad in ({"beam_size": 0}, {"topk": -1}, {"beam_size": 1.5}, {"topk": 2.0}, {"knn_beta": float("nan")},
                {"knn_beta": float("inf")}, {"beam_size": True},
                {"leaf_candidates": 0}, {"leaf_candidates": 1.5}, {"knn_beta": 1000}, {"knn_beta": -1}):
        with pytest.raises(ValueError):
            xm.resolve_predict_config(**bad)
    for depth in (0, -1, 1.0):
        with pytest.raises(ValueError, match="depth"):
            XModel(**{**CONFIG, "depth": depth})


def test_failed_retrain_leaves_no_tree(xm):
    """Second audit: a refit that fails after encoding must not pair the old tree with the new encoder/labels."""
    other = deepcopy(xm)
    texts = [[f"other{j} name{i}" for i in range(8)] for j in range(8)]
    with patch("xmr4el.xmodel.HierarchicalMLModel.train", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            other.train(texts, [f"M{j}" for j in range(8)])
    assert other.model is None


def test_retrain_clears_tree_id(xm, tmp_path):
    """Fifth audit: a retrained model keeps no saved id, so a reranker of the old tree is refused."""
    other = deepcopy(xm)
    other.save(str(tmp_path / "tree"))
    (tmp_path / "rr").mkdir()
    (tmp_path / "rr" / "reranker.json").write_text(json.dumps({"tree_id": other.tree_id, "max_length": 32}))
    other.train([[f"other{j} name{i}" for i in range(8)] for j in range(8)], [f"M{j}" for j in range(8)])
    assert other.tree_id is None
    with pytest.raises(ValueError, match="tree id"):
        CrossEncoderReranker.load(str(tmp_path / "rr"), other.tree_id)


def test_dense_and_sparse_features(xm):
    """Sixth audit #2/#3: SVD features stay dense through the tree; unreduced TF-IDF stays CSR (X, routing features,
    knn index) and still trains and predicts."""
    texts = [[f"concept{j} synonym{i} group{j // 4}" for i in range(8)] for j in range(8)]
    assert isinstance(xm.X, np.ndarray) and xm.X.dtype == np.float32
    config = deepcopy(CONFIG)
    config["dimension_config"]["kwargs"]["n_components"] = 10 ** 6  # SVD skipped
    sparse = XModel(**config)
    sparse.train(texts, [f"L{j}" for j in range(8)])
    assert isinstance(sparse.X, csr_matrix) and isinstance(sparse._knn_index()[0], csr_matrix)
    P = sparse.predict(["concept1 synonym9 group0", "concept6 group1"], knn_beta=10)
    assert P.shape == (2, 8) and P.getnnz(axis=1).all()


def test_save_without_save_method(xm, tmp_path):
    other = deepcopy(xm)
    other.text_encoder = object()
    with pytest.raises(TypeError):
        other.save(str(tmp_path))
    assert not list(tmp_path.iterdir()), "nothing written"


def test_split_cli_partial_and_ratios(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("1|t|t\n1|a|a\n\n")
    (tmp_path / "tr").write_text("1\n")
    script = str(ROOT / "scripts/split_pubtator.py")
    run = lambda *a: subprocess.run([sys.executable, script, "--input", str(corpus), "--outdir",
                                     str(tmp_path / "out"), *a], capture_output=True, text=True, cwd=ROOT)
    r = run("--train_pmids", str(tmp_path / "tr"))
    assert r.returncode != 0 and "all three" in r.stderr
    r = run("--train_ratio", "0.9", "--dev_ratio", "0.2")
    assert r.returncode != 0 and "<= 1" in r.stderr and not (tmp_path / "out").exists()


# C3: xmodel.json mirrors the pickled state's keys

def test_xmodel_json(xm, tmp_path):
    xm.predict(["concept1 synonym9 group0"], knn_beta=10)  # builds knn_index, which save leaves out (knn_beta 0)
    assert xm.knn_index is not None
    xm.save(str(tmp_path))
    (saved,) = tmp_path.iterdir()
    state = pickle.loads((saved / "xmodel.pkl").read_bytes())
    research = pickle.loads((saved / "research.pkl").read_bytes())
    view = json.loads((saved / "xmodel.json").read_text())
    assert view.keys() == state.keys() | research.keys() and state["knn_index"] is None
    assert tuple(research) == RESEARCH_STATE and not state.keys() & research.keys()
    assert view["initial_labels"] == {"type": "list", "len": 8, "first": "L0"}
    assert view["Z"]["shape"] == list(xm.Z.shape) and view["Y"]["nnz"] == xm.Y.nnz
    assert view["predict_config"] == xm.predict_config


# D (third audit, 2026-10-09)

def test_dict_to_pubtator_refuses_own_input(tmp_path):
    ctd = tmp_path / "ctd.tsv.gz"
    with gzip.open(ctd, "wt") as f:
        f.write("# header\nFever\tMESH:D005334\n")
    before = ctd.read_bytes()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/dict_to_pubtator.py"), "-ctd_path", str(ctd),
                        "-out", str(ctd), "-type", "Disease"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "is the input file" in r.stderr and ctd.read_bytes() == before


def test_pecos_export_refuses_nonempty_dir(tmp_path):
    (tmp_path / "pred_old.npz").write_bytes(b"")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/baselines/pecos_compare.py"), "export",
                        "-xmodel_path", "unused", "-out", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "not empty" in r.stderr


def test_beam_sweep_fails_when_a_child_fails(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/experiments/beam_sweep.py"), str(tmp_path / "none"),
                        str(tmp_path / "none"), "5", "5", "0"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "failed beam sizes: [5]" in r.stderr


def test_beam_sweep_rejects_bad_range(tmp_path):
    """Sixth audit #5: a reversed or nonpositive range fails before any evaluation."""
    for start, end in (("10", "5"), ("0", "5")):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/experiments/beam_sweep.py"), str(tmp_path / "none"),
                            str(tmp_path / "none"), start, end, "0"], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0 and "beam_start <= beam_end" in r.stderr and "Running test" not in r.stdout


def test_reader_warnings_carry_no_text(tmp_path, caplog):
    """Sixth audit #2: reader warnings name file, line and problem, never corpus text."""
    path = tmp_path / "c.pubtator"
    path.write_text("1|t|SENTINELA title\n1|a|abstract\n1\t0\t9\tSENTINELB\tDisease\tD1\n"
                    "1\t0\t9\tSENTINELC\tDisease\t\n\n")
    with caplog.at_level(logging.WARNING, logger="xmr4el.data.readers"):
        Preprocessor.load_pubtator_file(str(path))
    assert "differs from the document text" in caplog.text and "without concept id" in caplog.text
    assert "SENTINEL" not in caplog.text


def test_bench_latency_sparse_tfidf_tree(tmp_path):
    """Sixth audit #3: a TF-IDF tree without SVD (sparse X) benchmarks; the SapBERT comparison is skipped."""
    config = deepcopy(CONFIG)
    config["dimension_config"]["kwargs"]["n_components"] = 10 ** 6  # SVD skipped
    model = XModel(**config)
    model.train([[f"concept{j} synonym{i}" for i in range(4)] for j in range(4)], [f"L{j}" for j in range(4)])
    tree = model.save(str(tmp_path / "tree"))
    test = tmp_path / "test.pubtator"
    test.write_text("".join(f"{j}|t|concept{j} synonym9\n{j}|a|x\n{j}\t0\t8\tconcept{j}\tT\tL{j}\n\n" for j in range(4)))
    out = tmp_path / "bench.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/bench_latency.py"), "-xmodel_path", tree,
                        "-test_path", str(test), "-single", "2", "-out", str(out)], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    m = json.loads(out.read_text())
    assert m["sapbert"].startswith("skipped") and "tfidf" in m["batch_s"] and "xmr4el" in m["batch_s"]


def test_matcher_warns_at_max_iter(caplog):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 5)), rng.integers(0, 3, 40)
    with caplog.at_level(logging.WARNING, logger="xmr4el.learning.classifiers"):
        JointOvRLogistic(tol=1e-8, max_iter=1).fit(X, y)
    assert "max_iter=1" in caplog.text


# E (comprehensive audit, 2026-10-16)

def test_dict_to_pubtator_failure_keeps_output(tmp_path):
    ctd, out = tmp_path / "ctd.tsv.gz", tmp_path / "out.pubtator"
    with gzip.open(ctd, "wt") as f:
        f.write("Fever\tMESH:D005334\n")  # no column header: convert raises
    out.write_text("previous output\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/dict_to_pubtator.py"), "-ctd_path", str(ctd),
                        "-out", str(out), "-type", "Disease"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and out.read_text() == "previous output\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ctd.tsv.gz", "out.pubtator"], "no temp file left"


def test_model_config_checks():
    for bad in ({"features": "bogus"}, {"min_leaf_size": -1},
                {"transformer_config": {"type": "sapbert", "kwargs": {"batch_size": -1}}},
                {"predict_config": {"beam_size": None}}, {"predict_config": {"knn_beta": 1000}},
                {"predict_config": {"beam_szie": 1}}):
        with pytest.raises(ValueError):
            XModel(**{**CONFIG, **bad})
    with pytest.raises(ValueError, match="batch_size"):  # before any model load or cache lookup
        Transformer.transform(["a"], {"type": "sapbert", "kwargs": {"batch_size": -1}})


def test_split_cli_unknown_pmids_and_nan(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("1|t|t\n1|a|a\n\n2|t|t\n2|a|a\n\n")
    for name, ids in (("tr", "1\n"), ("dv", "99\n"), ("te", "")):
        (tmp_path / name).write_text(ids)
    script = str(ROOT / "scripts/split_pubtator.py")
    run = lambda *a: subprocess.run([sys.executable, script, "--input", str(corpus), "--outdir",
                                     str(tmp_path / "out"), *a], capture_output=True, text=True, cwd=ROOT)
    r = run(*[x for k, f in (("train", "tr"), ("dev", "dv"), ("test", "te")) for x in (f"--{k}_pmids", str(tmp_path / f))])
    assert r.returncode != 0 and "not in" in r.stderr and not (tmp_path / "out").exists()
    r = run("--train_ratio", "nan")
    assert r.returncode != 0 and "<= 1" in r.stderr


def test_reader_headers_and_offsets(tmp_path):
    pub = tmp_path / "doc.pubtator"
    pub.write_text("1|t|alpha a|t|b\n1|a|gamma\n1\t6\t11\ta|t|b\tDisease\tD1\n\n")
    assert Preprocessor.load_pubtator_file(str(pub))["labels"] == ["D1"], "a mention with |t| is not a title"
    for ann in ("2\t0\t5\talpha\tDisease\tD1", "1\t-4\t999\talpha\tDisease\tD1"):
        pub.write_text(f"1|t|alpha\n1|a|gamma\n{ann}\n\n")
        with pytest.raises(ValueError, match="doc.pubtator:3"):
            Preprocessor.load_pubtator_file(str(pub))
    with pytest.raises(ValueError):
        Preprocessor.load_pubtator_file(str(pub), abbrev="bogus")


def test_transformer_cache_write_failure_keeps_embeddings(tmp_path, caplog):
    (tmp_path / "file").write_text("")
    emb = np.arange(4, dtype=np.float32).reshape(2, 2)
    with patch("xmr4el.features.transformers.CACHE_DIR", str(tmp_path / "file" / "cache")), \
            patch.object(Transformer, "_predict", return_value=emb):
        _, out = Transformer.transform(["b", "a", "b"], {"type": "sapbert", "kwargs": {}})
    assert out.tolist() == [[2, 3], [0, 1], [2, 3]] and "not cached" in caplog.text


def test_out_of_fold_rejects_foreign_labels_and_folds(xm):
    data = {"corpus": ["concept0 x", "other y"], "labels": ["L0", "ZZ"], "docs": ["d1", "d2"], "spans": [(0, 1)] * 2}
    with pytest.raises(ValueError, match="not in the saved tree"):
        out_of_fold_top_k(xm, ["q1", "q2"], ["d1", "d2"], data, folds=2, k=1)
    with pytest.raises(ValueError, match="folds"):
        out_of_fold_top_k(xm, ["q1", "q2"], ["d1", "d2"], data, folds=3, k=1)


def test_to_pubtator_missing_input_leaves_no_output(tmp_path):
    out = tmp_path / "out.pubtator"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/to_pubtator.py"), "filter", "-input",
                        str(tmp_path / "missing"), "-out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and list(tmp_path.iterdir()) == [], "no output or temp file left"


def test_split_cli_never_mixes_generations(tmp_path):
    """Existing splits are refused; a failed write or publish leaves no output and no temp file."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("1|t|t\n1|a|a\n\n2|t|t\n2|a|a\n\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "corpus_pubtator_dev.txt").write_text("old")
    argv = ["split_pubtator.py", "--input", str(corpus), "--outdir", str(out)]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        runpy.run_path(str(ROOT / "scripts/split_pubtator.py"), run_name="__main__")
    assert [p.read_text() for p in out.iterdir()] == ["old"]
    (out / "corpus_pubtator_dev.txt").unlink()
    real_write, real_replace = splits.write_pubtator_file, os.replace

    def fail_on_test(pmids, blocks, path):
        if len([p for p in out.iterdir() if p.suffix == ".tmp"]) == 3:
            raise OSError("disk full")
        real_write(pmids, blocks, path)

    def fail_second(src, dst):
        if len(list(out.glob("corpus_pubtator_*"))) == 1:
            raise OSError("rename failed")
        real_replace(src, dst)
    for target, fake, msg in ((splits, ("write_pubtator_file", fail_on_test), "disk full"),
                              (os, ("replace", fail_second), "rename failed")):
        with patch.object(target, *fake), patch.object(sys, "argv", argv), pytest.raises(OSError, match=msg):
            runpy.run_path(str(ROOT / "scripts/split_pubtator.py"), run_name="__main__")
        assert list(out.iterdir()) == [], msg


# Second audit Phase C (2026-10-16)

def test_resolve_revision_pins_commit(monkeypatch):
    """#9: training resolves the transformer revision to a commit (Hub, else the local cache); the saved encoder
    keeps it."""
    cfg = {"type": "sapbert", "kwargs": {"batch_size": 7}}
    monkeypatch.setattr(transformers, "HfApi", lambda: Mock(model_info=Mock(return_value=Mock(sha="a" * 40))))
    assert resolve_revision(cfg) == {"type": "sapbert", "kwargs": {"batch_size": 7, "revision": "a" * 40}}
    assert "revision" not in cfg["kwargs"], "caller's config unchanged"
    monkeypatch.setattr(transformers, "HfApi", lambda: Mock(model_info=Mock(side_effect=OSError("offline"))))
    monkeypatch.setattr(transformers, "snapshot_download", lambda *a, **k: f"/hub/snapshots/{'b' * 40}")
    assert resolve_revision(cfg)["kwargs"]["revision"] == "b" * 40
    assert resolve_revision({"type": "stub"}) == {"type": "stub"} and resolve_revision(None) is None
    enc = TextEncoder(features="sapbert_char_context", transformer_config=cfg,
                      vectorizer_config={"type": "tfidf", "kwargs": {}},
                      context_vectorizer_config={"type": "tfidf", "kwargs": {}})
    with patch.object(TextEncoder, "_encode_text_using_transformer", staticmethod(lambda t, c, events=False: np.ones((len(t), 2)))):
        enc.encode(["aspirin [SEP] fever dose", "tumour [SEP] doc one"])
    assert enc.transformer_config["kwargs"]["revision"] == "b" * 40


# Third audit (2026-10-16)

def test_resave_loaded_tree_keeps_research(xm, tmp_path):
    """#1: a tree loaded without research state saves a complete research.pkl."""
    first = XModel.load(xm.save(str(tmp_path / "a")))
    assert first.X is None
    again = XModel.load(first.save(str(tmp_path / "b")), research=True)
    assert again.training_set == xm.training_set and (again.Y != xm.Y).nnz == 0
    assert again.tree_id == first.tree_id != xm.tree_id, "each save is a new artifact id"
    q = ["concept1 synonym9 group0"]
    assert np.allclose(again.predict(q, knn_beta=2).toarray(), xm.predict(q, knn_beta=2).toarray())


def test_splitter_headers_and_jsonl_offsets(tmp_path):
    """#2: a mention containing |t| is no header in the splitter; #6: JSONL spans index the exported context."""
    pub = tmp_path / "c.txt"
    pub.write_text("1|t|  alpha\n1|a|beta\n1\t2\t7\talpha\tT\tD1\n1\t8\t12\tabc|t|def\tT\tD2\n\n")
    blocks = splits.parse_pubtator(str(pub))
    assert list(blocks) == ["1"] and len(blocks["1"]) == 4
    splits.emit_jsonl_for_split(["1"], blocks, str(tmp_path / "o.jsonl"))
    rows = [json.loads(line) for line in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert [r["context"][r["start"]:r["end"]] for r in rows] == ["alpha", "beta"]


@pytest.mark.parametrize("cfg", [
    {"matcher_config": {"type": "jointlogisticregression", "kwargs": {"C": -1}}},
    {"matcher_config": {"type": "jointlogisticregression", "kwargs": {"C": float("nan")}}},
    {"matcher_config": {"type": "jointlogisticregression", "kwargs": {"tol": 0}}},
    {"matcher_config": {"type": "jointlogisticregression", "kwargs": {"max_iter": 0}}},
    {"clustering_config": {"type": "balancedkmeans", "kwargs": {"n_clusters": 0}}},
    {"matcher_config": {"type": "jointlogisticregression", "kwargs": {"class_weight": "balanaced"}}},
    {"clustering_config": {"type": "balancedkmeans", "kwargs": {"iter_limit": 0}}},
])
def test_bad_learner_settings_fail_at_init(cfg):
    """#4: invalid learner settings fail in XModel(), before any encoding."""
    with pytest.raises(ValueError):
        XModel(**cfg)


def test_svd_transform_checks_width():
    """#5: the used-columns shortcut keeps sklearn's feature-count check."""
    X = csr_matrix(np.random.default_rng(0).random((20, 4)))
    dim = DimensionModel.fit(X, {"type": "sklearntruncatedsvd", "kwargs": {"n_components": 2}})
    for bad in (X[:, :3], X[:, :3].toarray()):
        with pytest.raises(ValueError, match="features"):
            dim.transform(bad)


def test_cache_write_failure_keeps_embeddings():
    """#8: a cache write OSError (disk full) warns and returns the computed embeddings."""
    emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    with patch.object(Transformer, "_predict", return_value=emb), \
            patch.object(transformers.np, "save", side_effect=OSError("disk full")):
        _, out = Transformer.transform(["b", "a", "b"], {"type": "sapbert", "kwargs": {}})
    assert np.array_equal(out, emb[[1, 0, 1]]) and not os.listdir(transformers.CACHE_DIR)


def test_mismatched_inputs_raise(xm):
    """#10: no silent truncation of texts vs labels; #12: an empty batch is an empty (0, labels) matrix."""
    with pytest.raises(ValueError):
        Preprocessor.organize_pubtator_output({"corpus": ["a", "b"], "labels": ["L1"]})
    with pytest.raises(ValueError):
        filter_labels_and_inputs(["a", "b"], ["L1"], ["L1"])
    assert xm.predict([]).shape == (0, len(xm.initial_labels))


def test_transformer_cache_switch_and_empty_input(monkeypatch):
    """XMR4EL_TRANSFORMER_CACHE=0: embeddings computed, no cache file; no texts: ValueError."""
    emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    monkeypatch.setenv("XMR4EL_TRANSFORMER_CACHE", "0")
    with patch.object(Transformer, "_predict", return_value=emb) as predict:
        for _ in range(2):
            _, out = Transformer.transform(["b", "a", "b"], {"type": "sapbert", "kwargs": {}})
            assert np.array_equal(out, emb[[1, 0, 1]])
    assert predict.call_count == 2 and not os.path.exists(transformers.CACHE_DIR)
    with pytest.raises(ValueError, match="no texts"):
        Transformer.transform([], {"type": "sapbert", "kwargs": {}})


# Code quality audit (2026-10-16)

def test_classifier_rejects_bad_inputs():
    """#3: NaN/Inf X, no rows, or misaligned y raise before fitting."""
    y = np.array([0, 1, 0, 1])
    for X in (np.array([[0.0], [np.nan], [1.0], [2.0]]), np.array([[0.0], [np.inf], [1.0], [2.0]]),
              csr_matrix(np.array([[0.0], [np.nan], [1.0], [2.0]]))):
        with pytest.raises(ValueError, match="finite"):
            JointOvRLogistic().fit(X, y)
    with pytest.raises(ValueError):
        JointOvRLogistic().fit(np.zeros((0, 1)), np.zeros(0))
    with pytest.raises(ValueError):
        JointOvRLogistic().fit(np.ones((3, 1)), y)


def test_untrained_model_save_and_train_fail_first(tmp_path):
    """#4: saving an untrained model writes nothing; #11: train without learner configs fails before encoding."""
    with pytest.raises(TypeError, match="train or load"):
        XModel(features="tfidf").save(str(tmp_path))
    assert not os.listdir(tmp_path)
    with patch.object(XModel, "_fit") as fit, pytest.raises(ValueError, match="matcher_config"):
        XModel(features="tfidf", clustering_config=CONFIG["clustering_config"]).train([["a"]], ["L"])
    fit.assert_not_called()


def test_empty_contexts_fail_before_transformer():
    """#5: all-empty contexts fail at the TF-IDF fit, before the transformer encodes."""
    with patch.object(TextEncoder, "_encode_text_using_transformer") as enc, \
            patch("xmr4el.features.encoder.resolve_revision", side_effect=lambda c: c), \
            pytest.raises(ValueError, match="empty vocabulary"):
        TextEncoder(vectorizer_config=CONFIG["vectorizer_config"], context_vectorizer_config={"type": "tfidf", "kwargs": {}},
                    transformer_config={"type": "sapbert", "kwargs": {}}).encode(["aspirin [SEP] ", "fever [SEP] "])
    enc.assert_not_called()


def test_reader_empty_concept_and_mention_mismatch(tmp_path, caplog):
    """#8 / user 2026-10-20: an annotation with an empty or missing concept id is warned (file:line) and skipped, the
    rest of the file is read; relation lines are not warned; a mention != its offsets' text warns and is kept."""
    pub = tmp_path / "doc.pubtator"
    pub.write_text("1|t|Aspirin works\n1|a|x\n1\t0\t7\tAspirin\tChemical\t\n1\t0\t7\tAspirin\tChemical\n"
                   "1\tCID\tD1\tD2\n1\t8\t13\tworks\tDisease\tD3\n\n")
    with caplog.at_level(logging.WARNING):
        assert Preprocessor.load_pubtator_file(str(pub))["labels"] == ["D3"]
    assert "doc.pubtator:3" in caplog.text and "doc.pubtator:4" in caplog.text and "doc.pubtator:5" not in caplog.text
    caplog.clear()
    pub.write_text("1|t|Aspirin works\n1|a|x\n1\t0\t7\tWRONGTX\tChemical\tD1\n\n")
    with caplog.at_level(logging.WARNING):
        assert Preprocessor.load_pubtator_file(str(pub))["labels"] == ["D1"]
    assert "doc.pubtator:3" in caplog.text and "differs" in caplog.text and "WRONGTX" not in caplog.text


def test_transformer_rejects_nul_texts():
    """#12: NUL joins the cache key fields, so texts containing it are refused (no colliding keys)."""
    with pytest.raises(ValueError, match="NUL"):
        Transformer.transform(["a", "b\0c"], {"type": "sapbert", "kwargs": {}})


def test_training_arrays_are_float32(xm):
    """#6 / user 2026-10-20: trees train in float32 (int64 Y @ float32 X used to build a float64 PIFA)."""
    assert xm.X.dtype == xm.Z.dtype == np.float32
    assert all(n.matcher_model.model.model.W_.dtype == np.float32 for layer in xm.model.hmodel for n in layer)
    X = np.random.default_rng(0).random((2000, 300), dtype=np.float32)
    Y, _ = LabelEmbeddingFactory.label_binarizer([[i % 1000] for i in range(2000)])
    tracemalloc.start()
    Z = LabelEmbeddingFactory.generate_PIFA(X, Y)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert Z.dtype == np.float32 and peak < 3 * Z.nbytes, f"peak {peak} vs float32 Z {Z.nbytes}: a float64 copy"


# Fourth audit (user paste 2026-10-20)

def test_zero_context_equals_empty_context():
    """#1: diagnose_context's ctx-0 rows = the encoder's own empty-context rows (renormalised, norm 1)."""
    zero_context = runpy.run_path(str(ROOT / "scripts/diagnose_context.py"))["zero_context"]
    enc = TextEncoder(vectorizer_config={"type": "tfidf", "kwargs": {"analyzer": "char_wb", "ngram_range": [2, 4]}},
                      dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 3, "random_state": 0}},
                      transformer_config={"type": "stub", "kwargs": {}}, features="sapbert_char_context",
                      context_vectorizer_config={"type": "tfidf", "kwargs": {"ngram_range": [1, 1], "max_df": 1.0}},
                      context_dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 2}})
    fake = staticmethod(lambda texts, _c, events=False: np.array([[10.0 * len(t), 30.0, 1.0] for t in texts], dtype=np.float32))
    with patch.object(TextEncoder, "_encode_text_using_transformer", fake), \
            patch("xmr4el.features.encoder.resolve_revision", side_effect=lambda c: c):
        enc.encode(["aspirin [SEP] doc one", "tumour [SEP] another doc", "fever [SEP] doc dose"])
        Xq = enc.predict(["aspirin [SEP] doc one", "tumour [SEP] another doc"])
        X0 = enc.predict(["aspirin [SEP] ", "tumour [SEP] "])
    Xz = zero_context(Xq, slice(6, 8))
    assert isinstance(Xz, np.ndarray) and np.allclose(Xz, X0, atol=1e-6)


def test_transformer_output_not_copied_again():
    """#2: the stacked embeddings are not cast into a third full-size copy."""
    encode = lambda texts: np.ones((len(texts), 1024), dtype=np.float32)
    with patch.object(transformers, "sentence_model", return_value=encode):
        tracemalloc.start()
        out = Transformer._predict("m", ["x"] * 4096, batch_size=1024)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert out.dtype == np.float32 and peak < 2.5 * out.nbytes, f"peak {peak} vs output {out.nbytes}"


def test_training_ignores_torch_default_dtype():
    """#4: a float64 torch default leaves the matcher weights and clustering prices float32."""
    rng = np.random.default_rng(0)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        clf = JointOvRLogistic().fit(rng.random((20, 3), dtype=np.float32), np.arange(20) % 2)
        with patch.object(clusterers.np, "argsort", wraps=np.argsort) as argsort:
            clusterers._balanced_assign(rng.random((12, 3), dtype=np.float32), np.array([4, 4, 4]))
    finally:
        torch.set_default_dtype(old)
    assert clf.W_.dtype == np.float32 and argsort.call_args_list[0].args[0].dtype == np.float32


def test_reader_rejects_abstract_of_another_document(tmp_path):
    """#5: an abstract header must carry the active title's document id."""
    pub = tmp_path / "doc.pubtator"
    pub.write_text("1|t|Aspirin works\n2|a|Fever fell.\n1\t0\t7\tAspirin\tChemical\tD1\n\n")
    with pytest.raises(ValueError, match="doc.pubtator:2"):
        Preprocessor.load_pubtator_file(str(pub))


def test_tree_loads_under_layer_named_ancestor(xm, tmp_path):
    """#6: layer order comes from the layer directory's own name, not an ancestor's (any listdir order)."""
    texts = ["concept1 synonym0 group0", "concept6 synonym3 group1"]
    path = xm.save(str(tmp_path / "layer_99"))
    with patch("xmr4el.hierarchy.tree.plistdir", lambda d: sorted(os.listdir(d), reverse=True)):
        loaded = XModel.load(path)
    assert (loaded.predict(texts) != xm.predict(texts)).nnz == 0


def test_failed_save_keeps_tree_id(xm, tmp_path):
    """#7: tree_id changes only once xmodel.pkl is published."""
    xm.save(str(tmp_path))
    tree_id = xm.tree_id
    with patch("xmr4el.xmodel.os.replace", side_effect=OSError("disk full")), pytest.raises(OSError):
        xm.save(str(tmp_path))
    assert xm.tree_id == tree_id
