"""Regressions for the 2026-10-08 bug audit: knn fusion before topk, split leakage/aliasing, fitted block
widths, persistence (fused_scores, stale layers, same-second saves), reader edge cases; second audit (C1)."""
from copy import deepcopy
import gzip
import logging
import json
import pickle
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from xmr4el.data.readers import Preprocessor
from xmr4el.features.encoder import TextEncoder
from xmr4el.learning.classifiers import JointOvRLogistic
from xmr4el.learning.scoring import label_max_cos
from xmr4el.xmodel import XModel

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
    assert root.fused_scores.shape == (xm.X.shape[0], 2), "routing scores survive the node reload"
    (tmp_path / "old").write_text("x")
    with pytest.raises(FileExistsError):  # stale layer_* dirs would load into the tree
        xm.model.save(str(tmp_path))
    with patch("xmr4el.xmodel.datetime") as dt:
        dt.now.return_value.strftime.return_value = "same-second"
        xm.save(str(tmp_path / "trees"))
        xm.save(str(tmp_path / "trees"))
    assert sorted(p.name for p in (tmp_path / "trees").iterdir()) == ["xmodel_same-second", "xmodel_same-second_1"]


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
    assert r.returncode != 0 and "is the input file" in r.stderr and "2|a|a" in corpus.read_text()


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


def test_predict_config_checks(xm):
    for bad in ({"beam_size": 0}, {"topk": -1}):
        with pytest.raises(ValueError):
            xm.resolve_predict_config(**bad)


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
    xm.save(str(tmp_path))
    (saved,) = tmp_path.iterdir()
    state = pickle.loads((saved / "xmodel.pkl").read_bytes())
    view = json.loads((saved / "xmodel.json").read_text())
    assert view.keys() == state.keys()
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


def test_matcher_warns_at_max_iter(caplog):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 5)), rng.integers(0, 3, 40)
    with caplog.at_level(logging.WARNING, logger="xmr4el.learning.classifiers"):
        JointOvRLogistic(tol=1e-8, max_iter=1).fit(X, y)
    assert "max_iter=1" in caplog.text
