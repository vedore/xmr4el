"""Regressions for the 2026-10-08 bug audit: knn fusion before topk, split leakage/aliasing, fitted block
widths, persistence (fused_scores, stale layers, same-second saves), reader edge cases."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from xmr4el.data.readers import Preprocessor
from xmr4el.features.encoder import TextEncoder
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
