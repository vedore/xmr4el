"""scripts/baselines/nn_baselines.py: chunked per-label max ranks = a brute-force loop (ties to the lower label index),
and the script end to end on a synthetic tree (exact train strings -> tfidf 1-NN acc@1 1)."""
import json
from pathlib import Path
import random
import runpy
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from xmr4el.data.readers import Preprocessor
from xmr4el.xmodel import XModel

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/baselines/nn_baselines.py"
nn_ranks = runpy.run_path(str(SCRIPT))["nn_ranks"]
CONFIG = dict(vectorizer_config={"type": "tfidf", "kwargs": {"analyzer": "char", "ngram_range": [2, 4]}},
              dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 6, "random_state": 42}},
              clustering_config={"type": "balancedkmeans", "kwargs": {"n_clusters": 2}},
              matcher_config={"type": "jointlogisticregression", "kwargs": {"max_iter": 100}},
              features="tfidf", depth=2, min_leaf_size=2)


def test_nn_ranks_match_brute_force():
    rng = np.random.default_rng(0)
    n_labels, B = 7, rng.normal(size=(40, 5))
    B[5] = B[3]  # a duplicate training row under another label: exact ties
    b_labels = np.r_[np.arange(n_labels), rng.integers(0, n_labels, 33)]
    b_labels[5], b_labels[3] = 6, 2
    Q = np.vstack([rng.normal(size=(29, 5)), B[3:4]])
    gold = rng.integers(0, n_labels, len(Q))
    gold[-1] = 6
    B, Q = normalize(B), normalize(Q)
    for b, q in ((B, Q), (csr_matrix(B), csr_matrix(Q))):
        ranks, top1 = nn_ranks(q, b, b_labels, gold, n_labels, chunk=4)
        for i in rng.permutation(len(Q))[:15].tolist() + [len(Q) - 1]:
            s = np.array([max(Q[i] @ B[j] for j in np.flatnonzero(b_labels == lab)) for lab in range(n_labels)])
            g = gold[i]
            want = 1 + sum(s[k] > s[g] or (np.isclose(s[k], s[g]) and k < g) for k in range(n_labels) if k != g)
            assert ranks[i] == want and top1[i] == int(np.argmax(s)), i
    assert ranks[-1] == 2  # gold 6 ties with label 2 (same row), the lower index ranks first


def test_nn_baselines_script(tmp_path):
    r = random.Random(0)
    with open(tmp_path / "train.pubtator", "w") as f:
        for d in range(120):
            j = r.randrange(8)
            m = f"concept{j} syn{r.randrange(6)}"
            title = f"Study of {m} in group{j // 4}"
            s = title.index(m)
            f.write(f"{d}|t|{title}\n{d}|a|Results were clear.\n{d}\t{s}\t{s + len(m)}\t{m}\tDisease\tL{j}\n\n")
    xm = XModel(**CONFIG)
    xm.train(*Preprocessor.organize_pubtator_output(Preprocessor.load_pubtator_file(str(tmp_path / "train.pubtator"))))
    tree = xm.save(str(tmp_path / "trees"))
    res = subprocess.run([sys.executable, str(SCRIPT), "-xmodel_path", tree, "-test_path", "train.pubtator",
                          "-train_path", "train.pubtator", "-out", "nn.json"], cwd=tmp_path,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads((tmp_path / "nn.json").read_text())
    assert set(out) == {"sapbert", "tfidf"} and out["tfidf"]["rows"] == out["tfidf"]["rows_total"] == 120
    assert out["tfidf"]["acc@1"] == 1.0 and out["tfidf"]["strings"]["all"]["n"] == 120
