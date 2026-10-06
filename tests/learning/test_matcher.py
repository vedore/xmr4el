import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from xmr4el.learning.matcher import MatcherTrainer


def test_matcher_single_positive():
    """Identity C at the leaf (defect #6) turns the matcher into one-vs-rest over labels, where a
    label can have a single positive. SGD's `early_stopping` stratifies its validation split and
    raises on that; MLModel.train disables it for the last layer only. Asserts both halves."""

    cfg = json.load(open(Path(__file__).resolve().parents[2] / "configs/xmr4el_flag6_sapbert_config.json"))["matcher_config"]  # SGD matcher
    rng = np.random.default_rng(0)
    n, d, L = 40, 12, 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    Y = sp.lil_matrix((n, L), dtype=np.float32)
    for i in range(n - 1):
        Y[i, i % (L - 1)] = 1.0
    Y[n - 1, L - 1] = 1.0          # label L-1 gets exactly one positive
    Y = Y.tocsr()
    assert np.asarray(Y.sum(axis=0)).ravel()[L - 1] == 1

    kw = {"local_to_global_idx": list(range(L)),
          "global_to_local_idx": {g: g for g in range(L)},
          "C": sp.eye(L, format="csr", dtype=np.float32)}

    def _run(early_stopping):
        c = {**cfg, "kwargs": {**cfg["kwargs"], "early_stopping": early_stopping}}
        return MatcherTrainer.train(X=X, Y=Y, config=c, **kw)

    try:
        _run(True)
        raise AssertionError("expected a stratify ValueError with early_stopping=True")
    except ValueError as exc:
        assert "least populated" in str(exc), exc

    _, _, M, model = _run(False)
    assert M.shape == (n, L), M.shape
    assert model.predict_proba(X).shape == (n, L)
    print("selfcheck ok")
