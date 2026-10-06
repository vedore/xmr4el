"""Screen leaf label scorers inside the tree's own leaves, with oracle routing. No tree training.

Reads `pecos_compare.py export` data (tree features, same rows/label order) and the tree's leaf label sets. Per leaf:
fit each scorer on the train rows of that leaf's labels, score the dev rows whose gold label is in the leaf, over the
leaf's labels only. acc@1 here = within-leaf accuracy given perfect routing (end-to-end ~ this x routing recall).
  cosine      PIFA centroid of the leaf label (what the tree scores with at -alpha 1)
  logreg      LogisticRegression liblinear, L2, C=1, one-vs-rest (PECOS-like solver, probabilistic output)
  logreg_bal  same, class_weight balanced (as the tree's matcher config)
  svm         LinearSVC squared hinge, L2, C=1 (PECOS XR-Linear's default loss)
  logreg_bal_tol1e-2  logreg_bal with liblinear tol 1e-2 (default 1e-4): speed variant
  joint       logreg_bal's objective solved for all leaf labels at once (JointOvRLogistic, L-BFGS on BLAS)
-svd_dims k keeps the first k columns of the char and context SVD blocks (components are ordered by singular value,
  so this approximates SVD k) and re-normalises each block: speed variant for the 768-dim blocks.
"""
import json
import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from xmr4el.classifiers import JointOvRLogistic
from xmr4el.eval import string_breakdown
from xmr4el.xmodel import XModel


SCORERS = {
    "logreg": lambda: OneVsRestClassifier(LogisticRegression(solver="liblinear", C=1.0), n_jobs=-1),
    "logreg_bal": lambda: OneVsRestClassifier(
        LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced"), n_jobs=-1),
    "svm": lambda: LinearSVC(C=1.0),
    "logreg_bal_tol1e-2": lambda: OneVsRestClassifier(
        LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", tol=1e-2), n_jobs=-1),
    "joint": lambda: JointOvRLogistic(C=1.0, class_weight="balanced"),  # logreg_bal's objective, all labels at once
}


def shrink_blocks(X, d_mention, d_char, k):
    """[mention | char SVD | context SVD] -> [mention | char[:k] | context[:k]], each block L2-normalised, then the row."""
    X = csr_matrix(X)
    blocks = [X[:, :d_mention], X[:, d_mention:d_mention + k], X[:, d_mention + d_char:d_mention + d_char + k]]
    return normalize(hstack([normalize(b) for b in blocks], format="csr"))


def leaf_scores(name, X_tr, y_tr, X_te, n_labels):
    """(n_te, n_labels) scores; y_tr holds local label indices 0..n_labels-1."""
    if n_labels == 1:
        return np.ones((X_te.shape[0], 1))
    if name == "cosine":
        Z = np.zeros((n_labels, X_tr.shape[1]))
        np.add.at(Z, y_tr, X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr)
        return np.asarray(normalize(X_te) @ normalize(Z).T)
    est = SCORERS[name]().fit(X_tr, y_tr)
    S = est.decision_function(X_te)
    return S[:, None] * np.array([-1, 1]) if S.ndim == 1 else S  # binary leaf: one column -> two


def main():

    ap = ArgumentParser()
    ap.add_argument("-xmodel_path", required=True, help="tree whose leaves are used")
    ap.add_argument("-data", help="pecos_compare.py export dir; default test/test_data/pecos/<tree name>")
    ap.add_argument("-scorers", default="cosine,logreg,logreg_bal,svm")
    ap.add_argument("-svd_dims", type=int, default=0, help="0 = all; else keep the first k char/context SVD columns")
    args = ap.parse_args()
    data = args.data or os.path.join("test/test_data/pecos", os.path.basename(os.path.normpath(args.xmodel_path)))

    X = load_npz(f"{data}/X_trn.npz").tocsr()
    y = load_npz(f"{data}/Y_trn.npz").tocsr().indices
    X_dev = load_npz(f"{data}/X_dev.npz").tocsr()
    gold = np.load(f"{data}/y_dev.npy")
    with open(f"{data}/texts.json") as f:
        t = json.load(f)
    labels, train_pairs = t["labels"], list(zip(t["trn_texts"], t["trn_labels"]))

    xm = XModel.load(args.xmodel_path)
    leaves = [np.asarray(m.local_to_global_idx) for m in xm.model.hmodel[-1]]
    if args.svd_dims:
        d_char = xm.dimension_config["kwargs"]["n_components"]
        d_ctx = xm.context_dimension_config["kwargs"]["n_components"]
        d_mention = X.shape[1] - d_char - d_ctx
        X, X_dev = (shrink_blocks(M, d_mention, d_char, args.svd_dims) for M in (X, X_dev))
    assert sorted(np.concatenate(leaves).tolist()) == list(range(len(labels))), "leaves must partition the labels"
    print(f"{len(leaves)} leaves, sizes {[len(g) for g in leaves]}; {len(gold)} dev rows; {X.shape[1]} features")

    for name in args.scorers.split(","):
        t0, pred = time.time(), np.full(len(gold), -1)
        for g in leaves:
            local = {int(j): i for i, j in enumerate(g)}
            tr, te = np.flatnonzero(np.isin(y, g)), np.flatnonzero(np.isin(gold, g))
            if te.size:
                S = leaf_scores(name, X[tr], np.array([local[j] for j in y[tr]]), X_dev[te], len(g))
                pred[te] = g[S.argmax(axis=1)]
        br = string_breakdown(t["dev_texts"], [labels[i] for i in gold], [labels[i] for i in pred], train_pairs)
        print(f"\n[{name}] within-leaf acc@1 {np.mean(pred == gold):.4f}  hybrid {br['hybrid']:.4f}  "
              f"({time.time() - t0:.0f} s)")
        for k, (n, a, _, _) in ((k, v) for k, v in br.items() if k != "hybrid"):
            print(f"  {k:<15} n {n:5d}  acc@1 {a:.3f}")


def _selfcheck():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(3, 8))
    y = np.repeat(np.arange(3), 30)
    X = normalize(C[y] + 0.2 * rng.normal(size=(90, 8)))
    for name in ("cosine", *SCORERS):
        S = leaf_scores(name, X, y, X, 3)
        assert S.shape == (90, 3) and (S.argmax(axis=1) == y).mean() > 0.9, name
        S2 = leaf_scores(name, X[y < 2], y[y < 2], X[y < 2], 2)
        assert S2.shape == (60, 2) and (S2.argmax(axis=1) == y[y < 2]).mean() > 0.9, f"{name} binary leaf"
    Xb = np.hstack([np.ones((2, 2)), np.arange(8).reshape(2, 4), np.arange(8, 16).reshape(2, 4)])
    Xs = shrink_blocks(Xb, 2, 4, 2).toarray()
    assert Xs.shape == (2, 6) and np.allclose(Xs[:, 2:4] * np.sqrt(3), normalize(Xb[:, 2:4])), "char block cut + norm"
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if "-selfcheck" in sys.argv else main()
