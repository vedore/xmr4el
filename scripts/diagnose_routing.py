"""Phase A diagnostic: where does the dev signal die? No training.

For one saved tree, on train rows and on in-vocabulary dev mentions:
  - root top-1 cluster accuracy: trained matcher vs cosine-to-cluster-centroid over Z
  - the same cosine router restricted to each feature block ("sapbert_char_context": mention, char, context)
  - flat nearest-label acc@1: argmax cosine(x, Z) over every label, no hierarchy
  - dictionary baseline: normalised mention string -> most frequent train label
"""
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import numpy as np
from sklearn.preprocessing import normalize
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import mention_key
from xmr4el.xmodel import XModel


def cluster_centroids(Z, cluster_of):
    """Normalised mean label embedding per cluster, shape (K, D)."""
    cent = np.zeros((cluster_of.max() + 1, Z.shape[1]))
    np.add.at(cent, cluster_of, Z)
    return normalize(cent)


def cos_argmax(X, B):
    return (normalize(X) @ normalize(B).T).argmax(axis=1)


def dictionary_baseline(train_pairs, test_pairs):
    """(acc@1, coverage) of predicting the most frequent train label for an exact mention string."""
    counts = defaultdict(Counter)
    for text, y in train_pairs:
        counts[mention_key(text)][y] += 1
    preds = [counts[mention_key(t)].most_common(1)[0][0] if mention_key(t) in counts else None
             for t, _ in test_pairs]
    gold = [y for _, y in test_pairs]
    return (np.mean([p == y for p, y in zip(preds, gold)]),
            np.mean([p is not None for p in preds]))


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def report(split, X, gold_label, root, Z, cluster_of, blocks):
    K = cluster_of.max() + 1
    gold_c = cluster_of[gold_label]
    freq = np.sort(np.bincount(gold_c, minlength=K))[::-1] / len(gold_c)
    print(f"\n[{split}] n={len(gold_label)}  K={K}  chance 1/K={1 / K:.3f}  majority cluster={freq[0]:.3f}"
          f"  fixed top-2 clusters={freq[:2].sum():.3f}")

    proba = _dense(root.matcher_model.predict_proba(X))
    acc = np.mean(proba.argmax(axis=1) == gold_c)
    top2 = np.mean((np.argsort(-proba, axis=1)[:, :2] == gold_c[:, None]).any(axis=1))
    print(f"  root matcher            cluster acc {acc:.3f}  ({acc * K:.2f}x chance)  top-2 {top2:.3f}")

    Xd = _dense(X)
    if "mention" in blocks:
        # Rows are L2-normalised after [transformer | SVD(TF-IDF)] concat with no per-block scaling
        share = (Xd[:, blocks["mention"]] ** 2).sum(axis=1).mean()
        print(f"  mention block share of squared row norm: {share:.3f}")
    for name, sl in blocks.items():
        acc = np.mean(cos_argmax(Xd[:, sl], cluster_centroids(Z[:, sl], cluster_of)) == gold_c)
        flat = np.mean(cos_argmax(Xd[:, sl], Z[:, sl]) == gold_label)
        print(f"  cosine/{name:<16} cluster acc {acc:.3f}  ({acc * K:.2f}x chance)  "
              f"| flat nearest-label acc@1 {flat:.4f}")


def main():

    ap = ArgumentParser()
    ap.add_argument("-xmodel_path", required=True)
    ap.add_argument("-train_path", required=True, help="PubTator file the tree was trained from")
    ap.add_argument("-test_path", required=True)
    ap.add_argument("-max_rows", type=int, default=5000, help="random subsample per split")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    xm = XModel.load(args.xmodel_path)
    root = xm.model.hmodel[0][0]
    if root.is_last_layer:
        ap.error("one-layer tree: the root is a leaf, no routing to diagnose")
    Z = _dense(xm.Z)
    cluster_of = np.asarray(root.cluster_model.c_node.argmax(axis=1)).ravel()
    labels = list(xm.initial_labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    blocks = {"all": slice(None)}
    blocks.update(xm.feature_blocks())

    # Train rows: X[i] pairs with Y[i]; Y rows are one-hot over label columns
    Y = xm.Y.tocsr()
    assert (Y.getnnz(axis=1) == 1).all(), "expected exactly one gold label per training row"
    tr = rng.permutation(Y.shape[0])[:args.max_rows]
    report("train", xm.X[tr], Y.indices[tr], root, Z, cluster_of, blocks)

    abbrev = xm.abbrev_expansion
    test = Preprocessor.load_pubtator_file(args.test_path, window=xm.context_window,
                                           abbrev=abbrev)
    pairs = [(t, y) for t, y in zip(test["corpus"], test["labels"]) if y in label_to_idx]
    print(f"\nin-vocabulary dev mentions: {len(pairs)}/{len(test['labels'])}")
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:args.max_rows]]
    X_dev = xm.text_encoder.predict([t for t, _ in pairs])
    report("dev", X_dev, np.array([label_to_idx[y] for _, y in pairs]), root, Z, cluster_of, blocks)

    train = Preprocessor.load_pubtator_file(args.train_path, abbrev=abbrev)
    train_pairs = [(t, y) for t, y in zip(train["corpus"], train["labels"]) if y in label_to_idx]
    acc, cov = dictionary_baseline(train_pairs, pairs)
    print(f"\ndictionary baseline (dev): acc@1 {acc:.4f}  mention-string coverage {cov:.3f}")


def _selfcheck():
    Z = normalize(np.array([[1.0, 0], [0.9, 0.1], [0, 1.0]]))
    cluster_of = np.array([0, 0, 1])
    cent = cluster_centroids(Z, cluster_of)
    assert cent.shape == (2, 2) and np.allclose(np.linalg.norm(cent, axis=1), 1)
    assert list(cos_argmax(np.array([[1.0, 0.05], [0.1, 1.0]]), cent)) == [0, 1]
    acc, cov = dictionary_baseline(
        [("Aspirin [SEP] ctx", "A"), ("aspirin [SEP] x", "A"), ("aspirin [SEP] y", "B"), ("tumor [SEP] z", "T")],
        [("ASPIRIN [SEP] q", "A"), ("tumor [SEP] q", "X"), ("unseen [SEP] q", "U")],
    )
    assert np.isclose(acc, 1 / 3) and np.isclose(cov, 2 / 3)

    # Gold clusters [0, 1, 2, 0]: matcher top-1 hits 2/4, its top-2 covers all; fixed top-2 = 3/4
    root = Mock()
    root.matcher_model.predict_proba.return_value = np.array(
        [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.7, 0.2, 0.1]])
    out = StringIO()
    with redirect_stdout(out):
        report("t", np.eye(4, 2), np.arange(4), root, np.eye(4, 2), np.array([0, 1, 2, 0]), {})
    assert "fixed top-2 clusters=0.750" in out.getvalue() and "acc 0.500" in out.getvalue(), out.getvalue()
    assert "top-2 1.000" in out.getvalue(), out.getvalue()
    print("selfcheck ok")


if __name__ == "__main__":
    if "-selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
