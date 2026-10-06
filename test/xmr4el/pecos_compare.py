"""XMR4EL vs PECOS XR-Linear on the same split, label order and dev rows. No PECOS import here.

export  from a saved tree, write <out>/:
          X_trn.npz, Y_trn.npz  the tree's own training features and labels (column j = initial_labels[j])
          X_dev.npz, y_dev.npy  in-vocabulary dev rows (same loader and filter as eval), features from the
                                tree's text encoder
          texts.json            train/dev texts for PECOS's own TF-IDF variant, plus the label list
score   read PECOS predictions <out>/pred_*.npz (written by pecos_run.py) and print eval's metrics
"""
import glob
import json
import os
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from test_evaluate_pipeline import filter_labels_and_inputs, gold_rank, string_breakdown
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.model import XModel


def export(args):

    xm = XModel.load(args.xmodel_path)
    labels = list(xm.initial_labels)
    l2i = {lab: i for i, lab in enumerate(labels)}
    window, abbrev = getattr(xm, "context_window", None), getattr(xm, "abbrev_expansion", None)
    os.makedirs(args.out, exist_ok=True)

    # Training rows in the tree's order: first len(labels) label groups, flattened (test_train_pipeline + XModel._fit)
    groups, ids = Preprocessor.organize_pubtator_output(
        Preprocessor.load_pubtator_file(args.train_path, window=window, abbrev=abbrev))
    trn_texts = [t for g in groups[:len(labels)] for t in g]
    trn_labels = [y for g, y in zip(groups[:len(labels)], ids) for _ in g]
    X, Y = csr_matrix(xm.X, dtype=np.float32), csr_matrix(xm.Y, dtype=np.float32)
    assert (Y.getnnz(axis=1) == 1).all() and X.shape[0] == Y.shape[0] == len(trn_texts), \
        (X.shape, Y.shape, len(trn_texts))
    assert [labels[j] for j in Y.indices] == trn_labels, "tree rows do not match the reloaded training rows"

    test = Preprocessor.load_pubtator_file(args.test_path, window=window, abbrev=abbrev)
    dev_labels, dev_texts = filter_labels_and_inputs(test["corpus"], test["labels"], labels)
    enc = xm.text_encoder
    X_dev = csr_matrix(enc.predict(dev_texts), dtype=np.float32)

    for name, M in (("X_trn", X), ("Y_trn", Y), ("X_dev", X_dev)):
        M.sort_indices()
        save_npz(os.path.join(args.out, f"{name}.npz"), M)
    np.save(os.path.join(args.out, "y_dev.npy"), np.array([l2i[y] for y in dev_labels]))
    with open(os.path.join(args.out, "texts.json"), "w") as f:
        json.dump({"labels": labels, "trn_texts": trn_texts, "trn_labels": trn_labels,
                   "dev_texts": dev_texts, "dev_labels": dev_labels, "xmodel_path": args.xmodel_path}, f)
    print(f"exported {len(labels)} labels, {X.shape[0]} train rows, {X_dev.shape[0]}/{len(test['labels'])} dev rows, "
          f"{X.shape[1]} features -> {args.out}")


def report(P, gold, dev_texts, labels, train_pairs):
    """Eval's metrics for one (n_dev, n_labels) score matrix."""
    ranks = np.array([gold_rank(P.getrow(i), g) for i, g in enumerate(gold)])
    found = ranks > 0
    out = {"acc@1": np.mean(ranks == 1), "MRR": np.mean(np.where(found, 1.0 / np.maximum(ranks, 1), 0.0))}
    out.update({f"R@{k}": np.mean(found & (ranks <= k)) for k in (5, 10, 20)})
    top1 = [labels[r.indices[np.argmax(r.data)]] if r.nnz else None for r in (P.getrow(i) for i in range(P.shape[0]))]
    return out, string_breakdown(dev_texts, [labels[g] for g in gold], top1, train_pairs)


def score(args):
    with open(os.path.join(args.out, "texts.json")) as f:
        t = json.load(f)
    gold = np.load(os.path.join(args.out, "y_dev.npy"))
    train_pairs = list(zip(t["trn_texts"], t["trn_labels"]))
    files = sorted(glob.glob(os.path.join(args.out, "pred_*.npz")))
    assert files, f"no pred_*.npz in {args.out}: run pecos_run.py first"
    print(f"tree {t['xmodel_path']}: {len(t['labels'])} labels, {len(gold)} dev rows")
    for path in files:
        P = load_npz(path).tocsr()
        assert P.shape == (len(gold), len(t["labels"])), P.shape
        m, br = report(P, gold, t["dev_texts"], t["labels"], train_pairs)
        print(f"\n[{os.path.basename(path)[5:-4]}] " + "  ".join(f"{k} {v:.4f}" for k, v in m.items())
              + f"  hybrid {br['hybrid']:.4f}  ({P.nnz / P.shape[0]:.0f} candidates/query)")
        for k, (n, a, _, _) in ((k, v) for k, v in br.items() if k != "hybrid"):
            print(f"  {k:<15} n {n:5d}  acc@1 {a:.3f}")


def _selfcheck():
    P = csr_matrix(np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.0], [0.5, 0.0, 0.0]]))
    m, br = report(P, np.array([0, 0, 2]), ["a [SEP] x", "b [SEP] y", "c [SEP] z"], ["A", "B", "C"],
                   [("a [SEP] q", "A"), ("b [SEP] r", "A")])
    assert m["acc@1"] == 1 / 3 and np.isclose(m["MRR"], (1 + 0.5 + 0) / 3) and m["R@5"] == 2 / 3, m
    assert br["unseen string"][0] == 1 and np.isclose(br["hybrid"], 2 / 3), br
    print("selfcheck ok")


def main():
    ap = ArgumentParser()
    ap.add_argument("mode", choices=["export", "score", "selfcheck"])
    ap.add_argument("-xmodel_path")
    ap.add_argument("-train_path", default="datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt")
    ap.add_argument("-test_path", default="datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt")
    ap.add_argument("-out", help="default test/test_data/pecos/<tree name>")
    args = ap.parse_args()
    if args.mode == "selfcheck":
        return _selfcheck()
    if args.out is None:
        args.out = os.path.join("test/test_data/pecos", os.path.basename(os.path.normpath(args.xmodel_path)))
    export(args) if args.mode == "export" else score(args)


if __name__ == "__main__":
    main()
