# LOCAL TEST ARTIFACT (user's baseline comparison): kept on `repeat`, not merged into `main`.
"""XMR4EL vs PECOS XR-Linear on the same split, label order and dev rows. No PECOS import here.

export  from a saved tree, write <out>/:
          X_trn.npz, Y_trn.npz  the tree's own training features and labels (column j = initial_labels[j])
          X_dev.npz, y_dev.npy  in-vocabulary dev rows (same loader and filter as eval), features from the
                                tree's text encoder
          texts.json            train/dev texts for PECOS's own TF-IDF variant, plus the label list
          chain_<i>.npz         the tree's hierarchy as a PECOS cluster chain, top first (pecos_run.py -chain)
score   read PECOS predictions <out>/pred_*.npz (written by pecos_run.py) and print eval's metrics
agree   equivalence test: the tree's own scores (knn off) on the same rows vs each pred_*.npz, row by row
"""
import glob
import json
import os
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, load_npz, save_npz

from xmr4el.data.readers import Preprocessor
from xmr4el.eval import filter_labels_and_inputs, gold_rank, ranking_metrics, string_breakdown
from xmr4el.xmodel import XModel


def tree_chain(hml, n_labels):
    """The tree's hierarchy as PECOS's ClusterChain matrices, top first: [(n_1 x 1), (n_2 x n_1), ..., (L x n_leaves)],
    entry (child, parent) = 1; n_l = nodes of tree layer l (hmodel[l]), the last matrix maps each label to its leaf."""
    assert len(hml.hmodel[0]) == 1, "one root"
    chain = []
    for layer, maps in enumerate(hml.child_index_map[:-1]):
        pairs = [(child, parent) for parent, m in enumerate(maps) for child in m.values()]
        n_child = len(hml.hmodel[layer + 1])
        assert sorted(c for c, _ in pairs) == list(range(n_child)), "every child has exactly one parent"
        rows, cols = zip(*pairs)
        chain.append(csc_matrix((np.ones(len(pairs), np.float32), (rows, cols)), shape=(n_child, len(maps))))
    labels = [(int(g), leaf) for leaf, m in enumerate(hml.hmodel[-1]) for g in m.local_to_global_idx]
    assert sorted(g for g, _ in labels) == list(range(n_labels)), "every label in exactly one leaf"
    rows, cols = zip(*labels)
    chain.append(csc_matrix((np.ones(len(labels), np.float32), (rows, cols)), shape=(n_labels, len(hml.hmodel[-1]))))
    return chain


def export(args):
    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} is not empty: export into a fresh directory (old pred_*.npz would be scored)")
    xm = XModel.load(args.xmodel_path, research=True)
    labels = list(xm.initial_labels)
    l2i = {lab: i for i, lab in enumerate(labels)}
    window, abbrev = xm.context_window, xm.abbrev_expansion
    os.makedirs(args.out, exist_ok=True)

    # Training rows in the tree's order: first len(labels) label groups, flattened (scripts/train.py + XModel._fit)
    groups, ids = Preprocessor.organize_pubtator_output(
        Preprocessor.load_pubtator_file(args.train_path, window=window, abbrev=abbrev))
    trn_texts = [t for g in groups[:len(labels)] for t in g]
    trn_labels = [y for g, y in zip(groups[:len(labels)], ids) for _ in g]
    X, Y = csr_matrix(xm.X, dtype=np.float32), csr_matrix(xm.Y, dtype=np.float32)
    assert (Y.getnnz(axis=1) == 1).all() and X.shape[0] == Y.shape[0] == len(trn_texts), \
        (X.shape, Y.shape, len(trn_texts))
    assert [labels[j] for j in Y.indices] == trn_labels, "tree rows do not match the reloaded training rows"
    assert trn_texts == [t for g in xm.training_set[:len(labels)] for t in g], \
        f"{args.train_path} texts differ from the tree's training texts"

    test = Preprocessor.load_pubtator_file(args.test_path, window=window, abbrev=abbrev)
    dev_labels, dev_texts = filter_labels_and_inputs(test["corpus"], test["labels"], labels)
    enc = xm.text_encoder
    X_dev = csr_matrix(enc.predict(dev_texts), dtype=np.float32)

    for name, M in (("X_trn", X), ("Y_trn", Y), ("X_dev", X_dev)):
        M.sort_indices()
        save_npz(os.path.join(args.out, f"{name}.npz"), M)
    np.save(os.path.join(args.out, "y_dev.npy"), np.array([l2i[y] for y in dev_labels]))
    for i, M in enumerate(tree_chain(xm.model, len(labels))):
        save_npz(os.path.join(args.out, f"chain_{i}.npz"), M)
    with open(os.path.join(args.out, "texts.json"), "w") as f:
        json.dump({"labels": labels, "trn_texts": trn_texts, "trn_labels": trn_labels,
                   "dev_texts": dev_texts, "dev_labels": dev_labels, "xmodel_path": args.xmodel_path}, f)
    print(f"exported {len(labels)} labels, {X.shape[0]} train rows, {X_dev.shape[0]}/{len(test['labels'])} dev rows, "
          f"{X.shape[1]} features -> {args.out}")


def report(P, gold, dev_texts, labels, train_pairs):
    """Eval's metrics for one (n_dev, n_labels) score matrix."""
    ranks = np.array([gold_rank(P.getrow(i), g) for i, g in enumerate(gold)])
    out = ranking_metrics(ranks)
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


def agree(args):
    """Tree scores without knn (XModel.predict, knn_beta 0: same rows, same features as the export) vs each PECOS
    prediction: acc@1 of both, top-1 agreement, top-10 overlap (shared / list length), |score difference| on the
    labels both rank in top 10."""
    with open(os.path.join(args.out, "texts.json")) as f:
        t = json.load(f)
    gold = np.load(os.path.join(args.out, "y_dev.npy"))
    xm = XModel.load(t["xmodel_path"])
    T = xm.predict(t["dev_texts"], beam_size=args.beam_size, topk=100, knn_beta=0).tocsr()
    assert T.shape == (len(gold), len(t["labels"])), T.shape
    print(f"tree {t['xmodel_path']} (beam {args.beam_size}, knn off): {len(gold)} rows")
    for path in sorted(glob.glob(os.path.join(args.out, "pred_*.npz"))):
        P = load_npz(path).tocsr()
        print(f"\n[{os.path.basename(path)[5:-4]}] " + "  ".join(f"{k} {v:.4f}" for k, v in agreement(T, P, gold).items()))


def agreement(T, P, gold, k=10):
    top = lambda M, i: M.indices[M.indptr[i]:M.indptr[i + 1]][np.argsort(-M.data[M.indptr[i]:M.indptr[i + 1]])][:k]
    same1, overlap, diffs, acc_t, acc_p = 0, 0.0, [], 0, 0
    for i, g in enumerate(gold):
        a, b = top(T, i), top(P, i)
        acc_t += a.size > 0 and a[0] == g
        acc_p += b.size > 0 and b[0] == g
        same1 += a.size > 0 and b.size > 0 and a[0] == b[0]
        overlap += len(set(a) & set(b)) / max(len(a), len(b), 1)
        for j in set(a) & set(b):
            diffs.append(abs(T[i, j] - P[i, j]))
    n = len(gold)
    return {"acc@1_tree": acc_t / n, "acc@1_pecos": acc_p / n, "top1_agree": same1 / n, "top10_overlap": overlap / n,
            "score_diff_mean": float(np.mean(diffs)) if diffs else float("nan"),
            "score_diff_max": float(np.max(diffs)) if diffs else float("nan")}


def _selfcheck():
    P = csr_matrix(np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.0], [0.5, 0.0, 0.0]]))
    m, br = report(P, np.array([0, 0, 2]), ["a [SEP] x", "b [SEP] y", "c [SEP] z"], ["A", "B", "C"],
                   [("a [SEP] q", "A"), ("b [SEP] r", "A")])
    assert m["acc@1"] == 1 / 3 and np.isclose(m["MRR"], (1 + 0.5 + 0) / 3) and m["R@5"] == 2 / 3, m
    assert br["unseen string"][0] == 1 and np.isclose(br["hybrid"], 2 / 3), br
    a = agreement(P, csr_matrix(np.array([[0.8, 0.1, 0.0], [0.7, 0.2, 0.0], [0.5, 0.0, 0.0]])), np.array([0, 0, 2]))
    assert a["acc@1_tree"] == 1 / 3 and a["acc@1_pecos"] == 2 / 3 and a["top1_agree"] == 2 / 3, a
    assert np.isclose(a["score_diff_max"], 0.5) and a["top10_overlap"] == 1.0, a
    print("selfcheck ok")


def main():
    ap = ArgumentParser()
    ap.add_argument("mode", choices=["export", "score", "agree", "selfcheck"])
    ap.add_argument("-xmodel_path")
    ap.add_argument("-train_path", default="datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt")
    ap.add_argument("-test_path", default="datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt")
    ap.add_argument("-out", help="default outputs/pecos/<tree name>")
    ap.add_argument("-beam_size", type=int, default=10, help="agree: the tree's beam (match pecos_run.py -beams)")
    args = ap.parse_args()
    if args.mode == "selfcheck":
        return _selfcheck()
    if args.out is None:
        args.out = os.path.join("outputs/pecos", os.path.basename(os.path.normpath(args.xmodel_path)))
    {"export": export, "score": score, "agree": agree}[args.mode](args)


if __name__ == "__main__":
    main()
