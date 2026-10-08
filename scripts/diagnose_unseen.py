"""Unseen-string diagnostic: where do test mentions whose exact string is not in train fail? No training.

For one saved tree, on in-vocabulary test rows:
  - unseen-string rows: tree acc@1 and gold-rank buckets (1, 2-5, 6-20, >20 in the candidates, not in the candidates =
    routing loss); flat baselines per feature block ("sapbert_char_context": mention, char, context): 1-NN over the tree's own
    train rows (`XModel.X`, labels from `XModel.Y`) and nearest label embedding (root Z)
  - fusion screen, all rows: each tree candidate re-scored by log(tree score) + beta * knn, knn = max cosine of the
    query's mention block to that label's train rows (SapBERT's nearest-synonym score); beta inf = knn only within the
    candidates. Printed as the evaluate.py string breakdown, hybrid included.
  - a TSV of the tree's unseen errors with each label's most frequent train mention, for reading
"""
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import filter_labels_and_inputs, gold_rank, mention_key, string_breakdown
from xmr4el.learning.scoring import label_max_cos
from xmr4el.xmodel import XModel

BETAS = (0, 5, 10, 20, 40, np.inf)


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def nn_labels(Q, B, b_labels):
    """Label of the cosine-nearest row of B for every row of Q."""
    return label_max_cos(Q, B, b_labels, int(b_labels.max()) + 1).argmax(axis=1)


def fused_top1(score_csr, knn, beta):
    """Per row, the candidate maximising log(tree score) + beta * knn (beta inf: knn alone); -1 if no candidates."""
    out = np.full(score_csr.shape[0], -1)
    for q in range(score_csr.shape[0]):
        lo, hi = score_csr.indptr[q], score_csr.indptr[q + 1]
        if lo == hi:
            continue
        idx = score_csr.indices[lo:hi]
        k = knn[q, idx]
        f = k if np.isinf(beta) else np.log(np.maximum(score_csr.data[lo:hi], 1e-300)) + beta * k
        out[q] = idx[f.argmax()]
    return out


def rank_buckets(ranks):
    """Shares of gold rank 1, 2-5, 6-20, >20 (still a candidate) and 0 (not a candidate)."""
    return {"1": np.mean(ranks == 1), "2-5": np.mean((ranks >= 2) & (ranks <= 5)),
            "6-20": np.mean((ranks >= 6) & (ranks <= 20)), ">20": np.mean(ranks > 20),
            "not cand": np.mean(ranks == 0)}


def main():
    ap = ArgumentParser()
    ap.add_argument("-xmodel_path", required=True)
    ap.add_argument("-test_path", required=True)
    ap.add_argument("-train_path", required=True, help="PubTator file the tree was trained from")
    ap.add_argument("-beam_size", type=int, default=10)
    ap.add_argument("-out", default="outputs/logs/unseen_errors.tsv")
    args = ap.parse_args()

    xm = XModel.load(args.xmodel_path)
    labels = np.array(xm.initial_labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    test = Preprocessor.load_pubtator_file(args.test_path, window=xm.context_window, abbrev=xm.abbrev_expansion)
    gold, texts = filter_labels_and_inputs(test["corpus"], test["labels"], labels)
    gold_idx = np.array([label_to_idx[g] for g in gold])

    train = Preprocessor.load_pubtator_file(args.train_path, abbrev=xm.abbrev_expansion)
    train_pairs = [(t, y) for t, y in zip(train["corpus"], train["labels"]) if y in label_to_idx]
    by_string, by_label = defaultdict(Counter), defaultdict(Counter)
    for t, y in train_pairs:
        by_string[mention_key(t)][y] += 1
        by_label[y][mention_key(t)] += 1
    u_idx = np.flatnonzero([mention_key(t) not in by_string for t in texts])

    score_csr = xm.predict(texts, beam_size=args.beam_size)
    ranks = np.array([gold_rank(score_csr.getrow(i), gold_idx[i]) for i in u_idx])

    print("-" * 72)
    print(f"unseen-string test rows: {len(u_idx)}/{len(texts)} in vocabulary")
    print(f"tree acc@1 {np.mean(ranks == 1):.4f}  gold rank: "
          + "  ".join(f"{k} {v:.3f}" for k, v in rank_buckets(ranks).items()))

    Y = xm.Y.tocsr()
    assert (Y.getnnz(axis=1) == 1).all(), "expected exactly one gold label per training row"
    y_train = Y.indices
    Z = _dense(xm.model.hmodel[0][0].label_embeddings)
    X = _dense(xm.text_encoder.predict(texts))
    blocks = {"all": slice(None)}
    if xm.features == "sapbert_char_context":
        n_c = xm.dimension_config["kwargs"]["n_components"]
        n_x = xm.context_dimension_config["kwargs"]["n_components"]
        d_t = X.shape[1] - n_c - n_x
        blocks.update({"mention": slice(0, d_t), "char": slice(d_t, d_t + n_c), "context": slice(d_t + n_c, None)})
    nn_pred = {}
    print("flat baselines on the unseen rows (acc@1):")
    for name, sl in blocks.items():
        nn_pred[name] = nn_labels(X[u_idx, sl], xm.X[:, sl], y_train)
        flat = nn_labels(X[u_idx, sl], Z[:, sl], np.arange(Z.shape[0]))
        print(f"  {name:<8} 1-NN train row {np.mean(nn_pred[name] == gold_idx[u_idx]):.4f}  "
              f"nearest label z {np.mean(flat == gold_idx[u_idx]):.4f}")
    ok = {k: v == gold_idx[u_idx] for k, v in nn_pred.items()}
    tree_ok = ranks == 1
    nn_col = "mention" if "mention" in ok else "all"
    print(f"  tree wrong, {nn_col} 1-NN right {np.mean(~tree_ok & ok[nn_col]):.4f}; "
          f"tree right, {nn_col} 1-NN wrong {np.mean(tree_ok & ~ok[nn_col]):.4f}")

    knn = label_max_cos(X[:, blocks[nn_col]], xm.X[:, blocks[nn_col]], y_train, len(labels))
    tree_top1 = fused_top1(score_csr, knn, 0)  # beta 0: tree order (ties: lowest label index)
    print(f"fusion screen, all rows: log(tree score) + beta * {nn_col} knn (acc@1; hybrid = dict if string seen)")
    print(f"  {'beta':>5}  {'all':>6}  {'seen 1':>6}  {'seen >1':>7}  {'unseen':>6}  {'hybrid':>6}")
    for beta in BETAS:
        top1 = tree_top1 if beta == 0 else fused_top1(score_csr, knn, beta)
        b = string_breakdown(texts, gold, [labels[i] if i >= 0 else None for i in top1], train_pairs)
        print(f"  {beta:>5}  {b['all'][1]:.4f}  {b['seen, 1 label'][1]:.4f}  {b['seen, >1 label'][1]:>7.4f}  "
              f"{b['unseen string'][1]:.4f}  {b['hybrid']:.4f}")

    def example(lab_i):
        return by_label[labels[lab_i]].most_common(1)[0][0] if lab_i >= 0 and by_label[labels[lab_i]] else ""

    with open(args.out, "w") as f:
        f.write(f"mention\tgold\tgold_example\tpred\tpred_example\tgold_rank\t{nn_col}_1nn\t{nn_col}_1nn_example\n")
        for q in np.flatnonzero(~tree_ok):
            i, nn, p = u_idx[q], nn_pred[nn_col][q], tree_top1[u_idx[q]]
            f.write(f"{mention_key(texts[i])}\t{gold[i]}\t{example(gold_idx[i])}\t{labels[p] if p >= 0 else ''}"
                    f"\t{example(p)}\t{ranks[q]}\t{labels[nn]}\t{example(nn)}\n")
    print(f"errors: {args.out} ({int((~tree_ok).sum())} rows)")


def _selfcheck():
    B = np.array([[1.0, 0], [0, 1.0], [0.7, 0.7]])
    Q = np.array([[2.0, 0.1], [0.1, 3.0], [1.0, 1.1]])
    assert list(nn_labels(Q, B, np.array([5, 6, 7]))) == [5, 6, 7]
    # labels 0 and 2 own rows {0, 2} and {1}; label 1 has none
    k = label_max_cos(Q, B, np.array([0, 2, 0]), 3, chunk=2)
    assert np.allclose(k[:, 1], -1) and np.allclose(k[:, 0], np.max(normalize(Q) @ normalize(B[[0, 2]]).T, axis=1))
    S = csr_matrix(np.array([[0.9, 0.1, 0.0], [0.0, 0.0, 0.0]]))
    knn = np.array([[0.1, 0.5, 0.9], [0, 0, 0]], dtype=np.float32)
    assert list(fused_top1(S, knn, 0)) == [0, -1]               # tree alone
    assert list(fused_top1(S, knn, 10)) == [1, -1]              # log 0.9 + 1 < log 0.1 + 5; label 2 is no candidate
    assert list(fused_top1(S, knn, np.inf)) == [1, -1]
    b = rank_buckets(np.array([1, 3, 10, 50, 0]))
    assert all(np.isclose(v, 0.2) for v in b.values()), b
    print("selfcheck ok")


if __name__ == "__main__":
    if "-selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
