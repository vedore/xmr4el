from argparse import ArgumentParser
import logging
import os
import sys
import time

# These libraries read their quiet settings during import.
if "-verbose" not in sys.argv:
    os.environ.update(TRANSFORMERS_VERBOSITY="warning", HF_HUB_VERBOSITY="warning",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")

import numpy as np

from xmr4el.data.readers import Preprocessor
from xmr4el.eval import (_selfcheck, filter_labels_and_inputs, gold_rank, ranking_metrics, string_breakdown,
                         train_string_labels)
from xmr4el.rerank import CrossEncoderReranker, label_texts, rerank_order, top_k
from xmr4el.xmodel import XModel
from xmr4el import set_verbosity

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
set_verbosity(2 if "-verbose" in sys.argv else 0)


def main():
    parser = ArgumentParser()
    parser.add_argument("-xmodel_path", type=str, required=True)
    parser.add_argument("-test_path", type=str, required=True)
    parser.add_argument("-beam_size", type=int, default=None, help="default: the tree's predict_config")
    parser.add_argument("-topk", type=int, default=None,
                        help="keep each query's topk best labels; 0 = all candidates (default: predict_config)")
    parser.add_argument("-train_path", type=str, default=None,
                        help="train PubTator file; adds the seen/unseen mention-string breakdown")
    parser.add_argument("-knn_beta", type=float, default=None,
                        help="multiply each candidate's score by exp(beta * max cosine of the mention block to the "
                             "label's training rows); 0 = off (default: predict_config; docs/results.md 2026-10-08)")
    parser.add_argument("-reranker_path", type=str, default=None,
                        help="outputs/rerankers/<tree>_<ts>: rerank the top K, one table row per -rerank_w")
    parser.add_argument("-rerank_k", type=int, default=None, help="default: the reranker's training K")
    parser.add_argument("-rerank_w", type=float, nargs="+", default=[0, 0.5, 1, 2, float("inf")],
                        help="final = log(tree score) + w * logit over the top K; inf = logit only. "
                             "Sweep on dev, pass the chosen w alone on test")
    parser.add_argument("-verbose", action="store_true",
                        help="show DEBUG diagnostics and progress bars; warnings are always visible")
    args = parser.parse_args()

    start = time.time()
    print(f"evaluating {args.xmodel_path} on {args.test_path} ...",
          flush=True)

    trained_xtree = XModel.load(args.xmodel_path)

    abbrev = trained_xtree.abbrev_expansion  # train side too: same dictionary keys
    test_set = Preprocessor.load_pubtator_file(
        args.test_path, window=trained_xtree.context_window, abbrev=abbrev)
    labels = test_set["labels"]
    golden_labels, input_texts = filter_labels_and_inputs(test_set["corpus"], labels, trained_xtree.initial_labels)
    n_total, n = len(labels), len(golden_labels)
    if n == 0:
        print(f"rows     0/{n_total} gold label in vocabulary: nothing to evaluate")
        return

    search = trained_xtree.resolve_predict_config(beam_size=args.beam_size, topk=args.topk, knn_beta=args.knn_beta)
    score_csr = trained_xtree.predict(input_texts, **search)

    trained_labels = np.array(trained_xtree.initial_labels)
    label_to_idx = {lab: i for i, lab in enumerate(trained_labels)}

    # Rank of the gold label in each query's score row. 0 = never retrieved.
    # Single gold CUI per mention (Preprocessor.load_pubtator_file -> one-hot Y), so acc@1 / MRR /
    # recall@k are the right metrics; precision@k (the PECOS suite) is not.
    ranks = np.array([gold_rank(score_csr.getrow(qi), label_to_idx[g]) for qi, g in enumerate(golden_labels)],
                     dtype=int)

    metrics = ranking_metrics(ranks, ks=(5, 10, 20, 50, 100))
    mrr = lambda r: ranking_metrics(r)["MRR"] if r.size else 0.0
    nnz = score_csr.nnz / max(score_csr.shape[0], 1)

    print("-" * 72)
    print(f"tree     {os.path.basename(os.path.normpath(args.xmodel_path))}  "
          f"({len(trained_labels)} labels, features {trained_xtree.features})")
    print(f"rows     {n}/{n_total} gold label in vocabulary ({n / max(n_total, 1):.1%}); "
          f"{len(set(golden_labels))} distinct gold labels")
    print(f"search   beam {search['beam_size']}, topk {search['topk']}, knn beta {search['knn_beta']:g}, "
          f"{nnz:.0f} candidates/query")
    print()
    print(f"acc@1    {metrics['acc@1']:.4f}")
    print(f"MRR      {mrr(ranks):.4f}")
    print("recall   " + "  ".join(f"@{k} {metrics[f'R@{k}']:.4f}" for k in (5, 10, 20, 50, 100)))
    print(f"         @cand {np.mean(ranks > 0):.4f}  (gold among the candidates: the cap for every metric)")

    train_pairs = None
    if args.train_path:
        train = Preprocessor.load_pubtator_file(args.train_path, abbrev=abbrev)
        train_pairs = [(t, y) for t, y in zip(train["corpus"], train["labels"]) if y in label_to_idx]
        tree_top1 = [trained_labels[r.indices[np.argmax(r.data)]] if r.nnz else None
                     for r in (score_csr.getrow(i) for i in range(score_csr.shape[0]))]
        b = string_breakdown(input_texts, golden_labels, tree_top1, train_pairs)
        print("\nacc@1 by mention string (dict = most frequent train label for the exact string)")
        print(f"  {'':16s} {'n':>6s} {'share':>6s} {'tree':>7s} {'dict':>7s} {'either':>7s}")
        for k, (n_k, t_k, d_k, e_k) in ((k, v) for k, v in b.items() if k != "hybrid"):
            print(f"  {k:16s} {n_k:6d} {n_k / n:6.3f} {t_k:7.4f} {d_k:7.4f} {e_k:7.4f}")
        print(f"  hybrid (dict if string seen, else tree): {b['hybrid']:.4f}")

    if args.reranker_path:
        rerank(args, trained_xtree, score_csr, input_texts, golden_labels, ranks, train_pairs)

    print("-" * 72)
    print(f"{time.time() - start:.0f} s")


def rerank(args, xm, score_csr, input_texts, golden_labels, ranks, train_pairs):
    """Reorder each row's top K by log(tree score) + w * reranker logit; one acc@1 / MRR row per w
    (by mention-string group with -train_path). Logits are computed once for all w. With -train_path, also per
    scope: rerank only rows whose mention string is unseen in train, or not seen with exactly 1 label; other rows
    keep the tree's order."""
    tree = os.path.basename(os.path.normpath(args.xmodel_path))
    reranker = CrossEncoderReranker.load(args.reranker_path, tree=tree)
    k = args.rerank_k or reranker.meta["k"]
    start = time.time()
    idx, vals = top_k(score_csr, k)
    names = label_texts(xm.training_set, xm.Y)  # index j -> label j = initial_labels[j]
    logits = reranker.score([q for q, top in zip(input_texts, idx) for _ in top],
                            [names[j] for top in idx for j in top])
    logits = np.split(logits, np.cumsum([len(top) for top in idx])[:-1])
    label_to_idx = {lab: i for i, lab in enumerate(xm.initial_labels)}
    gold_pos = [np.flatnonzero(top == label_to_idx[g]) for top, g in zip(idx, golden_labels)]
    print(f"\nreranker {os.path.basename(os.path.normpath(args.reranker_path))}: top {k}, "
          f"{sum(map(len, idx))} pairs, {time.time() - start:.0f} s")
    groups = ["all", "seen, 1 label", "seen, >1 label", "unseen string"] if train_pairs else ["all"]
    scopes = {"all rows": np.ones(len(idx), dtype=bool)}
    if train_pairs:
        n_lab = np.array([len(c) for c in train_string_labels(input_texts, train_pairs)])
        scopes.update({"unseen": n_lab == 0, "not seen 1": n_lab != 1})
    print(f"  {'scope':>10s} {'w':>5s} {'MRR':>7s} " + " ".join(f"{g:>14s}" for g in groups) + "   (acc@1)")
    for scope, on in scopes.items():
        for w in args.rerank_w:
            orders = [rerank_order(v, lg, w) if a else np.arange(len(v)) for v, lg, a in zip(vals, logits, on)]
            # gold inside the top K moves with the reorder; below K (or not retrieved) the tree rank stands
            new_ranks = np.array([int(np.flatnonzero(o == p[0])[0]) + 1 if p.size else r
                                  for o, p, r in zip(orders, gold_pos, ranks)])
            mrr = ranking_metrics(new_ranks)["MRR"]
            if train_pairs:
                top1 = [xm.initial_labels[top[o[0]]] if top.size else None for top, o in zip(idx, orders)]
                acc = string_breakdown(input_texts, golden_labels, top1, train_pairs)
                cells = [acc[g][1] for g in groups]
            else:
                cells = [np.mean(new_ranks == 1)]
            print(f"  {scope:>10s} {w:5g} {mrr:7.4f} " + " ".join(f"{c:14.4f}" for c in cells))


if __name__ == "__main__":
    if "-selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
