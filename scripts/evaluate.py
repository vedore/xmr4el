from argparse import ArgumentParser
import contextlib
import io
import logging
import os
import sys
import time

# These libraries read their quiet settings during import.
if "-verbose" not in sys.argv:
    os.environ.update(TRANSFORMERS_VERBOSITY="error", HF_HUB_VERBOSITY="error",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")
    logging.disable(logging.WARNING)

import numpy as np

from xmr4el.data.readers import Preprocessor
from xmr4el.eval import _selfcheck, filter_labels_and_inputs, gold_rank, ranking_metrics, split_by_ranker, string_breakdown
from xmr4el.xmodel import XModel


def main():
    parser = ArgumentParser()
    parser.add_argument("-xmodel_path", type=str, required=True)
    parser.add_argument("-test_path", type=str, required=True)
    parser.add_argument("-beam_size", type=int, default=5)
    parser.add_argument("-topk", type=int, default=20)
    parser.add_argument("-alpha", type=float, default=1.0,
                        help="leaf score weight in the fusion with the matcher probability; 1 = leaf score "
                             "only (all Session 8+ rows), 0 = matcher only")
    parser.add_argument("-train_path", type=str, default=None,
                        help="train PubTator file; adds the seen/unseen mention-string breakdown")
    parser.add_argument("-scorer", choices=["ranker", "cosine"], default="cosine",
                        help="leaf label score fused with the matcher: cosine to leaf z (default; trained "
                             "rankers anti-rank, docs/results.md Session 7) or the trained rankers")
    parser.add_argument("-path_score", action="store_true",
                        help="multiply each leaf score by the routing path probability (XR-Linear); needed to "
                             "compare leaves in a beam when alpha < 1 (leaf matcher probs are only comparable within a leaf)")
    parser.add_argument("-verbose", action="store_true",
                        help="show library prints, logs and progress bars (hidden by default)")
    args = parser.parse_args()

    quiet = contextlib.nullcontext if args.verbose else lambda: contextlib.redirect_stdout(io.StringIO())


    start = time.time()
    print(f"evaluating {args.xmodel_path} on {args.test_path} (library output hidden; -verbose shows it) ...",
          flush=True)

    with quiet():
        trained_xtree = XModel.load(args.xmodel_path)
    if args.verbose:
        print(trained_xtree)

    abbrev = trained_xtree.abbrev_expansion  # train side too: same dictionary keys
    test_set = Preprocessor.load_pubtator_file(
        args.test_path, window=trained_xtree.context_window, abbrev=abbrev)
    labels = test_set["labels"]
    golden_labels, input_texts = filter_labels_and_inputs(test_set["corpus"], labels, trained_xtree.initial_labels)
    n_total, n = len(labels), len(golden_labels)

    with quiet():
        routes, score_csr = trained_xtree.predict(input_texts,
                                                  beam_size=args.beam_size,
                                                  topk=args.topk,
                                                  fusion="lp_fusion",
                                                  alpha=args.alpha,
                                                  topk_mode="per_leaf",
                                                  path_score=args.path_score,
                                                  scorer=args.scorer)

    trained_labels = np.array(trained_xtree.initial_labels)
    label_to_idx = {lab: i for i, lab in enumerate(trained_labels)}

    # Rank of the gold label in each query's score row. 0 = never retrieved.
    # Single gold CUI per mention (Preprocessor.load_pubtator_file -> one-hot Y), so acc@1 / MRR /
    # recall@k are the right metrics; precision@k (the PECOS suite) is not.
    ranks = np.zeros(n, dtype=int)
    hit_counts = []
    for r in routes:
        qi = r["query_index"]
        ranks[qi] = gold_rank(score_csr.getrow(qi), label_to_idx.get(golden_labels[qi], -1))
        cand = set(trained_labels[r.get("final_path").get("leaf_global_labels", [])])
        hit_counts.append(1 if golden_labels[qi] in cand else 0)

    metrics = ranking_metrics(ranks, ks=(5, 10, 20, 50, 100))
    mrr = lambda r: ranking_metrics(r)["MRR"] if r.size else 0.0
    nnz = score_csr.nnz / max(score_csr.shape[0], 1)

    print("-" * 72)
    print(f"tree     {os.path.basename(os.path.normpath(args.xmodel_path))}  "
          f"({len(trained_labels)} labels, emb_flag {trained_xtree.emb_flag})")
    print(f"rows     {n}/{n_total} gold label in vocabulary ({n / max(n_total, 1):.1%}); "
          f"{len(set(golden_labels))} distinct gold labels")
    print(f"search   beam {args.beam_size}, scorer {args.scorer}, alpha {args.alpha}, "
          f"path score {'on' if args.path_score else 'off'}, "
          f"{nnz:.0f} candidates/query")
    if args.scorer == "cosine" and 0 < args.alpha < 1:
        print(f"WARNING  alpha {args.alpha} mixes the matcher probability into the cosine score; "
              f"not comparable with alpha 1 rows")
    print()
    print(f"acc@1    {metrics['acc@1']:.4f}")
    print(f"MRR      {mrr(ranks):.4f}")
    print("recall   " + "  ".join(f"@{k} {metrics[f'R@{k}']:.4f}" for k in (5, 10, 20, 50, 100)))
    print(f"         @cand {np.mean(hit_counts):.4f}  (gold among the candidates: the cap for every metric)")

    # Defect #6 tie detector: a cluster-level leaf matcher gives every label in a cluster the same
    # score. Silent unless scores collapse (~2 distinct values per row instead of ~nnz).
    _rows = range(min(200, score_csr.shape[0]))
    _d = np.mean([np.unique(np.round(score_csr.getrow(i).data, 9)).size for i in _rows])
    if _d < 0.5 * nnz:
        print(f"WARNING  {_d:.1f} distinct scores per query of {nnz:.0f} candidates: leaf scores are tied (defect #6)")

    if args.scorer == "ranker":
        # A fused gain alone cannot show that rankers helped: labels without one get a cosine fallback,
        # and so does a trained ranker that raises at predict time.
        leaves = trained_xtree.model.hmodel[-1]
        failed = set().union(*(getattr(m, "ranker_failed", set()) for m in leaves))
        trained = {g for m in leaves if m.ranker_model for g in m.ranker_model.model_dict} - failed
        print(f"\nrankers  {len(trained)} scored, {len(failed)} fell back to cosine at predict time")
        gold_idx = np.array([label_to_idx[g] for g in golden_labels])
        for name, mask in split_by_ranker(gold_idx, trained):
            r = ranks[mask]
            print(f"  gold {name:14s} n {mask.sum():6d}  acc@1 {np.mean(r == 1) if r.size else 0:.4f}  "
                  f"MRR {mrr(r):.4f}")

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

    print("-" * 72)
    print(f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    if "-selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
