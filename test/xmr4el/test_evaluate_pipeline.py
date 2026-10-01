from argparse import ArgumentParser
import numpy as np
from collections import Counter
import time


def filter_labels_and_inputs(input_texts, gold_labels, allowed_labels):
    """
    Filters out gold_labels (list of lists) and corresponding input_texts
    where the first label in each gold label list is not in allowed_labels.

    Args:
        gold_labels (List[List[str]]): Nested list of gold labels.
        input_texts (List[str]): Raw input texts, aligned with gold_labels.
        allowed_labels (Iterable[str]): Set or list of valid labels.

    Returns:
        Tuple[List[List[str]], List[str]]: Filtered gold_labels and input_texts.
    """
    allowed_set = set(allowed_labels)

    # print(allowed_set)
    # exit()

    filtered_labels = []
    filtered_texts = []

    for label_list, text in zip(gold_labels, input_texts):
        if label_list in allowed_set:
            filtered_labels.append(label_list)
            filtered_texts.append(text)

    return filtered_labels, filtered_texts

def gold_rank(row, gold_idx):
    """1-based rank of gold_idx in one CSR score row, 0 if absent."""
    ranked = row.indices[np.argsort(-row.data)]
    pos = np.flatnonzero(ranked == gold_idx)
    return int(pos[0]) + 1 if pos.size else 0


def split_by_ranker(gold_idx, trained):
    has = np.isin(gold_idx, list(trained))
    return (("with ranker", has), ("without ranker", ~has))


def _selfcheck():
    from scipy.sparse import csr_matrix
    m = csr_matrix(np.array([[0.1, 0.9, 0.5], [0.0, 0.0, 0.0]]))
    assert gold_rank(m.getrow(0), 1) == 1
    assert gold_rank(m.getrow(0), 2) == 2
    assert gold_rank(m.getrow(0), 0) == 3
    assert gold_rank(m.getrow(1), 0) == 0, "empty row must report not-found"
    (_, has), (_, no) = split_by_ranker(np.array([0, 2, 2, 5]), {2})
    assert has.tolist() == [False, True, True, False] and (has ^ no).all()
    print("selfcheck ok")


# debug_tables: list of DataFrames returned from predict(debug=True)
        # print(f"Saved debug table to {fname}")

def main():
    from xmr4el.featurization.preprocessor import Preprocessor
    from xmr4el.xmr.model import XModel

    parser = ArgumentParser()
    parser.add_argument("-xmodel_path", type=str, required=True)
    parser.add_argument("-test_path", type=str, required=True)
    parser.add_argument("-beam_size", type=int, default=5)
    parser.add_argument("-topk", type=int, default=20)
    parser.add_argument("-alpha", type=float, default=0.5,
                        help="ranker weight in the leaf fusion; 0 = matcher only (rankers still run)")
    parser.add_argument("-scorer", choices=["ranker", "cosine"], default="cosine",
                        help="leaf label score fused with the matcher: cosine to leaf z (default; trained "
                             "rankers anti-rank, docs/results.md Session 7) or the trained rankers")
    
    args = parser.parse_args()

    start = time.time()

    load_path = args.xmodel_path
    
    print(load_path, args.beam_size, args.topk, "alpha", args.alpha)
    
    trained_xtree = XModel.load(load_path)
    
    print(trained_xtree)
    for leaf in trained_xtree.model.hmodel[-1]:
        leaf.cosine_scorer = args.scorer == "cosine"
    print("scorer", args.scorer)
    
    test_set = Preprocessor.load_pubtator_file(args.test_path)
    
    corpus = test_set["corpus"]
    labels = test_set["labels"]
    
    print("Corpus", corpus[:1], len(corpus), type(corpus))
    print("Labels", labels[:1], len(labels), type(labels))
    print("Initial Labels", len(trained_xtree.initial_labels))
    
    golden_labels, input_texts = filter_labels_and_inputs(corpus, labels, trained_xtree.initial_labels)

    n_total = len(labels)
    n_kept = len(golden_labels)
    print(f"In-vocabulary mentions: {n_kept}/{n_total} ({n_kept / max(n_total, 1):.1%}) "
          f"-- {n_total - n_kept} zero-shot mentions dropped before scoring")
    print("Unique gold CUIs kept:", np.unique(np.array(golden_labels)).shape[0])

    routes, score_csr = trained_xtree.predict(input_texts,
                                              beam_size=args.beam_size,
                                              topk=args.topk,
                                              fusion="lp_fusion",
                                              alpha=args.alpha,
                                              topk_mode="per_leaf")

    trained_labels = np.array(trained_xtree.initial_labels)
    label_to_idx = {lab: i for i, lab in enumerate(trained_labels)}

    # Rank of the gold label in each query's score row. 0 = never retrieved.
    # Single gold CUI per mention (preprocessor.py:113 -> one-hot Y), so acc@1 / MRR /
    # recall@k are the right metrics; precision@k (the PECOS suite) is not.
    ranks = np.zeros(len(golden_labels), dtype=int)
    hit_counts = []
    for r in routes:
        qi = r["query_index"]
        ranks[qi] = gold_rank(score_csr.getrow(qi), label_to_idx.get(golden_labels[qi], -1))
        cand = set(trained_labels[r.get("final_path").get("leaf_global_labels", [])])
        hit_counts.append(1 if golden_labels[qi] in cand else 0)

    found = ranks > 0
    print("Hit counts per query:", Counter(hit_counts))
    print("recall@candidates:", np.mean(hit_counts))
    print("acc@1:", np.mean(ranks == 1))
    print("MRR:", np.mean(np.where(found, 1.0 / np.maximum(ranks, 1), 0.0)))
    for k in (1, 5, 10, 20, 50, 100):
        print(f"recall@{k}:", np.mean(found & (ranks <= k)))
    print("candidates/query (mean nnz):", score_csr.nnz / max(score_csr.shape[0], 1))

    # Split by whether the gold label's ranker actually scored. A fused gain alone cannot show that
    # learned rankers helped: labels without one get a cosine fallback, and so does a trained ranker
    # that raises at predict time.
    leaves = trained_xtree.model.hmodel[-1]
    failed = set().union(*(getattr(m, "ranker_failed", set()) for m in leaves))
    trained = {g for m in leaves if m.ranker_model for g in m.ranker_model.model_dict} - failed
    print(f"rankers: {len(trained)} scored, {len(failed)} fell back to cosine at predict time")
    gold_idx = np.array([label_to_idx[g] for g in golden_labels])
    for name, mask in split_by_ranker(gold_idx, trained):
        r = ranks[mask]
        print(f"  gold {name}: n={mask.sum()}  acc@1 {np.mean(r == 1) if r.size else 0:.4f}  "
              f"MRR {np.mean(np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)) if r.size else 0:.4f}")

    # Defect #6 tie detector. A cluster-level leaf matcher gives every label in a cluster the same
    # fused score, so ordering is CSR index order and recall@k collapses onto the random line
    # recall@cand * k/100. Distinct scores per row is the direct read: ~2 means the score is dead,
    # ~nnz means it discriminates.
    _rows = range(min(200, score_csr.shape[0]))
    _d = [np.unique(np.round(score_csr.getrow(i).data, 9)).size for i in _rows]
    _n = [score_csr.getrow(i).nnz for i in _rows]
    print(f"distinct scores/query: {np.mean(_d):.1f} of {np.mean(_n):.1f} candidates "
          f"(first {len(_d)} queries; ~2 = scores are tied, defect #6 alive)")

    # Random-ordering reference: if ranking carries no signal, recall@k ~= recall@cand * k/nnz.
    _cand = np.mean(hit_counts)
    _nnz = score_csr.nnz / max(score_csr.shape[0], 1)
    print("random-ordering line (recall@cand * k/nnz):",
          {k: round(_cand * k / max(_nnz, 1), 4) for k in (1, 5, 20, 50)})

    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    import sys
    if "-selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()