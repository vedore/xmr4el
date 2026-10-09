"""Evaluation metrics shared by the evaluate, run_experiment and diagnostic scripts."""
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from xmr4el.data.readers import Preprocessor

RECALL_KS = (5, 10, 20, 50, 100)


def mention_key(text):
    """Normalised mention string (text before [SEP]) used by the dictionary baselines."""
    return text.split("[SEP]", 1)[0].strip().lower()


def filter_labels_and_inputs(input_texts, gold_labels, allowed_labels):
    """Keep the (label, text) rows whose label is in allowed_labels, in input order.

    Args:
        input_texts (List[str]): Raw input texts, aligned with gold_labels.
        gold_labels (List[str]): One gold label per text (flat, as the PubTator loader returns).
        allowed_labels (Iterable[str]): Valid labels.

    Returns:
        Tuple[List[str], List[str]]: Filtered gold_labels and input_texts.
    """
    allowed_set = set(allowed_labels)

    filtered_labels = []
    filtered_texts = []

    for label_list, text in zip(gold_labels, input_texts):
        if label_list in allowed_set:
            filtered_labels.append(label_list)
            filtered_texts.append(text)

    return filtered_labels, filtered_texts

def gold_rank(row, gold_idx):
    """1-based rank of gold_idx in one CSR score row, 0 if absent. Ties keep CSR order (= rerank.top_k)."""
    ranked = row.indices[np.argsort(-row.data, kind="stable")]
    pos = np.flatnonzero(ranked == gold_idx)
    return int(pos[0]) + 1 if pos.size else 0


def ranking_metrics(ranks, ks=(5, 10, 20)):
    """Aggregate existing ranks without changing callers' tie ordering."""
    ranks = np.asarray(ranks)
    found = ranks > 0
    metrics = {"acc@1": np.mean(ranks == 1),
               "MRR": np.mean(np.where(found, 1.0 / np.maximum(ranks, 1), 0.0))}
    metrics.update({f"R@{k}": np.mean(found & (ranks <= k)) for k in ks})
    return metrics


def train_string_labels(test_texts, train_pairs):
    """Per test text: Counter of train labels for its exact mention string (empty = unseen string)."""
    counts = defaultdict(Counter)
    for t, y in train_pairs:
        counts[mention_key(t)][y] += 1
    return [counts.get(mention_key(t), Counter()) for t in test_texts]


def string_breakdown(test_texts, gold, tree_top1, train_pairs):
    """acc@1 of the tree and of the mention dictionary, split by whether the exact mention string
    occurs in train (and with how many labels). Returns {group: (n, tree, dict, either)} plus the
    hybrid (dictionary if the string was seen, tree otherwise)."""
    seen = train_string_labels(test_texts, train_pairs)
    n_lab = np.array([len(c) for c in seen])
    dict_ok = np.array([bool(c) and c.most_common(1)[0][0] == g for c, g in zip(seen, gold)])
    tree_ok = np.array([p == g for p, g in zip(tree_top1, gold)])
    groups = {"all": n_lab >= 0, "seen, 1 label": n_lab == 1,
              "seen, >1 label": n_lab > 1, "unseen string": n_lab == 0}
    out = {k: (int(g.sum()), tree_ok[g].mean() if g.any() else 0.0,
               dict_ok[g].mean() if g.any() else 0.0, (tree_ok | dict_ok)[g].mean() if g.any() else 0.0)
           for k, g in groups.items()}
    out["hybrid"] = np.where(n_lab > 0, dict_ok, tree_ok).mean()
    return out


def evaluate_tree(xm, tree, test_path, train_path=None, beam_size=None, topk=None, knn_beta=None):
    """Rank the PubTator `test_path` with the loaded tree `xm` (named `tree`); search settings left None come from
    its predict_config. Rows whose gold label is not in the tree are dropped (`rows` of `rows_total` kept).
    Returns (metrics, state): metrics is JSON-ready (acc@1, MRR, R@k, R@cand, with `train_path` the mention-string
    breakdown and hybrid); state = the ranked rows for the reranker (None if no row is kept)."""
    abbrev = xm.abbrev_expansion  # train side too: same dictionary keys
    test = Preprocessor.load_pubtator_file(test_path, window=xm.context_window, abbrev=abbrev)
    gold, texts = filter_labels_and_inputs(test["corpus"], test["labels"], xm.initial_labels)
    labels = np.array(xm.initial_labels)
    m = {"tree": tree, "test_path": test_path, "train_path": train_path, "labels": len(labels),
         "features": xm.features, "rows": len(gold), "rows_total": len(test["labels"]),
         "distinct_gold": len(set(gold))}
    if not gold:
        return m, None
    m["search"] = xm.resolve_predict_config(beam_size=beam_size, topk=topk, knn_beta=knn_beta)
    start = time.perf_counter()
    scores = xm.predict(texts, **m["search"])
    m["predict_seconds"] = round(time.perf_counter() - start, 1)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    # Rank of the gold label in each query's score row. 0 = never retrieved.
    # Single gold CUI per mention (Preprocessor.load_pubtator_file -> one-hot Y), so acc@1 / MRR /
    # recall@k are the right metrics; precision@k (the PECOS suite) is not.
    ranks = np.array([gold_rank(scores.getrow(qi), label_to_idx[g]) for qi, g in enumerate(gold)], dtype=int)
    m.update({k: float(v) for k, v in ranking_metrics(ranks, ks=RECALL_KS).items()})
    m["R@cand"] = float(np.mean(ranks > 0))
    m["candidates_per_query"] = scores.nnz / max(scores.shape[0], 1)
    train_pairs = None
    if train_path:
        train = Preprocessor.load_pubtator_file(train_path, abbrev=abbrev)
        train_pairs = [(t, y) for t, y in zip(train["corpus"], train["labels"]) if y in label_to_idx]
        top1 = [labels[r.indices[np.argmax(r.data)]] if r.nnz else None
                for r in (scores.getrow(i) for i in range(scores.shape[0]))]
        b = string_breakdown(texts, gold, top1, train_pairs)
        m["hybrid"] = float(b.pop("hybrid"))
        m["strings"] = {k: dict(zip(("n", "tree", "dict", "either"), (int(v[0]), *map(float, v[1:]))))
                        for k, v in b.items()}
    state = {"scores": scores, "texts": texts, "gold": gold, "ranks": ranks, "train_pairs": train_pairs}
    return m, state


def format_metrics(m):
    """The scripts/evaluate.py report of `evaluate_tree` metrics, without the separator lines."""
    n = m["rows"]
    lines = [f"tree     {m['tree']}  ({m['labels']} labels, features {m['features']})",
             f"rows     {n}/{m['rows_total']} gold label in vocabulary ({n / max(m['rows_total'], 1):.1%}); "
             f"{m['distinct_gold']} distinct gold labels"]
    if not n:
        return f"rows     0/{m['rows_total']} gold label in vocabulary: nothing to evaluate"
    s = m["search"]
    lines += [f"search   beam {s['beam_size']}, topk {s['topk']}, knn beta {s['knn_beta']:g}, "
              f"{m['candidates_per_query']:.0f} candidates/query", "",
              f"acc@1    {m['acc@1']:.4f}", f"MRR      {m['MRR']:.4f}",
              "recall   " + "  ".join(f"@{k} {m[f'R@{k}']:.4f}" for k in RECALL_KS),
              f"         @cand {m['R@cand']:.4f}  (gold among the candidates: the cap for every metric)"]
    if "strings" in m:
        lines += ["", "acc@1 by mention string (dict = most frequent train label for the exact string)",
                  f"  {'':16s} {'n':>6s} {'share':>6s} {'tree':>7s} {'dict':>7s} {'either':>7s}"]
        lines += [f"  {k:16s} {g['n']:6d} {g['n'] / n:6.3f} {g['tree']:7.4f} {g['dict']:7.4f} {g['either']:7.4f}"
                  for k, g in m["strings"].items()]
        lines.append(f"  hybrid (dict if string seen, else tree): {m['hybrid']:.4f}")
    return "\n".join(lines)


def _selfcheck():
    m = csr_matrix(np.array([[0.1, 0.9, 0.5], [0.0, 0.0, 0.0]]))
    assert gold_rank(m.getrow(0), 1) == 1
    assert gold_rank(m.getrow(0), 2) == 2
    assert gold_rank(m.getrow(0), 0) == 3
    assert gold_rank(m.getrow(1), 0) == 0, "empty row must report not-found"
    b = string_breakdown(["Aspirin [SEP] q", "aspirin [SEP] r", "tumor [SEP] s", "new [SEP] t"],
                         ["A", "B", "T", "N"], ["B", "B", "X", "N"],
                         [("aspirin [SEP] a", "A"), ("aspirin [SEP] b", "A"), ("aspirin [SEP] c", "B"),
                          ("tumor [SEP] d", "T")])
    assert b["seen, >1 label"] == (2, 0.5, 0.5, 1.0) and b["seen, 1 label"] == (1, 0.0, 1.0, 1.0)
    assert b["unseen string"] == (1, 1.0, 0.0, 1.0) and b["hybrid"] == 0.75
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
