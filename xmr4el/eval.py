"""Evaluation metrics shared by the evaluate and diagnostic scripts."""
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix


def mention_key(text):
    """Normalised mention string (text before [SEP]) used by the dictionary baselines."""
    return text.split("[SEP]", 1)[0].strip().lower()


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


def ranking_metrics(ranks, ks=(5, 10, 20)):
    """Aggregate existing ranks without changing callers' tie ordering."""
    ranks = np.asarray(ranks)
    found = ranks > 0
    metrics = {"acc@1": np.mean(ranks == 1),
               "MRR": np.mean(np.where(found, 1.0 / np.maximum(ranks, 1), 0.0))}
    metrics.update({f"R@{k}": np.mean(found & (ranks <= k)) for k in ks})
    return metrics


def split_by_ranker(gold_idx, trained):
    has = np.isin(gold_idx, list(trained))
    return (("with ranker", has), ("without ranker", ~has))


def string_breakdown(test_texts, gold, tree_top1, train_pairs):
    """acc@1 of the tree and of the mention dictionary, split by whether the exact mention string
    occurs in train (and with how many labels). Returns {group: (n, tree, dict, either)} plus the
    hybrid (dictionary if the string was seen, tree otherwise)."""
    counts = defaultdict(Counter)
    for t, y in train_pairs:
        counts[mention_key(t)][y] += 1
    m = [mention_key(t) for t in test_texts]
    n_lab = np.array([len(counts[s]) if s in counts else 0 for s in m])
    dict_ok = np.array([s in counts and counts[s].most_common(1)[0][0] == g for s, g in zip(m, gold)])
    tree_ok = np.array([p == g for p, g in zip(tree_top1, gold)])
    groups = {"all": n_lab >= 0, "seen, 1 label": n_lab == 1,
              "seen, >1 label": n_lab > 1, "unseen string": n_lab == 0}
    out = {k: (int(g.sum()), tree_ok[g].mean() if g.any() else 0.0,
               dict_ok[g].mean() if g.any() else 0.0, (tree_ok | dict_ok)[g].mean() if g.any() else 0.0)
           for k, g in groups.items()}
    out["hybrid"] = np.where(n_lab > 0, dict_ok, tree_ok).mean()
    return out


def _selfcheck():
    m = csr_matrix(np.array([[0.1, 0.9, 0.5], [0.0, 0.0, 0.0]]))
    assert gold_rank(m.getrow(0), 1) == 1
    assert gold_rank(m.getrow(0), 2) == 2
    assert gold_rank(m.getrow(0), 0) == 3
    assert gold_rank(m.getrow(1), 0) == 0, "empty row must report not-found"
    (_, has), (_, no) = split_by_ranker(np.array([0, 2, 2, 5]), {2})
    assert has.tolist() == [False, True, True, False] and (has ^ no).all()
    b = string_breakdown(["Aspirin [SEP] q", "aspirin [SEP] r", "tumor [SEP] s", "new [SEP] t"],
                         ["A", "B", "T", "N"], ["B", "B", "X", "N"],
                         [("aspirin [SEP] a", "A"), ("aspirin [SEP] b", "A"), ("aspirin [SEP] c", "B"),
                          ("tumor [SEP] d", "T")])
    assert b["seen, >1 label"] == (2, 0.5, 0.5, 1.0) and b["seen, 1 label"] == (1, 0.0, 1.0, 1.0)
    assert b["unseen string"] == (1, 1.0, 0.0, 1.0) and b["hybrid"] == 0.75
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
