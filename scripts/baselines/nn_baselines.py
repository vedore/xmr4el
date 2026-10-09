"""Nearest-neighbour baselines on a saved tree's own training rows and evaluate_tree's eval rows (plan G, 3-4).

sapbert  each label scored by the max cosine of the query's transformer (mention) block to that label's training rows:
         SapBERT-style 1-NN linking without the tree, the matcher or the char/context blocks
tfidf    the same with raw char n-gram TF-IDF of the mention string (the tree's vectorizer_config, no SVD): the
         fuzzy-string-match baseline
Same loader settings (context window, abbreviations), vocabulary filter and metric keys as `evaluate_tree`; ties go to
the lower label index. -train_path adds the mention-string breakdown and hybrid. Writes <out>.json when -out is given.
"""
import json
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from xmr4el.data.readers import Preprocessor
from xmr4el.eval import RECALL_KS, filter_labels_and_inputs, ranking_metrics, string_breakdown
from xmr4el.xmodel import XModel


def mention(text):
    """The encoder's mention string (text before [SEP], as `TextEncoder._encode` splits it)."""
    return text.split("[SEP]", 1)[0]


def nn_ranks(Q, B, b_labels, gold, n_labels, chunk=128):
    """1-based rank of each gold label when every label is scored by the max cosine of the query row to its B rows,
    and the top-1 label per query. Q, B: row-normalised (dense or sparse); every label needs at least one B row."""
    order = np.argsort(b_labels, kind="stable")
    labels, starts = np.unique(b_labels[order], return_index=True)
    assert len(labels) == n_labels, f"{n_labels - len(labels)} labels have no training row"
    B = B[order]
    ranks, top1 = np.empty(len(gold), dtype=int), np.empty(len(gold), dtype=int)
    for s in range(0, Q.shape[0], chunk):
        sims = Q[s:s + chunk] @ B.T
        sims = sims.toarray() if issparse(sims) else np.asarray(sims)
        scores = np.maximum.reduceat(sims, starts, axis=1)
        g = np.asarray(gold[s:s + chunk])
        sg = scores[np.arange(len(g)), g][:, None]
        before = np.arange(n_labels)[None, :] < g[:, None]
        ranks[s:s + chunk] = 1 + (scores > sg).sum(1) + ((scores == sg) & before).sum(1)
        top1[s:s + chunk] = scores.argmax(1)
    return ranks, top1


def main():
    parser = ArgumentParser()
    parser.add_argument("-xmodel_path", required=True, help="saved tree: training rows, label set, loader settings")
    parser.add_argument("-test_path", required=True)
    parser.add_argument("-train_path", help="PubTator train file of the tree: mention-string breakdown + hybrid")
    parser.add_argument("-baselines", nargs="+", default=["sapbert", "tfidf"], choices=["sapbert", "tfidf"])
    parser.add_argument("-out", help="write {baseline: metrics} to this JSON file")
    args = parser.parse_args()

    xm = XModel.load(args.xmodel_path)
    labels = np.array(xm.initial_labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    test = Preprocessor.load_pubtator_file(args.test_path, window=xm.context_window, abbrev=xm.abbrev_expansion)
    gold, texts = filter_labels_and_inputs(test["corpus"], test["labels"], xm.initial_labels)
    assert gold, "no eval row has a gold label in the tree's vocabulary"
    gold_idx = np.array([label_to_idx[g] for g in gold])
    Y = xm.Y.tocsr()
    assert (Y.getnnz(axis=1) == 1).all(), "one label per training row"
    train_texts = [t for group in xm.training_set for t in group]  # row i of X / Y (Preprocessor.prepare_data)
    assert len(train_texts) == Y.shape[0]
    train_pairs = None
    if args.train_path:
        train = Preprocessor.load_pubtator_file(args.train_path, abbrev=xm.abbrev_expansion)
        train_pairs = [(t, y) for t, y in zip(train["corpus"], train["labels"]) if y in label_to_idx]

    out = {}
    for name in args.baselines:
        if name == "sapbert":
            block = xm.mention_block()
            Q = normalize(xm.text_encoder.predict(texts)[:, block].toarray().astype(np.float32))
            B = normalize(xm.X[:, block].toarray().astype(np.float32))
        else:
            vec = TfidfVectorizer(**{k: tuple(v) if isinstance(v, list) else v
                                     for k, v in xm.vectorizer_config["kwargs"].items()}, dtype=np.float32)
            B = normalize(vec.fit_transform([mention(t) for t in train_texts]))
            Q = normalize(vec.transform([mention(t) for t in texts]))
        ranks, top1 = nn_ranks(Q, B, Y.indices, gold_idx, len(labels))
        m = {"baseline": name, "tree": args.xmodel_path, "test_path": args.test_path, "labels": len(labels),
             "rows": len(gold), "rows_total": len(test["labels"])}
        m.update({k: float(v) for k, v in ranking_metrics(ranks, ks=RECALL_KS).items()})
        m["acc@1_all"] = m["acc@1"] * m["rows"] / m["rows_total"]
        if train_pairs is not None:
            b = string_breakdown(texts, gold, list(labels[top1]), train_pairs)
            m["hybrid"] = float(b.pop("hybrid"))
            m["strings"] = {k: dict(zip(("n", "tree", "dict", "either"), (int(v[0]), *map(float, v[1:]))))
                            for k, v in b.items()}
        out[name] = m
        print(f"{name:8s} rows {m['rows']}/{m['rows_total']}  acc@1 {m['acc@1']:.4f} (all rows {m['acc@1_all']:.4f})"
              f"  MRR {m['MRR']:.4f}  " + "  ".join(f"R@{k} {m[f'R@{k}']:.4f}" for k in RECALL_KS)
              + (f"  hybrid {m['hybrid']:.4f}" if "hybrid" in m else ""))
        if "strings" in m:
            print("         " + "  ".join(f"{k} {g['tree']:.4f} (n={g['n']})" for k, g in m["strings"].items()))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
