"""Screen mention featurizations by flat retrieval, no tree training.

Same label vocabulary as a saved tree, all in-vocabulary dev rows (identical to the eval rows).
For each feature variant two label scorers:
  centroid  cosine to PIFA z (normalised sum of the label's train rows), what the tree uses
  max       max cosine over the label's train rows (1-NN)
Reports acc@1, MRR and the mention-string breakdown; prints a sample of unseen-string errors.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from scipy.special import digamma
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from diagnose_routing import _mention
from test_evaluate_pipeline import string_breakdown


def label_scores(X_tr, y_tr, X_te, n_labels, scorer, chunk=1000):
    """(n_test, n_labels) cosine scores. y_tr holds label indices of the train rows."""
    X_tr, X_te = normalize(X_tr), normalize(X_te)
    if scorer == "centroid":
        Y = csr_matrix((np.ones(len(y_tr)), (y_tr, np.arange(len(y_tr)))), shape=(n_labels, len(y_tr)))
        S = X_te @ normalize(Y @ X_tr).T
        return S.toarray() if issparse(S) else np.asarray(S)
    order = np.argsort(y_tr, kind="stable")
    starts = np.searchsorted(y_tr[order], np.arange(n_labels))
    assert np.all(np.diff(np.append(starts, len(y_tr))) > 0), "every label needs a train row"
    X_tr = X_tr[order]
    out = np.empty((X_te.shape[0], n_labels), dtype=np.float32)
    for i in range(0, X_te.shape[0], chunk):
        S = X_te[i:i + chunk] @ X_tr.T
        S = S.toarray() if issparse(S) else np.asarray(S)
        out[i:i + chunk] = np.maximum.reduceat(S, starts, axis=1)
    return out


def acc_mrr(S, gold):
    """Expected acc@1 and MRR when ties are broken uniformly at random, plus the top-1 tie rate."""
    gold_s = S[np.arange(len(gold)), gold][:, None]
    g = (S > gold_s).sum(axis=1)
    e = (S == gold_s).sum(axis=1)  # includes gold itself
    H = lambda n: digamma(n + 1) + np.euler_gamma
    return np.mean((g == 0) / e), np.mean((H(g + e) - H(g)) / e), np.mean((g == 0) & (e > 1))


def top1(S, rng):
    """argmax with uniform random tie-breaking (np.argmax alone favours the lowest label index)."""
    S = np.asarray(S, dtype=np.float64)
    return (S + rng.random(S.shape) * 1e-12).argmax(axis=1)


ENCODERS = {
    "sbiobert": "pritamdeka/S-BioBert-snli-multinli-stsb",  # flag 4/5 mention block
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}


def encode_mentions(model_name, texts):
    """Raw (unnormalised) embeddings on CPU with the tree's own model loader. Not via
    Transformer._predict: that clears ./batch_dir, which a concurrent training run may be using."""
    from xmr4el.models.featurization_wrapper.transformers import sentence_model
    return sentence_model(model_name, "cpu").encode(
        texts, batch_size=256, normalize_embeddings=False, show_progress_bar=True).astype(np.float32)


def featurize(name, tr_m, te_m):
    """Mention-only features for train/test mention strings."""
    if name in ("char", "word", "charsvd"):
        kw = {} if name == "word" else dict(analyzer="char_wb", ngram_range=(2, 4))
        vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, **kw).fit(tr_m)
        A, B = vec.transform(tr_m), vec.transform(te_m)
        if name == "charsvd":  # tree training densifies X (base.py), so the tree needs a dense block
            svd = TruncatedSVD(n_components=min(768, A.shape[1] - 1), random_state=42).fit(A)
            A, B = svd.transform(A), svd.transform(B)
        return A, B
    if name in ENCODERS:
        enc = encode_mentions(ENCODERS[name], list(tr_m) + list(te_m))
        return enc[:len(tr_m)], enc[len(tr_m):]
    raise ValueError(name)


def show_errors(pred, gold, te_m, unseen, labels, names, n, rng):
    idx = np.flatnonzero(unseen & (pred != gold))
    print(f"\n  sample of {min(n, len(idx))}/{len(idx)} unseen-string errors "
          f"(label shown by its most frequent train mention):")
    for i in rng.permutation(idx)[:n]:
        print(f"    {te_m[i]!r:40.40}  gold {labels[gold[i]]} {names[gold[i]]!r:30.30}"
              f"  pred {labels[pred[i]]} {names[pred[i]]!r}")


def main():
    from xmr4el.featurization.preprocessor import Preprocessor
    from xmr4el.xmr.model import XModel

    ap = ArgumentParser()
    ap.add_argument("-xmodel_path", required=True, help="saved tree: label vocabulary + transformer config")
    ap.add_argument("-train_path", required=True)
    ap.add_argument("-test_path", required=True)
    ap.add_argument("-features", default="sbiobert,sapbert,sbiobert+charsvd,sapbert+charsvd,sapbert+char",
                    help="comma list; a+b = per-block L2 normalised, equal-weight concat")
    ap.add_argument("-show_errors", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    xm = XModel.load(args.xmodel_path)
    labels = list(xm.initial_labels)
    l2i = {lab: i for i, lab in enumerate(labels)}

    def load(path):
        d = Preprocessor.load_pubtator_file(path)
        pairs = [(t, y) for t, y in zip(d["corpus"], d["labels"]) if y in l2i]
        return pairs, len(d["labels"])

    train_pairs, _ = load(args.train_path)
    test_pairs, n_test_all = load(args.test_path)
    print(f"labels {len(labels)}  train rows {len(train_pairs)}  dev rows {len(test_pairs)}/{n_test_all}")
    # Raw text before [SEP], exactly what the flag-4 transformer block sees
    tr_m = [t.split("[SEP]", 1)[0] for t, _ in train_pairs]
    te_m = [t.split("[SEP]", 1)[0] for t, _ in test_pairs]
    y_tr = np.array([l2i[y] for _, y in train_pairs])
    gold = np.array([l2i[y] for _, y in test_pairs])

    seen = Counter(_mention(t) for t, _ in train_pairs)
    unseen = np.array([_mention(t) not in seen for t, _ in test_pairs])
    by_label = defaultdict(Counter)
    for t, y in train_pairs:
        by_label[l2i[y]][_mention(t)] += 1
    names = [by_label[j].most_common(1)[0][0] for j in range(len(labels))]

    cache = {}
    for spec in args.features.split(","):
        parts = []
        for name in spec.split("+"):
            if name not in cache:
                cache[name] = featurize(name, tr_m, te_m)
            parts.append(cache[name])
        if len(parts) == 1:
            X_tr, X_te = parts[0]
        else:
            X_tr = hstack([csr_matrix(normalize(a)) for a, _ in parts], format="csr")
            X_te = hstack([csr_matrix(normalize(b)) for _, b in parts], format="csr")
        for scorer in ("centroid", "max"):
            S = label_scores(X_tr, y_tr, X_te, len(labels), scorer)
            acc, mrr, ties = acc_mrr(S, gold)
            pred = top1(S, rng)
            br = string_breakdown([t for t, _ in test_pairs], [labels[g] for g in gold],
                                  [labels[p] for p in pred], train_pairs)
            print(f"\n[{spec} / {scorer}] acc@1 {acc:.4f}  MRR {mrr:.4f}  hybrid {br['hybrid']:.4f}"
                  f"  (gold tied at top {ties:.3f}; breakdown = one random tie-break)")
            for k, (n, a, _, _) in ((k, v) for k, v in br.items() if k != "hybrid"):
                print(f"  {k:<15} n {n:5d}  acc@1 {a:.3f}")
            if scorer == "centroid" and spec == args.features.split(",")[0] and args.show_errors:
                show_errors(pred, gold, te_m, unseen, labels, names, args.show_errors, rng)


def _selfcheck():
    X_tr = np.array([[1.0, 0], [0.6, 0.8], [0, 1.0]])
    y_tr = np.array([0, 0, 1])
    X_te = np.array([[0.5, 0.866], [0.1, 1.0]])
    S_max = label_scores(X_tr, y_tr, X_te, 2, "max", chunk=1)
    assert np.allclose(S_max[0], [0.3 + 0.8 * 0.866, 0.866] / np.hypot(0.5, 0.866), atol=1e-6)
    S_cen = label_scores(X_tr, y_tr, X_te, 2, "centroid")
    assert S_cen.argmax(axis=1).tolist() == [1, 1] and S_max.argmax(axis=1).tolist() == [0, 1]
    S_sp = label_scores(csr_matrix(X_tr), y_tr, csr_matrix(X_te), 2, "max")
    assert np.allclose(S_sp, S_max)
    acc, mrr, _ = acc_mrr(np.array([[0.9, 0.1], [0.9, 0.1]]), np.array([0, 1]))
    assert acc == 0.5 and mrr == 0.75
    # gold tied with one other label at the top: expected acc 1/2, expected RR (1 + 1/2) / 2
    acc, mrr, ties = acc_mrr(np.array([[0.0, 0.0, -1.0]]), np.array([0]))
    assert np.isclose(acc, 0.5) and np.isclose(mrr, 0.75) and ties == 1.0
    picks = top1(np.zeros((2000, 2), dtype=np.float32), np.random.default_rng(0))
    assert 0.4 < picks.mean() < 0.6, "ties must not always go to label 0"
    Xc_tr, Xc_te = featurize("char", ["aspirin", "tumour"], ["Aspirine"])
    assert label_scores(Xc_tr, np.array([0, 1]), Xc_te, 2, "max").argmax() == 0
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "-selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
