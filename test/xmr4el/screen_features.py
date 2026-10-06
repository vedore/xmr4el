"""Screen mention featurizations by flat retrieval, no tree training.

Same label vocabulary as a saved tree, all in-vocabulary dev rows (identical to the eval rows).
For each feature variant two label scorers:
  centroid  cosine to PIFA z (normalised sum of the label's train rows), what the tree uses
  max       max cosine over the label's train rows (1-NN)
Reports acc@1, MRR and the mention-string breakdown; prints a sample of unseen-string errors.
"""
import sys
import tempfile
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
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.models.featurization_wrapper.transformers import sentence_model
from xmr4el.xmr.model import XModel


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
    "sbiobert": "pritamdeka/S-BioBert-snli-multinli-stsb",  # old flag-4/5 mention encoder
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}


def encode_mentions(model_name, texts):
    """Raw (unnormalised) embeddings on CPU with the tree's own model loader. Not via
    Transformer._predict: that clears ./batch_dir, which a concurrent training run may be using."""
    return sentence_model(model_name, "cpu").encode(
        texts, batch_size=256, normalize_embeddings=False, show_progress_bar=True).astype(np.float32)


def context_window(text, span, w):
    """Window around the mention; text = "mention [SEP] document" as the loader builds it."""
    return Preprocessor.context_window(text.split("[SEP]", 1)[1][1:], span, w)


TFIDF = {  # name -> (text field, TfidfVectorizer kwargs)
    "char": ("m", dict(analyzer="char_wb", ngram_range=(2, 4))),
    "word": ("m", {}),
    "ctxwin": ("win", dict(stop_words="english")),
    "ctxdoc": ("doc", dict(stop_words="english")),
}


def featurize(name, tr, te):
    """Train/test features. tr/te: {"m": mentions, "win": context windows, "doc": documents}.
    TF-IDF names take an "svd" suffix (-> SVD 768: tree training densifies X, so the tree needs dense
    blocks). Encoder names embed the mention; a "win" prefix embeds the context window instead."""
    base = name[:-3] if name.endswith("svd") else name
    if base in TFIDF:
        field, kw = TFIDF[base]
        vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, **kw).fit(tr[field])
        A, B = vec.transform(tr[field]), vec.transform(te[field])
        if name != base:
            svd = TruncatedSVD(n_components=min(768, A.shape[1] - 1), random_state=42).fit(A)
            A, B = svd.transform(A), svd.transform(B)
        return A, B
    field, enc = ("win", name[3:]) if name.startswith("win") else ("m", name)
    if enc in ENCODERS:
        u, inv = np.unique(list(tr[field]) + list(te[field]), return_inverse=True)
        E = encode_mentions(ENCODERS[enc], list(u))[inv]
        return E[:len(tr[field])], E[len(tr[field]):]
    raise ValueError(name)


def combine(parts):
    """Each block L2-normalised and scaled by its weight, then concatenated."""
    if len(parts) == 1 and parts[0][2] == 1.0:
        return parts[0][0], parts[0][1]
    return tuple(hstack([csr_matrix(w * normalize(p[i])) for *p, w in parts], format="csr")
                 for i in (0, 1))


def expand_mentions(mentions, documents, mode):
    """mode none | replace (mention -> long form) | append ("SSI surgical site infection").
    Returns (new mentions, bool mask of rows whose mention is a short form defined in its document)."""
    cache, new, hit = {}, [], []
    for m, doc in zip(mentions, documents):
        if doc not in cache:
            cache[doc] = Preprocessor.abbreviations(doc)
        lf = cache[doc].get(m.strip())
        hit.append(lf is not None)
        new.append(m if lf is None or mode == "none" else lf if mode == "replace" else f"{m.strip()} {lf}")
    return new, np.array(hit)


def show_errors(pred, gold, te_m, unseen, labels, names, n, rng):
    idx = np.flatnonzero(unseen & (pred != gold))
    print(f"\n  sample of {min(n, len(idx))}/{len(idx)} unseen-string errors "
          f"(label shown by its most frequent train mention):")
    for i in rng.permutation(idx)[:n]:
        print(f"    {te_m[i]!r:40.40}  gold {labels[gold[i]]} {names[gold[i]]!r:30.30}"
              f"  pred {labels[pred[i]]} {names[pred[i]]!r}")


def main():

    ap = ArgumentParser()
    ap.add_argument("-xmodel_path", required=True, help="saved tree: label vocabulary + transformer config")
    ap.add_argument("-train_path", required=True)
    ap.add_argument("-test_path", required=True)
    ap.add_argument("-features", default=",".join(
        ["sapbert+charsvd"] + [f"sapbert+charsvd+ctxwin*{w}" for w in (0.3, 0.5, 1)]
        + ["sapbert+charsvd+ctxdoc*0.5", "sapbert+charsvd+winsbiobert*0.5",
           "sapbert+charsvd+ctxwinsvd*0.5"]),
                    help="comma list; a+b = per-block L2 normalised concat; a*w scales block a by w")
    ap.add_argument("-scorers", default="centroid", help="comma list of centroid,max")
    ap.add_argument("-window", type=int, default=10, help="context words each side of the mention")
    ap.add_argument("-show_errors", type=int, default=30)
    ap.add_argument("-abbrev", default="none",
                    help="comma list of none,replace,append: expand mentions that are a short form defined "
                         "as 'long form (SF)' in their document, in train and dev")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    xm = XModel.load(args.xmodel_path)
    labels = list(xm.initial_labels)
    l2i = {lab: i for i, lab in enumerate(labels)}

    def load(path):
        d = Preprocessor.load_pubtator_file(path)
        keep = [i for i, y in enumerate(d["labels"]) if y in l2i]
        pairs = [(d["corpus"][i], d["labels"][i]) for i in keep]
        fields = {"m": [t.split("[SEP]", 1)[0] for t, _ in pairs],  # what the transformer block sees
                  "win": [context_window(d["corpus"][i], d["spans"][i], args.window) for i in keep],
                  "doc": [t.split("[SEP]", 1)[1] for t, _ in pairs]}
        for mode in args.abbrev.split(","):
            fields["m:" + mode], fields["abbr"] = expand_mentions(
                fields["m"], [doc[1:] for doc in fields["doc"]], mode)
        return pairs, fields, len(d["labels"])

    train_pairs, tr, _ = load(args.train_path)
    test_pairs, te, n_test_all = load(args.test_path)
    print(f"labels {len(labels)}  train rows {len(train_pairs)}  dev rows {len(test_pairs)}/{n_test_all}")
    te_m = te["m"]  # raw mentions; the abbrev loop rebinds te["m"]
    y_tr = np.array([l2i[y] for _, y in train_pairs])
    gold = np.array([l2i[y] for _, y in test_pairs])

    seen = Counter(_mention(t) for t, _ in train_pairs)
    unseen = np.array([_mention(t) not in seen for t, _ in test_pairs])
    by_label = defaultdict(Counter)
    for t, y in train_pairs:
        by_label[l2i[y]][_mention(t)] += 1
    names = [by_label[j].most_common(1)[0][0] for j in range(len(labels))]

    def field_of(name):
        base = name[:-3] if name.endswith("svd") else name
        return TFIDF[base][0] if base in TFIDF else "win" if name.startswith("win") else "m"

    cache = {}
    for mode in args.abbrev.split(","):
        tr["m"], te["m"] = tr["m:" + mode], te["m:" + mode]
        abbr = te["abbr"]
        print(f"\n=== abbrev {mode}: short-form mentions defined in their document: train "
              f"{tr['abbr'].sum()}/{len(tr['abbr'])}, dev {abbr.sum()}/{len(abbr)} "
              f"(unseen strings {(abbr & unseen).sum()}/{unseen.sum()})")
        if mode != "none" and args.show_errors:
            for i in rng.permutation(np.flatnonzero(abbr))[:args.show_errors]:
                print(f"    {te_m[i].strip()!r:14.14} -> {te['m'][i]!r:50.50}  gold {names[gold[i]]!r}")
        for spec in args.features.split(","):
            parts = []
            for item in spec.split("+"):
                name, _, w = item.partition("*")
                key = (mode if field_of(name) == "m" else "", name)
                if key not in cache:
                    cache[key] = featurize(name, tr, te)
                parts.append((*cache[key], float(w or 1)))
            X_tr, X_te = combine(parts)
            for scorer in args.scorers.split(","):
                S = label_scores(X_tr, y_tr, X_te, len(labels), scorer)
                acc, mrr, ties = acc_mrr(S, gold)
                pred = top1(S, rng)
                br = string_breakdown([t for t, _ in test_pairs], [labels[g] for g in gold],
                                      [labels[p] for p in pred], train_pairs)
                print(f"\n[{spec} / {scorer} / abbrev {mode}] acc@1 {acc:.4f}  MRR {mrr:.4f}  hybrid {br['hybrid']:.4f}"
                      f"  (gold tied at top {ties:.3f}; breakdown = one random tie-break; groups by raw string)")
                for k, (n, a, _, _) in ((k, v) for k, v in br.items() if k != "hybrid"):
                    print(f"  {k:<15} n {n:5d}  acc@1 {a:.3f}")
                ok = pred == gold
                print(f"  {'abbrev rows':<15} n {abbr.sum():5d}  acc@1 {ok[abbr].mean():.3f}"
                      f"  (unseen {ok[abbr & unseen].mean():.3f})")
                if mode != "none":
                    bx = string_breakdown(te["m"], [labels[g] for g in gold], [labels[p] for p in pred],
                                          list(zip(tr["m"], [y for _, y in train_pairs])))
                    print(f"  hybrid with expanded dictionary key {bx['hybrid']:.4f}")
                if (scorer == "centroid" and spec == args.features.split(",")[0] and mode == "none"
                        and args.show_errors):
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
    Xc_tr, Xc_te = featurize("char", {"m": ["aspirin", "tumour"]}, {"m": ["Aspirine"]})
    assert label_scores(Xc_tr, np.array([0, 1]), Xc_te, 2, "max").argmax() == 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1|t|Aspirin works\n1|a|It lowers fever.\n"
                "1\t0\t7\tAspirin\tT\tC1\n1\t17\t23\tlowers\tT\tC2\n")
    d = Preprocessor.load_pubtator_file(f.name)
    for t, (s, e) in zip(d["corpus"], d["spans"]):
        assert t.split("[SEP]", 1)[1][1:][s:e] == t.split(" [SEP]", 1)[0], (t, s, e)
    assert context_window(d["corpus"][1], d["spans"][1], 2) == "works It fever."
    text = "Ki-67 [SEP] Cells with high Ki-67 staining grew fast"
    s = text.split("[SEP]", 1)[1][1:].index("Ki-67")
    assert context_window(text, (s, s + 5), 2) == "with high staining grew"
    A, B = combine([(np.array([[3.0, 4.0]]), np.array([[1.0, 0.0]]), 1.0),
                    (np.array([[0.0, 2.0]]), np.array([[5.0, 0.0]]), 0.5)])
    assert np.allclose(A.toarray(), [[0.6, 0.8, 0.0, 0.5]]) and np.allclose(B.toarray(), [[1, 0, 0.5, 0]])
    best_long_form, abbreviations = Preprocessor.best_long_form, Preprocessor.abbreviations
    assert best_long_form("SSI", "after surgery a surgical site infection") == "surgical site infection"
    assert best_long_form("HIV", "the human immunodeficiency virus") == "human immunodeficiency virus"
    assert best_long_form("XYZ", "no match here") is None
    doc = ("Surgical site infection (SSI) is common. Tumor necrosis factor (TNF, a cytokine) rose. "
           "Patients (n = 12) and controls (p < 0.05). SSI rates fell.")
    assert abbreviations(doc) == {"SSI": "Surgical site infection", "TNF": "Tumor necrosis factor"}
    new, hit = expand_mentions(["SSI ", "SSI", "rates"], [doc, "no definition", doc], "append")
    assert new == ["SSI Surgical site infection", "SSI", "rates"] and hit.tolist() == [True, False, False]
    print("selfcheck ok")


if __name__ == "__main__":
    if "-selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
