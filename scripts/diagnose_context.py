"""Context-shift routing diagnostic: does the root matcher send dev mentions to their gold cluster? No training.

Training rows from a dictionary (CTD pseudo-documents: title = the name, no abstract) have an empty context, so their
context block is 0; dev mentions always have one. The root matcher learns the context weights from the few corpus
rows only. Per string group (seen 1 / seen >1 / unseen) this prints the root's gold-cluster recall @1 / @beam with
  as is      the dev features
  ctx 0      the same rows with the context block zeroed (what a dictionary row of that string looks like)
split by whether the gold cluster holds any training row with a context ("ctx cluster") or dictionary rows only.
Then the full search (predict_encoded, the tree's predict_config) on both: acc@1 and R@cand per group.
"sapbert_char_context" trees only (needs the context block). Writes -out JSON.
"""
import json
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import diags

from xmr4el.data.readers import Preprocessor
from xmr4el.eval import filter_labels_and_inputs, gold_rank, train_string_labels
from xmr4el.xmodel import XModel


def recall(ranks, mask, k):
    return float((ranks[mask] <= k).mean()) if mask.any() else None


def main():
    ap = ArgumentParser()
    ap.add_argument("-xmodel_path", required=True)
    ap.add_argument("-test_path", required=True)
    ap.add_argument("-train_path", required=True, help="PubTator file the tree was trained from (string groups)")
    ap.add_argument("-out", required=True)
    args = ap.parse_args()

    xm = XModel.load(args.xmodel_path)
    labels = np.array(xm.initial_labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    test = Preprocessor.load_pubtator_file(args.test_path, window=xm.context_window, abbrev=xm.abbrev_expansion)
    gold, texts = filter_labels_and_inputs(test["corpus"], test["labels"], labels)
    gold_idx = np.array([label_to_idx[g] for g in gold])
    train = Preprocessor.load_pubtator_file(args.train_path, abbrev=xm.abbrev_expansion)
    pairs = [(t, y) for t, y in zip(train["corpus"], train["labels"]) if y in label_to_idx]
    n_lab = np.array([len(c) for c in train_string_labels(texts, pairs)])
    groups = {"all": n_lab >= 0, "seen 1": n_lab == 1, "seen >1": n_lab > 1, "unseen": n_lab == 0}

    ctx = xm.feature_blocks()["context"]
    mask = np.ones(xm.X.shape[1], dtype=np.float32)
    mask[ctx] = 0
    Xq = xm.text_encoder.predict(texts).tocsr()
    Xz = (Xq @ diags(mask)).tocsr()

    # gold cluster of every row at the root (each root label sits in exactly one cluster)
    root = xm.model.hmodel[0][0]
    C = root.cluster_model.c_node.tocsr()
    assert (C.getnnz(axis=1) == 1).all(), "root C must be one cluster per label"
    cluster_of = np.full(len(labels), -1)
    cluster_of[np.asarray(root.local_to_global_idx)] = C.indices
    gold_cluster = cluster_of[gold_idx]
    ctx_rows = np.asarray(abs(xm.X[:, ctx]).sum(axis=1)).ravel() > 0
    ctx_clusters = np.unique(cluster_of[xm.Y.tocsr()[ctx_rows].indices])
    in_ctx = np.isin(gold_cluster, ctx_clusters)
    beam = xm.resolve_predict_config()["beam_size"]

    m = {"tree": args.xmodel_path, "test_path": args.test_path, "rows": len(gold), "beam": beam,
         "train_rows": int(xm.X.shape[0]), "train_rows_with_context": int(ctx_rows.sum()),
         "clusters": int(C.shape[1]), "ctx_clusters": int(ctx_clusters.size),
         "rows_gold_in_ctx_cluster": int(in_ctx.sum()), "root": {}, "search": {}}
    print(f"train rows {m['train_rows']}, with context {m['train_rows_with_context']}; clusters {m['clusters']}, "
          f"with a context row {m['ctx_clusters']}; dev rows {len(gold)}, gold in a ctx cluster {in_ctx.sum()}")

    print(f"\nroot gold-cluster recall @1 / @{beam} (n)")
    print(f"{'group':8} {'cluster':12} {'as is':>14} {'ctx 0':>14}")
    for name, X in (("as is", Xq), ("ctx 0", Xz)):
        P = np.asarray(root.matcher_model.predict_proba(X))
        g = P[np.arange(len(gold)), gold_cluster]
        ranks = (P > g[:, None]).sum(axis=1) + 1
        for gname, gm in groups.items():
            for cname, cm in (("any", gm), ("ctx", gm & in_ctx), ("dict only", gm & ~in_ctx)):
                m["root"].setdefault(name, {}).setdefault(gname, {})[cname] = {
                    "n": int(cm.sum()), "r@1": recall(ranks, cm, 1), f"r@{beam}": recall(ranks, cm, beam)}
    for gname in groups:
        for cname in ("any", "ctx", "dict only"):
            a, z = m["root"]["as is"][gname][cname], m["root"]["ctx 0"][gname][cname]
            fmt = lambda r: "-" if r["r@1"] is None else f"{r['r@1']:.3f} / {r[f'r@{beam}']:.3f}"
            print(f"{gname:8} {cname:12} {fmt(a):>14} {fmt(z):>14}   ({a['n']})")

    print("\nfull search (tree predict_config): acc@1 / R@cand")
    for name, X in (("as is", Xq), ("ctx 0", Xz)):
        S = xm.predict_encoded(X).tocsr()
        ranks = np.array([gold_rank(S.getrow(i), gold_idx[i]) for i in range(len(gold))])
        m["search"][name] = {g: {"n": int(gm.sum()), "acc@1": float((ranks[gm] == 1).mean()),
                                 "R@cand": float((ranks[gm] > 0).mean())} for g, gm in groups.items() if gm.any()}
        print(name, "  ".join(f"{g} {v['acc@1']:.4f} / {v['R@cand']:.4f} ({v['n']})"
                              for g, v in m["search"][name].items()))

    with open(args.out, "w") as f:
        json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
