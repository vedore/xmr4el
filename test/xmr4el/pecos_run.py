"""PECOS XR-Linear on data written by `pecos_compare.py export`. Runs where libpecos installs (Linux / Docker).

-features ours   the tree's own instance features (system comparison: same X, Y, rows)
-features tfidf  PECOS's own TF-IDF on the same "mention [SEP] context" texts (end-to-end baseline)
Hierarchy close to the tree: PIFA label embeddings, hierarchical k-means, nr_splits 8, max leaf 100 -> at 500
labels 8 clusters -> labels (same depth as the tree's 6 -> labels). PECOS treats nr_splits as a power of 2:
6 silently becomes binary splits over 3 levels. Writes <data>/pred_<features>_b<beam>.npz (top-k scores per dev row).
"""
import json
import os
import time
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import load_npz, save_npz

from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.xmc.xlinear.model import XLinearModel


def main():
    ap = ArgumentParser()
    ap.add_argument("-data", required=True)
    ap.add_argument("-features", default="ours,tfidf")
    ap.add_argument("-beams", default="2,10", help="2 = the tree's eval beam")
    ap.add_argument("-nr_splits", type=int, default=8, help="power of 2; tree config has n_clusters 6")
    ap.add_argument("-max_leaf_size", type=int, default=100)
    ap.add_argument("-topk", type=int, default=100)
    args = ap.parse_args()

    Y = load_npz(os.path.join(args.data, "Y_trn.npz")).tocsr().astype(np.float32)
    for feat in args.features.split(","):
        t0 = time.time()
        if feat == "ours":
            X = load_npz(os.path.join(args.data, "X_trn.npz")).tocsr().astype(np.float32)
            X_dev = load_npz(os.path.join(args.data, "X_dev.npz")).tocsr().astype(np.float32)
        else:
            from pecos.utils.featurization.text.preprocess import Preprocessor
            with open(os.path.join(args.data, "texts.json")) as f:
                t = json.load(f)
            prep = Preprocessor.train(t["trn_texts"], {"type": "tfidf", "kwargs": {}})
            X, X_dev = prep.predict(t["trn_texts"]), prep.predict(t["dev_texts"])
        label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
        chain = Indexer.gen(label_feat, indexer_type="hierarchicalkmeans",
                            nr_splits=args.nr_splits, max_leaf_size=args.max_leaf_size, seed=0)
        xlm = XLinearModel.train(X, Y, C=chain)
        t_train = time.time() - t0
        print(f"[{feat}] X {X.shape}, cluster sizes {[c.shape for c in chain]}, train {t_train:.1f}s")
        for beam in map(int, args.beams.split(",")):
            t0 = time.time()
            P = xlm.predict(X_dev, beam_size=beam, only_topk=args.topk)
            save_npz(os.path.join(args.data, f"pred_{feat}_b{beam}.npz"), P.tocsr())
            print(f"  beam {beam}: predict {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
