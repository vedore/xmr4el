# LOCAL TEST ARTIFACT (user's baseline comparison): kept on `repeat`, not merged into `main`.
"""PECOS XR-Linear on data written by `pecos_compare.py export`. Runs where libpecos installs (Linux / Docker).

-features ours   the tree's own instance features (system comparison: same X, Y, rows)
-features tfidf  PECOS's own TF-IDF on the same "mention [SEP] context" texts (end-to-end baseline)
Hierarchy close to the tree: PIFA label embeddings, hierarchical k-means, nr_splits 8, max leaf 100 -> at 500
labels 8 clusters -> labels (same depth as the tree's 6 -> labels). PECOS treats nr_splits as a power of 2:
6 silently becomes binary splits over 3 levels. Writes <data>/pred_<features>_b<beam>.npz (top-k scores per dev row).
-model <dir>     predict <data>'s rows with a saved model (e.g. the dev export's model_ours) instead of training
-chain           the tree's own hierarchy (<data>/chain_*.npz from the export) instead of PIFA + hierarchical k-means;
                 with -solver L2R_LR_DUAL -post sigmoid and a tree trained with class_weight null, routing_features
                 false, the objective, hierarchy, negatives (rows of the parent cluster) and path product are the tree's
-tag             suffix of model_<features><tag> / pred_<features><tag>_b<beam> (several variants in one export)
"""
import glob
import json
import os
import time
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import load_npz, save_npz

from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.xmc.xlinear.model import XLinearModel
from pecos.utils.featurization.text.preprocess import Preprocessor


def main():
    ap = ArgumentParser()
    ap.add_argument("-data", required=True)
    ap.add_argument("-features", default="ours,tfidf")
    ap.add_argument("-beams", default="2,10", help="2 = the tree's eval beam")
    ap.add_argument("-nr_splits", type=int, default=8, help="power of 2; tree config has n_clusters 6")
    ap.add_argument("-max_leaf_size", type=int, default=100)
    ap.add_argument("-topk", type=int, default=100)
    ap.add_argument("-threshold", type=float, default=0.1,
                    help="XR-Linear weight pruning (PECOS default 0.1); 0 keeps all weights")
    ap.add_argument("-model", help="saved model_ours to predict with (no training; -features ours only)")
    ap.add_argument("-chain", action="store_true", help="use <data>/chain_*.npz (the tree's hierarchy)")
    ap.add_argument("-solver", default="L2R_L2LOSS_SVC_DUAL", help="L2R_LR_DUAL = logistic, the tree's loss")
    ap.add_argument("-post", default="l3-hinge", help="score post-processor; sigmoid = product of sigmoids")
    ap.add_argument("-eps", type=float, default=0.1, help="solver stopping tolerance (PECOS default 0.1)")
    ap.add_argument("-tag", default="")
    args = ap.parse_args()
    pred = dict(only_topk=args.topk, post_processor=args.post)
    if args.model:
        assert args.features == "ours", "-model predicts our features only (the TF-IDF preprocessor is not saved)"
        X_dev = load_npz(os.path.join(args.data, "X_dev.npz")).tocsr().astype(np.float32)
        xlm = XLinearModel.load(args.model)
        for beam in map(int, args.beams.split(",")):
            t0 = time.time()
            P = xlm.predict(X_dev, beam_size=beam, **pred)
            save_npz(os.path.join(args.data, f"pred_ours{args.tag}_b{beam}.npz"), P.tocsr())
            print(f"[ours{args.tag}] model {args.model}: beam {beam}: predict {time.time() - t0:.1f}s")
        return

    Y = load_npz(os.path.join(args.data, "Y_trn.npz")).tocsr().astype(np.float32)
    for feat in args.features.split(","):
        t0 = time.time()
        if feat == "ours":
            X = load_npz(os.path.join(args.data, "X_trn.npz")).tocsr().astype(np.float32)
            X_dev = load_npz(os.path.join(args.data, "X_dev.npz")).tocsr().astype(np.float32)
        else:
            with open(os.path.join(args.data, "texts.json")) as f:
                t = json.load(f)
            prep = Preprocessor.train(t["trn_texts"], {"type": "tfidf", "kwargs": {}})
            X, X_dev = prep.predict(t["trn_texts"]), prep.predict(t["dev_texts"])
        if args.chain:
            chain = [load_npz(p).tocsc().astype(np.float32) for p in
                     sorted(glob.glob(os.path.join(args.data, "chain_*.npz")), key=lambda p: int(p[:-4].split("_")[-1]))]
            assert chain, f"no chain_*.npz in {args.data}: export with the current pecos_compare.py"
        else:
            label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
            chain = Indexer.gen(label_feat, indexer_type="hierarchicalkmeans",
                                nr_splits=args.nr_splits, max_leaf_size=args.max_leaf_size, seed=0)
        t1 = time.time()
        xlm = XLinearModel.train(X, Y, C=chain, shallow=args.chain, threshold=args.threshold,
                                 solver_type=args.solver, eps=args.eps, **pred)
        t_train = time.time() - t0
        print(f"[{feat}{args.tag}] X {X.shape}, cluster sizes {[c.shape for c in chain]}, train {t_train:.1f}s "
              f"(rankers {time.time() - t1:.1f}s), solver {args.solver}, post {args.post}, eps {args.eps}")
        xlm.save(os.path.join(args.data, f"model_{feat}{args.tag}"))
        for beam in map(int, args.beams.split(",")):
            t0 = time.time()
            P = xlm.predict(X_dev, beam_size=beam, **pred)
            save_npz(os.path.join(args.data, f"pred_{feat}{args.tag}_b{beam}.npz"), P.tocsr())
            print(f"  beam {beam}: predict {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
