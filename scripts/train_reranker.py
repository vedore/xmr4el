"""Train the cross-encoder reranker on a saved tree's top-K candidates (STATUS.md § R).

Rows = the PubTator mentions of -train_path whose gold is a tree label; negatives = the tree's top K on the row
(its predict_config) minus gold, cut to K-1. With -folds N the top K comes instead from N fold trees (the saved
tree's config, trained on -tree_train_path minus the fold's documents), so no row is ranked by a tree that saw it.
Saved to outputs/rerankers/<tree>_<timestamp>/."""
from argparse import ArgumentParser
import logging
import os
import sys
import time
from datetime import datetime

if "-verbose" not in sys.argv:
    os.environ.update(TRANSFORMERS_VERBOSITY="warning", HF_HUB_VERBOSITY="warning",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")

import numpy as np

from xmr4el import set_verbosity
from xmr4el.data.readers import Preprocessor
from xmr4el.features.transformers import MODEL_NAMES
from xmr4el.rerank import CrossEncoderReranker, label_texts, out_of_fold_top_k, top_k
from xmr4el.xmodel import XModel

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    parser = ArgumentParser()
    parser.add_argument("-xmodel_path", type=str, required=True)
    parser.add_argument("-train_path", type=str, required=True,
                        help="PubTator rows with context (BC5CDR train, not train_plus_ctd)")
    parser.add_argument("-k", type=int, default=10, help="list size: gold + K-1 tree negatives")
    parser.add_argument("-folds", type=int, default=0,
                        help="N >= 2: negatives from out-of-fold trees (by document); 0 = the saved tree (in-sample)")
    parser.add_argument("-tree_train_path", type=str, default=None,
                        help="with -folds: the saved tree's training PubTator (e.g. train_plus_ctd)")
    parser.add_argument("-epochs", type=int, default=2)
    parser.add_argument("-lr", type=float, default=2e-5)
    parser.add_argument("-rows_per_batch", type=int, default=4)
    parser.add_argument("-max_length", type=int, default=96)
    parser.add_argument("-init", type=str, default=MODEL_NAMES["sapbert"])
    parser.add_argument("-verbose", action="store_true")
    args = parser.parse_args()
    if args.k < 2 or args.epochs < 1 or args.rows_per_batch < 1:
        parser.error("need -k >= 2 (K = 1 gives zero loss), -epochs >= 1, -rows_per_batch >= 1")
    if args.folds == 1 or args.folds < 0 or bool(args.folds) != bool(args.tree_train_path):
        parser.error("-folds N >= 2 and -tree_train_path go together")
    set_verbosity(2 if args.verbose else 1)
    start = time.perf_counter()

    xm = XModel.load(args.xmodel_path)
    tree = os.path.basename(os.path.normpath(args.xmodel_path))
    data = Preprocessor.load_pubtator_file(args.train_path, window=xm.context_window, abbrev=xm.abbrev_expansion)
    label_to_idx = {lab: i for i, lab in enumerate(xm.initial_labels)}
    rows = [i for i, y in enumerate(data["labels"]) if y in label_to_idx]
    golds, queries = [data["labels"][i] for i in rows], [data["corpus"][i] for i in rows]

    if args.folds:
        tree_data = Preprocessor.load_pubtator_file(args.tree_train_path, window=xm.context_window,
                                                    abbrev=xm.abbrev_expansion)
        idx = out_of_fold_top_k(xm, queries, [data["docs"][i] for i in rows], tree_data, args.folds, args.k)
    else:
        idx, _ = top_k(xm.predict(queries), args.k)
    candidates, kept = [], []
    for i, (g, top) in enumerate(zip(golds, idx)):
        negs = top[top != label_to_idx[g]][:args.k - 1]
        if len(negs) == args.k - 1:  # ponytail: rows with < K-1 tree candidates are dropped, no knn top-up
            candidates.append([label_to_idx[g], *negs.tolist()])
            kept.append(i)
    gold_in_top = np.mean([label_to_idx[g] in top for g, top in zip(golds, idx)])
    print(f"rows {len(kept)}/{len(data['labels'])} (gold in vocabulary, >= K-1 negatives); "
          f"gold in tree top {args.k}: {gold_in_top:.4f}", flush=True)
    if not kept:
        raise SystemExit(f"no row has >= {args.k - 1} tree negatives: lower -k")

    reranker = CrossEncoderReranker.from_pretrained(
        args.init, {"tree": tree, "k": args.k, "train_path": args.train_path, "rows": len(kept), "folds": args.folds,
                    "tree_train_path": args.tree_train_path,
                    "epochs": args.epochs, "lr": args.lr, "init": args.init}, max_length=args.max_length)
    reranker.fit([queries[i] for i in kept], candidates, label_texts(xm.training_set, xm.Y),
                 epochs=args.epochs, rows_per_batch=args.rows_per_batch, lr=args.lr)

    save_dir = os.path.join("outputs", "rerankers", f"{tree}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    reranker.save(save_dir)
    print(f"saved {save_dir} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
