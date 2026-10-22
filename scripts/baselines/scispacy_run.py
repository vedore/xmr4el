# LOCAL TEST ARTIFACT (user's baseline comparison): kept on `repeat`, not merged into `main`.
"""scispaCy's entity-linker candidate generator on the gold mentions of a `pecos_compare.py export` (plan G, 5).

Runs in its own container (`scispacy.dockerfile`). Each dev row's mention string (text before [SEP], as the 1-NN
baselines; abbreviation expansion as the tree's loader) goes to `CandidateGenerator(name=<kb>)`: char 3-gram TF-IDF
nearest neighbours over the KB's aliases, a concept's score = its max alias similarity (`EntityLinker` without the
0.7 threshold, so every candidate is ranked). Concepts outside the tree's label set are dropped: the same candidate
space as every other system. Labels match KB ids after their "UMLS:" prefix (MedMentions). Writes
<data>/pred_scispacy_<kb>.npz (n_dev x n_labels), scored by `pecos_compare.py score`.
"""
import json
import os
import time
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from scispacy.candidate_generation import CandidateGenerator


def to_matrix(cands, l2i, n_labels):
    """Per row: concept -> max alias similarity, kept for concepts in l2i. Returns the matrix and, per row, whether
    the unrestricted top-1 concept was outside the label set."""
    rows, cols, vals, outside = [], [], [], []
    for i, row in enumerate(cands):
        best = {c.concept_id: max(c.similarities) for c in row}
        top = max(best, key=best.get) if best else None
        outside.append(top is not None and top not in l2i)
        for cid, s in best.items():
            if cid in l2i:
                rows.append(i), cols.append(l2i[cid]), vals.append(s)
    P = csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(cands), n_labels))
    return P, np.array(outside)


def main():
    ap = ArgumentParser()
    ap.add_argument("-data", required=True, help="pecos_compare.py export dir (texts.json)")
    ap.add_argument("-kb", default="umls", help="scispaCy KB; umls = CUIs (MedMentions)")
    ap.add_argument("-k", type=int, default=100, help="alias neighbours per mention (EntityLinker default 30)")
    ap.add_argument("-batch", type=int, default=2048)
    args = ap.parse_args()
    with open(os.path.join(args.data, "texts.json")) as f:
        t = json.load(f)
    l2i = {lab.split(":", 1)[-1]: i for i, lab in enumerate(t["labels"])}
    assert len(l2i) == len(t["labels"]), "labels collide after prefix strip"
    mentions = [x.split("[SEP]", 1)[0].strip() for x in t["dev_texts"]]

    t0 = time.time()
    cg = CandidateGenerator(name=args.kb)
    t1 = time.time()
    cands = [c for s in range(0, len(mentions), args.batch) for c in cg(mentions[s:s + args.batch], args.k)]
    P, outside = to_matrix(cands, l2i, len(t["labels"]))
    save_npz(os.path.join(args.data, f"pred_scispacy_{args.kb}.npz"), P)
    print(f"[scispacy_{args.kb}] {len(mentions)} rows, load {t1 - t0:.1f}s, link {time.time() - t1:.1f}s, "
          f"{P.nnz / len(mentions):.1f} in-vocab candidates/row, rows with no in-vocab candidate "
          f"{(P.getnnz(axis=1) == 0).mean():.4f}, unrestricted top-1 outside the label set {outside.mean():.4f}")


if __name__ == "__main__":
    main()
