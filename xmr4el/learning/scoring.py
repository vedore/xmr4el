"""Label scoring, ranker fusion and result packing used by hierarchy nodes."""
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.special import expit
from xmr4el.learning.ranker import RankerTrainer, ranker_input
from numpy import (asarray, array, concatenate, unique, ones, maximum, clip,
                   float32, int32, int64, argsort, argpartition,
                   full, r_, full_like)


def predict_labels(node, X_query, beam_size: int = 5, topk: int | None = None, return_matrix: bool = False, 
                fusion: str = "lp_fusion", eps: float = 1e-6, alpha: float = 0.5, p: int = 3, scorer: str | None = None):
    if scorer not in (None, "ranker", "cosine"):
        raise ValueError(f"Unknown scorer: {scorer}")
    cluster_scores = node.matcher_model.predict_proba(X_query)
    n, K = cluster_scores.shape

    C = node.cluster_model.c_node                                # shape (L, K) label->cluster (CSR)
    L = C.shape[0]
    assert K == C.shape[1], f"Matcher outputs K={K} clusters, but C has {C.shape[1]}."
    assert L == len(node.local_to_global_idx), "C rows must equal #local labels"

    k = min(beam_size, K)
    if k == 0:
        M = csr_matrix((n, 0))
        M.global_labels = np.array([], dtype=object)
        return M if return_matrix else ([np.array([])] * n, [np.array([], dtype=object)] * n)

    idx_part = np.argpartition(cluster_scores, K - k, axis=1)[:, -k:]     # (n, k), unordered
    part = np.take_along_axis(cluster_scores, idx_part, axis=1)
    order = np.argsort(-part, axis=1)
    topk_clusters = np.take_along_axis(idx_part, order, axis=1)           # (n, k)

    # --- 2) Precompute mappings ---
    C_csc = C.tocsc()
    cluster_to_global = [node.local_to_global_idx[C_csc[:, c].indices] for c in range(K)]
    label_cluster = np.asarray(C.argmax(axis=1)).ravel()
    g2l = node.global_to_local_idx
    Z = node.label_embeddings

    POS_COL = 1

    if fusion == "lp_fusion":
        def _fuse(m, r, _a=alpha, _p=p, _eps=eps):
            m = clip(m, _eps, 1.0)
            r = clip(r, _eps, 1.0)
            return ((m ** _p) * (1 - _a) + (r ** _p) * _a) ** (1.0 / _p)
    else: # geometric
        def _fuse(m, r, _a=alpha, _eps=eps):
            m = clip(m, _eps, 1.0)
            r = clip(r, _eps, 1.0)
            return (m ** (1 - _a)) * (r ** _a)

    # Slice helper to appease W8201 for "[:, POS_COL]"
    def _pos_col(a, _col=POS_COL):
        return a[:, _col]

    # --- 3) Build queries_per_label by expanding top-k clusters to all their labels ---
    cand_lists = [
        unique(concatenate([cluster_to_global[c] for c in topk_clusters_q]))
        for topk_clusters_q in topk_clusters
    ]

    pairs = [
        (full(cg.size, qi, dtype=int32), cg.astype(int64, copy=False))
        for qi, cg in enumerate(cand_lists)
        if cg.size
    ]
    qi_list, gid_list = (list(t) for t in zip(*pairs)) if pairs else ([], [])

    if qi_list:
        all_qi = np.concatenate(qi_list)
        all_gid = np.concatenate(gid_list)
    else:
        all_qi = np.empty(0, dtype=int32)
        all_gid = np.empty(0, dtype=int64)

    queries_per_label = defaultdict(list)

    if all_qi.size:
        order = argsort(all_gid, kind="mergesort")
        gids_sorted = all_gid[order]
        qis_sorted = all_qi[order]
        # boundaries where gid changes
        boundaries = np.flatnonzero(np.diff(gids_sorted, prepend=gids_sorted[:1]-1))  # start indices
        # iterate unique gids and slice qi ranges
        for start, end in zip(boundaries, r_[boundaries[1:], gids_sorted.size]):
            gid = int(gids_sorted[start])
            queries_per_label[gid].extend(qis_sorted[start:end].tolist())

    # --- 4) Batch ranker per label and fuse with the label's cluster score ---
    # Honor the saved eval-only switch when no explicit scorer was supplied.
    cosine = scorer == "cosine" or (scorer is None and getattr(node, "cosine_scorer", False))
    no_rankers = cosine or node.ranker_model is None
    model_dict = {} if no_rankers else node.ranker_model.model_dict
    node.ranker_failed = set()  # gids whose trained ranker fell back to cosine

    # detect hinge-style rankers (like in your earlier code)
    is_hinge = False
    if node.ranker_model and node.ranker_model.model_dict:
        first_model = next(iter(node.ranker_model.model_dict.values()))
        cfg = getattr(first_model, "config", {})
        is_hinge = (cfg.get("type") == "sklearnsgdclassifier" and
                    cfg.get("kwargs", {}).get("loss") == "hinge")

    # Accumulate all (qi, gid, score) triples; we’ll pack at the end
    triples_qi, triples_gid, triples_sc = [], [], []

    def _cos_fallback(rows, zv):
        """Score for labels with no ranker (38.4% of them: n_pos < 2 in ranker.py).
        Returning ones() collapsed them onto the shared cluster score; cosine at least
        orders them. Mapped to [0, 1] to match the probability range _fuse expects."""
        den = RankerTrainer._row_norms(rows) * np.linalg.norm(zv) + 1e-12
        return ((rows @ zv) / den + 1.0) / 2.0

    for gid, q_indices in queries_per_label.items():
        li = g2l.get(gid)
        mdl = model_dict.get(gid)
        if li is None or not q_indices:
            continue

        c = int(label_cluster[li])
        q_idx = asarray(q_indices, dtype=int)
        m = cluster_scores[q_idx, c]

        zvec = Z[li]
        rows = X_query[q_idx]

        if mdl is None:
            r = _cos_fallback(rows, zvec)
        else:
            try:
                dense_rows = rows.toarray() if hasattr(rows, "toarray") else np.asarray(rows)
                batch_inp = ranker_input(dense_rows, zvec)
                proba_fn = getattr(mdl, "predict_proba", None)
                if proba_fn is not None and not is_hinge:
                    proba = proba_fn(batch_inp)
                    r = _pos_col(proba) 
                elif hasattr(mdl, "decision_function"):
                    r = expit(mdl.decision_function(batch_inp))
                else:
                    raise TypeError("ranker has no probability or decision scorer")
            except Exception as exc:
                node.logger.warning("Ranker failed for label %s; using cosine: %s", gid, exc)
                node.ranker_failed.add(gid)
                r = _cos_fallback(rows, zvec)

        fused = _fuse(m, r)   # branch decided once above

        # stash; avoid per-item append at the original line
        triples_qi.append(q_idx)
        triples_gid.append(full_like(q_idx, gid))
        triples_sc.append(fused.astype(float, copy=False))

    # --- 5) Pack results ---
    if return_matrix:
        if topk == 0 or not triples_qi:
            M = csr_matrix((n, 0))
            M.global_labels = np.array([], dtype=object)
            return M

        all_qi = np.concatenate(triples_qi)
        all_gid = np.concatenate(triples_gid)
        all_sc = np.concatenate(triples_sc)

        # optional per-query topk before CSR packing
        if topk is not None:
            # group by qi
            order = np.argsort(all_qi, kind="mergesort")
            qi_sorted = all_qi[order]
            gid_sorted = all_gid[order]
            sc_sorted = all_sc[order]

            rows, cols, data = [], [], []
            start = 0
            while start < qi_sorted.size:
                qi = qi_sorted[start]
                end = start + 1
                while end < qi_sorted.size and qi_sorted[end] == qi:
                    end += 1
                s = sc_sorted[start:end]
                g = gid_sorted[start:end]
                if topk and s.size > topk:
                    idx = argpartition(s, s.size - topk)[-topk:]
                    s = s[idx]; g = g[idx]
                    ord_ = argsort(-s); s = s[ord_]; g = g[ord_]
                rows.extend([qi] * s.size)
                cols.extend(g.tolist())
                data.extend(s.tolist())
                start = end

            all_gids = sorted(set(cols))
            gid_to_col = {g: i for i, g in enumerate(all_gids)}
            cols = [gid_to_col[g] for g in cols]

            M = csr_matrix((data, (rows, cols)), shape=(n, len(all_gids)))
            M.global_labels = array(all_gids, dtype=object)
            return M

        # no topk: straight CSR pack
        rows = np.concatenate(triples_qi)
        cols = np.concatenate(triples_gid)
        data = np.concatenate(triples_sc)

        all_gids = sorted(set(cols.tolist()))
        gid_to_col = {g: i for i, g in enumerate(all_gids)}
        cols = asarray([gid_to_col[g] for g in cols], dtype=int)

        M = csr_matrix((data, (rows, cols)), shape=(n, len(all_gids)))
        M.global_labels = array(all_gids, dtype=object)
        return M

    # list-of-lists output
    results = [[] for _ in range(n)]
    if triples_qi:
        all_qi = np.concatenate(triples_qi)
        all_gid = np.concatenate(triples_gid)
        all_sc = np.concatenate(triples_sc)

        # group by qi once; apply optional topk
        order = np.argsort(all_qi, kind="mergesort")
        qi_sorted = all_qi[order]
        gid_sorted = all_gid[order]
        sc_sorted = all_sc[order]

        start = 0
        while start < qi_sorted.size:
            qi = int(qi_sorted[start])
            end = start + 1
            while end < qi_sorted.size and qi_sorted[end] == qi:
                end += 1
            g = gid_sorted[start:end]
            s = sc_sorted[start:end]

            if topk and s.size > topk:
                idx = argpartition(s, s.size - topk)[-topk:]
                g = g[idx]; s = s[idx]
                ord_ = argsort(-s); g = g[ord_]; s = s[ord_]

            results[qi] = list(zip(g.astype(object).tolist(),
                                s.astype(float).tolist()))
            start = end

    # convert to arrays for return
    scores_per_query, labels_per_query = [], []
    for pairs in results:
        if not pairs:
            scores_per_query.append(array([], dtype=float))
            labels_per_query.append(array([], dtype=object))
        else:
            gids, scs = zip(*pairs)
            labels_per_query.append(array(gids, dtype=object))
            scores_per_query.append(array(scs, dtype=float))

    return scores_per_query, labels_per_query


def fused_cluster_scores(node, X, Z, C, alpha=0.5, batch_size=32768,
                  fusion: str = "lp_hinge", p: int = 3):
    """Batched matcher/ranker fusion."""
    N = X.shape[0]
    L_local = Z.shape[0]

    # --- matcher (local label-level) scores ---
    ms = csr_matrix(node.matcher_model.model.predict_proba(X), dtype=np.float32)

    # --- flatten mention-local label pairs (vectorized; no Python loop) ---
    indptr = ms.indptr
    rows_list = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr).astype(np.int32))
    cols_list = ms.indices.astype(np.int32, copy=False)
    matcher_flat = ms.data.astype(np.float32, copy=False)

    ranker_score = ones(len(rows_list), dtype=np.float32)

    # ---- SINGLE RANKER SHORTCUT ----
    if node.ranker_model and getattr(node.ranker_model, "model_dict", None):
        try:
            mdl = next(iter(node.ranker_model.model_dict.values()))
        except StopIteration:
            mdl = None

        if mdl is not None:
            clip_low, clip_high = 1e-6, 1.0
            POS_COL = 1

            # Detect hinge once
            cfg = getattr(mdl, "config", {})
            is_hinge = (cfg.get("type") == "sklearnsgdclassifier" and
                        cfg.get("kwargs", {}).get("loss") == "hinge")

            proba_fn = getattr(mdl, "predict_proba", None)
            dec_fn   = getattr(mdl, "decision_function", None)

            # Decide MODE ONCE (outside the loop), and bind a scorer callable.
            if is_hinge and callable(dec_fn):
                def scorer(Xb, _dec=dec_fn, _lo=clip_low, _hi=clip_high):
                    return clip(expit(_dec(Xb)), _lo, _hi)
            elif callable(proba_fn):
                def scorer(Xb, _pf=proba_fn, _col=POS_COL, _lo=clip_low, _hi=clip_high):
                    proba = _pf(Xb)
                    return clip(proba[:, _col], _lo, _hi)
            else:
                def scorer(Xb):
                    return ones(Xb.shape[0], dtype=float32)

            num_pairs = len(rows_list)
            for start in range(0, num_pairs, batch_size):
                end = min(start + batch_size, num_pairs)
                b_rows = rows_list[start:end]
                b_cols = cols_list[start:end]

                X_part = X[b_rows]
                X_part = X_part.toarray() if hasattr(X_part, "toarray") else asarray(X_part)
                Z_part = Z[b_cols]
                batch_inp = ranker_input(X_part, Z_part)

                # No loop-invariant checks here anymore:
                ranker_score[start:end] = scorer(batch_inp)
        else:
            alpha = 0
    else:
        alpha = 0  # no ranker available

    node.alpha = alpha

    if fusion == "lp_hinge":
        fused = ((1 - node.alpha) * (matcher_flat ** p) + node.alpha * (ranker_score ** p)) ** (1.0 / p)
        fused = maximum(fused, 0.0)
    else:
        fused = (matcher_flat ** (1 - node.alpha)) * (ranker_score ** node.alpha)

    # --- build local-label fused matrix ---
    entity_fused = csr_matrix((fused, (rows_list, cols_list)), shape=(N, L_local), dtype=np.float32)

    # --- project to clusters ---
    cluster_fused = entity_fused.dot(C)
    return csr_matrix(cluster_fused)
