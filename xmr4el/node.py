import os
import gc
import pickle
import tempfile
import logging
import shutil

import numpy as np

from os.path import dirname, isfile, join as pjoin, exists as pexists
from pickle import dump as pkl_dump
from joblib import dump as jdump
from scipy.sparse import csr_matrix, eye as sp_eye
from scipy.special import expit
from numpy import (
    asarray, array, concatenate, unique, tile, ones, maximum, clip,
    float32, int32, int64, argsort, argpartition, hstack as np_hstack,
    full, r_, full_like
)
from sklearn.preprocessing import normalize
from collections import defaultdict
from pathlib import Path
from xmr4el.clusterers import Clustering
from xmr4el.classifiers import Matcher
from xmr4el.ranker import Ranker


model_dir = Path(tempfile.mkdtemp(prefix="ml_model_dir"))

class MLModel():

    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 ranker_config=None,
                 cur_config=None,
                 min_leaf_size=20,
                 max_leaf_size=None,
                 ranker_every_layer=False,
                 is_last_layer=False,
                 layer=None,
                 n_workers=8,
                 train_rankers=True,
                 ):
        
        self.logger = logging.getLogger(__name__)
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.ranker_config = ranker_config
        self.cur_config = cur_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.ranker_every_layer = ranker_every_layer
        self.is_last_layer = is_last_layer
        self.layer = layer
        self.n_workers = n_workers
        self.train_rankers = train_rankers
        
        self._local_to_global_idx = None
        self._global_to_local_idx = None
        
        self._cluster_model = None
        self._matcher_model = None
        self._ranker_model = None
        self._fused_scores = None
        self._alpha = None
        self._label_embeddings = None
        self._ranker_score_fn_cache = None
    
    @property
    def local_to_global_idx(self):
        return self._local_to_global_idx
    
    @local_to_global_idx.setter
    def local_to_global_idx(self, arr: np.ndarray):
        """
        arr[i] should be the global KB label ID for local label index i.
        """
        self._local_to_global_idx = arr
        # build inverse map
        self._global_to_local_idx = {g: i for i, g in enumerate(arr)}
    
    @property
    def global_to_local_idx(self):
        return self._global_to_local_idx
    
    @global_to_local_idx.setter
    def global_to_local_idx(self, value):
        self._global_to_local_idx = value
    
    @property
    def cluster_model(self):
        return self._cluster_model
    
    @cluster_model.setter
    def cluster_model(self, value):
        self._cluster_model = value

    @property
    def matcher_model(self):
        return self._matcher_model
    
    @matcher_model.setter
    def matcher_model(self, value):
        self._matcher_model = value

    @property
    def ranker_model(self):
        return self._ranker_model
    
    @ranker_model.setter
    def ranker_model(self, value):
        self._ranker_model = value
        
    @property
    def fused_scores(self):
        return self._fused_scores
    
    @fused_scores.setter
    def fused_scores(self, value):
        self._fused_scores = value
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        
    @property
    def label_embeddings(self):
        return self._label_embeddings
    
    @label_embeddings.setter
    def label_embeddings(self, value):
        self._label_embeddings = value
    
    @property
    def is_empty(self):
        return True if self.cluster_model is None else False
    
    @staticmethod
    def save_model_temp(model , label: int) -> str:
        """Persist a temporary model for the given label."""
        sub_dir = model_dir / str(label)
        sub_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(sub_dir))
        return str(sub_dir)

    @staticmethod
    def delete_model_temp() -> None:
        """Remove all temporary ranker models from disk."""
        shutil.rmtree(model_dir)
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        state = self.__dict__.copy()

        # Mapping attribute names to their internal keys
        model_attrs = {
            "cluster_model": "_cluster_model",
            "matcher_model": "_matcher_model",
            "ranker_model": "_ranker_model"
        }

        for model_name, attr_key in model_attrs.items():
            model = getattr(self, model_name)
            if model is not None:
                model_path = pjoin(save_dir, model_name)

                if hasattr(model, 'save') and callable(model.save):
                    model.save(model_path)
                else:
                    jdump(model, f"{model_path}.joblib")

                # Remove model from state before pickling
                state.pop(attr_key, None)

        # Save fused scores separately
        fused_scores = self.fused_scores
        if fused_scores is None and not self.is_last_layer:
            raise ValueError("fused_scores is None. Cannot save.")
        if fused_scores is not None:
            np.save(pjoin(save_dir, "fused_scores.npy"), fused_scores)
        state.pop("_fused_scores", None)
        state.pop("_ranker_score_fn_cache", None)

        # Save label embeddings separately
        label_embeddings = self.label_embeddings
        if label_embeddings is not None:
            np.save(pjoin(save_dir, "label_embeddings.npy"), label_embeddings)
            state.pop("_label_embeddings", None)

        # Save remaining state
        with open(pjoin(save_dir, "mlmodel.pkl"), "wb") as fout:
            pkl_dump(state, fout)
    
    @classmethod
    def load(cls, load_path):
        # Accept either a directory OR a direct file to mlmodel.pkl
        base_dir = load_path
        # If they passed a file (e.g., .../mlmodel.pkl), go up one level
        if isfile(base_dir):
            base_dir = dirname(base_dir)

        model_state_path = pjoin(base_dir, "mlmodel.pkl")
        assert pexists(model_state_path), f"MLModel path {model_state_path} does not exist"

        with open(model_state_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)

        # Load sub-models saved in subfolders: <base_dir>/cluster_model, matcher_model, ranker_model
        model_dirs = {
            "cluster_model": Clustering if hasattr(Clustering, "load") else None,
            "matcher_model": Matcher if hasattr(Matcher, "load") else None,
            "ranker_model": Ranker if hasattr(Ranker, "load") else None,
        }

        for name, cls_ in model_dirs.items():
            subdir = pjoin(base_dir, name)
            if pexists(subdir) and cls_ is not None:
                setattr(model, name, cls_.load(subdir))
            else:
                print(f"Model {name} is not being loaded")

        # Load fused scores / label embeddings
        emb_path = pjoin(base_dir, "fused_scores.npy")
        assert pexists(emb_path) or model.is_last_layer, f"Expecting fused_scores at {emb_path}"
        model.fused_scores = np.load(emb_path, allow_pickle=True) if pexists(emb_path) else None

        label_emb_path = pjoin(base_dir, "label_embeddings.npy")
        model.label_embeddings = np.load(label_emb_path, allow_pickle=True) if pexists(label_emb_path) else None

        return model
        
    def __str__(self):
        return (
            f"Cluster Model: {self.cluster_model or 'None'}\n"
            f"Matcher Model: {self.matcher_model or 'None'}\n"
            f"Ranker Model: {self.ranker_model or 'None'}\n"
        )
    
    def fused_predict(self, X, Z, C, alpha=0.5, batch_size=32768,
                      fusion: str = "lp_hinge", p: int = 3):
        """Batched matcher/ranker fusion."""
        N = X.shape[0]
        L_local = Z.shape[0]

        # --- matcher (local label-level) scores ---
        ms = csr_matrix(self.matcher_model.model.predict_proba(X), dtype=np.float32)

        # --- flatten mention-local label pairs (vectorized; no Python loop) ---
        indptr = ms.indptr
        rows_list = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr).astype(np.int32))
        cols_list = ms.indices.astype(np.int32, copy=False)
        matcher_flat = ms.data.astype(np.float32, copy=False)

        ranker_score = ones(len(rows_list), dtype=np.float32)

        # ---- SINGLE RANKER SHORTCUT ----
        if self.ranker_model and getattr(self.ranker_model, "model_dict", None):
            try:
                mdl = next(iter(self.ranker_model.model_dict.values()))
            except StopIteration:
                mdl = None

            if mdl is not None:
                X_dense = X.toarray() if hasattr(X, "toarray") else asarray(X)
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

                    X_part = X_dense[b_rows]
                    Z_part = Z[b_cols]
                    batch_inp = np_hstack([X_part, Z_part])

                    # No loop-invariant checks here anymore:
                    ranker_score[start:end] = scorer(batch_inp)
            else:
                alpha = 0
        else:
            alpha = 0  # no ranker available

        self.alpha = alpha

        if fusion == "lp_hinge":
            fused = ((1 - self.alpha) * (matcher_flat ** p) + self.alpha * (ranker_score ** p)) ** (1.0 / p)
            fused = maximum(fused, 0.0)
        else:
            fused = (matcher_flat ** (1 - self.alpha)) * (ranker_score ** self.alpha)

        # --- build local-label fused matrix ---
        entity_fused = csr_matrix((fused, (rows_list, cols_list)), shape=(N, L_local), dtype=np.float32)

        # --- project to clusters ---
        cluster_fused = entity_fused.dot(C)
        return csr_matrix(cluster_fused)
    
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
            X_train: X_processed
            Y_train, Y_binazier
            Z, Pifa embeddings
        """
        
        # --- Ensure Z is in fused space ---
        Z_train = normalize(Z_train, norm="l2", axis=1) 
        self.label_embeddings = Z_train
        del Z_train
        
        self.global_to_local_idx = global_to_local
        self.local_to_global_idx = np.array(local_to_global, dtype=int)
        
        del global_to_local
        
        self.logger.info("Training ML: Clustering Phase")
        
        cluster_model = Clustering()
        if not self.is_last_layer:  # a leaf uses identity C below, so it is never clustered
            cluster_model.train(Z=self.label_embeddings,
                                local_to_global_idx=self.local_to_global_idx,
                                min_leaf_size=self.min_leaf_size,
                                max_leaf_size=self.max_leaf_size,
                                clustering_config=self.clustering_config,
                                dtype=np.float32
                                )
            if cluster_model.is_empty:
                return
        
        self.cluster_model = cluster_model
        del cluster_model
        gc.collect()
        
        # Retrieve C
        C = self.cluster_model.c_node

        # Defect #6: a cluster-level matcher makes every label in a cluster tie at predict time
        # (`m = cluster_scores[q_idx, c]` below), so ordering inside a leaf is decided by CSR index
        # order. Identity C at the leaf makes the matcher one-vs-rest over labels -- XR-Linear leaf
        # semantics -- so `M = Y_node @ I = Y_node` and `label_cluster` becomes arange(L), which
        # makes `m` per-label with no change at the scoring site.
        if self.is_last_layer:
            C = sp_eye(self.label_embeddings.shape[0], format="csr", dtype=np.float32)
            self.cluster_model.c_node = C   # predict() re-reads this and asserts K == C.shape[1]

        cluster_labels = np.asarray(C.argmax(axis=1)).flatten()
    
        self.logger.info("Training ML: Matcher Phase")

        # With identity C above, the leaf matcher is one-vs-rest over labels and a leaf label can
        # have a single positive instance. SGDClassifier's `early_stopping` splits off a validation
        # set *stratified* on y, which needs >= 2 members per class and raises otherwise. Disabled
        # for the leaf only, so the non-leaf layers stay byte-identical and the #6 row stays
        # attributable to #6.
        matcher_config = self.matcher_config
        if self.is_last_layer and matcher_config.get("type") == "sklearnsgdclassifier":
            matcher_config = {
                **matcher_config,
                "kwargs": {**matcher_config.get("kwargs", {}), "early_stopping": False},
            }

        # Make the Matcher
        matcher_model = Matcher()  
        matcher_model.train(X_train, 
                            Y_train, 
                            local_to_global_idx=self.local_to_global_idx, 
                            global_to_local_idx=self.global_to_local_idx, 
                            C=C,
                            matcher_config=matcher_config,
                            dtype=np.float32
                            )     
         
        self.matcher_model = matcher_model 
        del matcher_model
        gc.collect()
        
        # Rankers are only used for prediction with -scorer ranker; internal training scores
        # are matcher-only. train_rankers=False skips them; predict then uses cosine.
        train_ranker = self.train_rankers and (self.ranker_every_layer or self.is_last_layer)
        
        def _topb_sparse(P: np.ndarray, b: int) -> csr_matrix:
            # P: (n x K_or_L) dense proba; returns (n x K_or_L) CSR 0/1 mask of top-b per row
            n, K = P.shape
            b = max(1, min(b, K))
            idx_part = np.argpartition(P, K - b, axis=1)[:, -b:]
            rows = np.repeat(np.arange(n, dtype=np.int32), b)
            cols = idx_part.ravel()
            data = np.ones(n * b, dtype=np.int8)
            return csr_matrix((data, (rows, cols)), shape=(n, K))
        
        if train_ranker:
        
            M_TFN = self.matcher_model.m_node
            M_MAN = None
        
            if self.is_last_layer:
                P = self.matcher_model.predict_proba(X_train)
                # With identity C above, M_TFN is Y_node, so this top-b mask is the ranker's whole
                # negative pool (ranker.py) instead of a cluster's worth of instances.
                # b=5 leaves too few negatives for neg_mult * n_pos; 20 keeps the pool fed.
                M_MAN = _topb_sparse(P, b=20)
            
            self.logger.info("Training ML: Ranker Phase")
            
            # print("Ranker")
            ranker_model = Ranker()
            ranker_model.train(X_train, 
                                Y_train, 
                                self.label_embeddings, 
                                M_TFN, 
                                M_MAN, 
                                cluster_labels,
                                local_to_global_idx=self.local_to_global_idx,
                                layer=self.layer,
                                n_label_workers=self.n_workers,
                                ranker_config=self.ranker_config,
                                cur_config=self.cur_config
                                )
            
            self.ranker_model = ranker_model
            del ranker_model
        else:
            self.ranker_model = None
            
        gc.collect()
        
        self.fused_scores = None
        if not self.is_last_layer:
            print("Fusing Scores")
            cluster_scores = self.matcher_model.predict_proba(X_train)
            self.fused_scores = csr_matrix(np.maximum(cluster_scores, 0.0))
        
    def predict(self, X_query, beam_size: int = 5, topk: int | None = None, return_matrix: bool = False, 
                fusion: str = "lp_fusion", eps: float = 1e-6, alpha: float = 0.5, p: int = 3):
        cluster_scores = self.matcher_model.predict_proba(X_query)
        n, K = cluster_scores.shape

        C = self.cluster_model.c_node                                # shape (L, K) label->cluster (CSR)
        L = C.shape[0]
        assert K == C.shape[1], f"Matcher outputs K={K} clusters, but C has {C.shape[1]}."
        assert L == len(self.local_to_global_idx), "C rows must equal #local labels"

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
        cluster_to_global = [self.local_to_global_idx[C_csc[:, c].indices] for c in range(K)]
        label_cluster = np.asarray(C.argmax(axis=1)).ravel()
        g2l = self.global_to_local_idx
        Z = self.label_embeddings

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
        X_dense = X_query.toarray() if hasattr(X_query, "toarray") else np.asarray(X_query)
        # eval-only switch (set by test_evaluate_pipeline -scorer cosine): skip trained rankers and
        # score every label by cosine to its leaf z, like the no-ranker fallback.
        no_rankers = getattr(self, "cosine_scorer", False) or self.ranker_model is None
        model_dict = {} if no_rankers else self.ranker_model.model_dict
        self.ranker_failed = set()  # gids whose trained ranker silently fell back to cosine

        # detect hinge-style rankers (like in your earlier code)
        is_hinge = False
        if self.ranker_model and self.ranker_model.model_dict:
            first_model = next(iter(self.ranker_model.model_dict.values()))
            cfg = getattr(first_model, "config", {})
            is_hinge = (cfg.get("type") == "sklearnsgdclassifier" and
                        cfg.get("kwargs", {}).get("loss") == "hinge")

        # Accumulate all (qi, gid, score) triples; we’ll pack at the end
        triples_qi, triples_gid, triples_sc = [], [], []

        def _cos_fallback(rows, zv):
            """Score for labels with no ranker (38.4% of them: n_pos < 2 in ranker.py).
            Returning ones() collapsed them onto the shared cluster score; cosine at least
            orders them. Mapped to [0, 1] to match the probability range _fuse expects."""
            den = np.linalg.norm(rows, axis=1) * np.linalg.norm(zv) + 1e-12
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
            Z_tiled = tile(zvec, (len(q_idx), 1))
            batch_inp = np_hstack([X_dense[q_idx], Z_tiled])

            if mdl is None:
                r = _cos_fallback(X_dense[q_idx], zvec)
            else:
                try:
                    proba_fn = getattr(mdl, "predict_proba", None)
                    if proba_fn is not None and not is_hinge:
                        proba = proba_fn(batch_inp)
                        r = _pos_col(proba) 
                    elif hasattr(mdl, "decision_function"):
                        r = expit(mdl.decision_function(batch_inp))
                    else:
                        self.ranker_failed.add(gid)
                        r = _cos_fallback(X_dense[q_idx], zvec)
                except Exception:
                    self.ranker_failed.add(gid)
                    r = _cos_fallback(X_dense[q_idx], zvec)

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
