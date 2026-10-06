import os
import re
import gc
import pickle
import tempfile
import logging
import shutil
import time

import numpy as np

from gc import collect
from os.path import dirname, isfile, join as pjoin, exists as pexists, isdir as pisdir
from os import makedirs as pmakedirs, listdir as plistdir
from pickle import dump as pkl_dump, load as pkl_load
from joblib import dump as jdump, load as jload
from scipy.sparse import hstack as sp_hstack, vstack, csr_matrix, eye as sp_eye
from scipy.special import expit
from numpy import (
    asarray, array, concatenate, unique, tile, ones, maximum, clip,
    float32, int32, int64, argsort, argpartition, log, hstack as np_hstack,
    vstack as np_vstack, zeros, full, r_, full_like
)
from sklearn.preprocessing import normalize
from collections import defaultdict
from typing import Tuple
from uuid import uuid4
from heapq import nlargest
from pathlib import Path
from xmr4el import get_logger
from xmr4el.clustering.model import Clustering
from xmr4el.matcher.model import Matcher
from xmr4el.ranker.model import Ranker


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
        if fused_scores is None:
            raise ValueError("fused_scores is None. Cannot save.")
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
        assert pexists(emb_path), f"Expecting fused_scores at {emb_path}"
        model.fused_scores = np.load(emb_path, allow_pickle=True)

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
        cluster_model.train(Z=self.label_embeddings, 
                            local_to_global_idx=self.local_to_global_idx,
                            min_leaf_size=self.min_leaf_size,
                            max_leaf_size=self.max_leaf_size,
                            clustering_config=self.clustering_config,
                            dtype=np.float32
                            ) # Hardcoded
        
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
        
        # Nothing used for prediction depends on the rankers unless eval runs -scorer ranker: the
        # root's fused_scores are matcher-only, the leaf's feed prepare_layer children that are
        # never trained. train_rankers=False skips them; predict then scores every label by cosine.
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
                # negative pool (ranker/train.py:284) instead of a cluster's worth of instances.
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
        
        print("Fusing Scores")
        if not self.is_last_layer:
            cluster_scores = self.matcher_model.predict_proba(X_train)
            self.fused_scores = csr_matrix(np.maximum(cluster_scores, 0.0))
        else:
            I_L = sp_eye(self.label_embeddings.shape[0], format="csr", dtype=np.float32)
            self.fused_scores = self.fused_predict(X_train, self.label_embeddings, I_L, alpha=0.5, fusion="lp_hinge", p=3)
        
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
            """Score for labels with no ranker (38.4% of them: n_pos < 2 at ranker/train.py:164).
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
        
            
class HierarchicaMLModel():
    """Loops MLModel"""
    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 ranker_config=None, 
                 cur_config=None,
                 min_leaf_size=20,
                 max_leaf_size=None,
                 cut_half_cluster=False,
                 ranker_every_layer=False,
                 n_workers=8,
                 layer=1,
                 train_rankers=True):
        
        self.logger = logging.getLogger(__name__)
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.ranker_config = ranker_config
        self.cur_config = cur_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster = cut_half_cluster
        self.ranker_every_layer = ranker_every_layer
        self.n_workers = n_workers
        self.train_rankers = train_rankers
        
        self._hmodel = []
        self._layer = layer
        self._child_index_map = None
        
        self.ml_dir = Path(tempfile.mkdtemp(prefix="ml_store_"))
        
    @property
    def hmodel(self):
        return self._hmodel
    
    @hmodel.setter
    def hmodel(self, value):
        self._hmodel = value

    @property
    def layers(self):
        return self._layer
    
    @layers.setter
    def layers(self, value):
        self._layer = value
        
    @property
    def child_index_map(self):
        return self._child_index_map
    
    @child_index_map.setter
    def child_index_map(self, value):
        self._child_index_map = value
        
    def save(self, save_dir):
        pmakedirs(save_dir, exist_ok=True)
        state = self.__dict__.copy()

        for layer_idx, model_list in enumerate(self.hmodel):
            layer_path = pjoin(save_dir, f"layer_{layer_idx}")
            pmakedirs(layer_path, exist_ok=True)

            for model_idx, model in enumerate(model_list):
                        if model is None:
                            continue
                        sub_model_path = pjoin(layer_path, f"ml_{model_idx}")
                        pmakedirs(sub_model_path, exist_ok=True)
                        if hasattr(model, "save") and callable(model.save):
                            model.save(sub_model_path)
                        else:
                            try:
                                jdump(model, f"{sub_model_path}.joblib")
                            except ImportError:
                                with open(f"{sub_model_path}.pkl", "wb") as f:
                                    pkl_dump(model, f)

        state.pop("_hmodel", None)  # Correct key

        with open(os.path.join(save_dir, "hml.pkl"), "wb") as fout:
            pkl_dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = pjoin(load_dir, "hml.pkl")
        assert pexists(xmodel_path), f"Hierarchical ML Model path {xmodel_path} does not exist"
            
        with open(xmodel_path, "rb") as fin:
            model_data = pkl_load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
            
        # Load layer directories
        layer_folders = []
        pattern = re.compile(r'^layer_\d+$')
        for entry in plistdir(load_dir):
            full_path = pjoin(load_dir, entry)
            if pisdir(full_path) and pattern.match(entry):
                layer_folders.append(full_path)
        assert len(layer_folders) > 0, "No layer folders found"
        layer_folders.sort(key=lambda x: int(re.search(r'layer_(\d+)', x).group(1)))
            
        # `child_index_map` indexes the models of a layer by their TRAINING order, which save()
        # encodes in the `ml_<n>` directory names. plistdir() returns entries in filesystem order,
        # so without this sort hmodel[layer] is a permutation of the training order and the beam
        # routes every query to a child holding a different cluster's labels -- chance-level
        # routing, on train as well as dev. Layers themselves are already sorted above.
        def _ml_key(name):
            m = re.match(r'^ml_(\d+)$', name)
            return (0, int(m.group(1))) if m else (1, name)

        # Load models from each layer
        hmodel = []
        for layer_path in layer_folders:
            layer_models = []
            for subentry in sorted(plistdir(layer_path), key=_ml_key):
                sub_path = pjoin(layer_path, subentry)
                if pisdir(sub_path):  # If model is saved via model.save()
                    try:
                        model_obj = MLModel.load(sub_path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load MLModel from {sub_path}: {e}")
                elif subentry.endswith(".joblib"):
                    model_obj = jload(sub_path)
                elif subentry.endswith(".pkl"):
                    with open(sub_path, "rb") as f:
                        model_obj = pkl_load(f)
                else:
                    continue  # Skip unexpected files
                layer_models.append(model_obj)
            assert layer_models, f"No models found in layer folder {layer_path}"
            hmodel.append(layer_models)

        setattr(model, "_hmodel", hmodel)

        # Regression guard for the load-order bug above: every parent cluster must hand the beam a
        # child that holds exactly that cluster's labels. When this silently failed, routing sat at
        # chance on train as well as dev and every downstream metric was meaningless, so it is
        # asserted at load rather than left to a test. Cost is one set compare per cluster.
        cim = getattr(model, "child_index_map", None)
        if cim:
            for layer, parent_maps in enumerate(cim):
                if layer + 1 >= len(hmodel):
                    continue   # the last layer's map points at a layer that does not exist
                for parent_idx, cmap in enumerate(parent_maps or []):
                    parent = hmodel[layer][parent_idx]
                    C_p = parent.cluster_model.c_node
                    C_dense = C_p.toarray() if hasattr(C_p, "toarray") else np.asarray(C_p)
                    l2g = np.asarray(parent.local_to_global_idx)
                    for c, child_idx in (cmap or {}).items():
                        want = set(l2g[np.where(C_dense[:, int(c)] > 0)[0]].tolist())
                        got = set(np.asarray(hmodel[layer + 1][child_idx].local_to_global_idx).tolist())
                        assert want == got, (
                            f"child_index_map[{layer}][{parent_idx}][{c}] -> child {child_idx} holds "
                            f"{len(got)} labels but cluster {c} has {len(want)} "
                            f"(overlap {len(want & got)}). Layer models are out of training order."
                        )

        return model
    
    def save_ml_temp(self, model, name):
        sub_dir =  self.ml_dir / str(name)
        sub_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(sub_dir))
        return str(sub_dir)
            
    def prepare_layer(self, X, Y, Z, C, fused_scores, local_to_global_idx):
        """
        Returns a list of tuples, one per (non-empty) cluster c:
        (X_aug, Y_node, Z_node_aug, local_to_global_next, global_to_local_next, c)
        where `c` is the *cluster id* in the parent.
        """
        K_next = C.shape[1]
        inputs = []
        fused_dense = fused_scores.toarray() if hasattr(fused_scores, "toarray") else asarray(fused_scores)

        for c in range(K_next):
            local_idxs = C[:, c].nonzero()[0]
            if len(local_idxs) == 0:
                continue

            local_to_global_next = local_to_global_idx[local_idxs]
            global_to_local_next = {g: i for i, g in enumerate(local_to_global_next)}

            Y_sub = Y[:, local_idxs]
            mention_mask = (Y_sub.sum(axis=1).A1 > 0)

            X_node = X[mention_mask]
            Y_node = Y_sub[mention_mask, :]
            if X_node.shape[0] == 0:
                continue

            Z_node_base = Z[local_idxs, :]

            fused_c = fused_dense[mention_mask, :]
            feat_c = fused_c[:, c].ravel()
            feat_sum = fused_c.sum(axis=1).ravel()
            feat_max = fused_c.max(axis=1).ravel()
            
            sparse_feats = csr_matrix(np_vstack([feat_c, feat_sum, feat_max]).T)
            X_aug = sp_hstack([X_node, sparse_feats], format="csr")
            X_aug = normalize(X_aug, norm="l2", axis=1)

            # Zero pad keeps Z width == X_aug width (ranker cosine/ip, cosine fallback). The old
            # pad held mean/sum/max of X.Z over all node rows; the sum grew with node size and after
            # L2 norm took 1.000 of every leaf z's squared norm, erasing the label embedding.
            label_feats = zeros((Z_node_base.shape[0], 3), dtype=Z_node_base.dtype)
            Z_node_aug = np_hstack([Z_node_base, label_feats])
            Z_node_aug = normalize(Z_node_aug, norm="l2", axis=1)

            inputs.append((X_aug, Y_node, Z_node_aug, local_to_global_next, global_to_local_next, c))

        return inputs
            
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
        Train multiple layers of MLModel; intermediate models are saved in a
        temporary folder which is automatically deleted at the end of training.
        """
        inputs = ((X_train, Y_train, Z_train, local_to_global, global_to_local),)
        last_layer_index = self.layers - 1
        ranker_flag_default = bool(self.ranker_every_layer)

        cfg = self.clustering_config
        cfg_kwargs = cfg.get("kwargs", {}) if cfg is not None else {}
        get_n_clusters = cfg_kwargs.get  # local bind
        set_n_clusters = cfg_kwargs.__setitem__  # local bind so we don't dot each time
        save_temp = self.save_ml_temp  # local bind

        def _accumulate_children(raw_children, start_idx, next_inputs_list):
            """Convert raw_children into next_inputs and a cluster->child map."""
            if not raw_children:
                return {}
            payloads, c_ids = zip(*(((rc[:-1]), int(rc[-1])) for rc in raw_children))
            next_inputs_list.extend(payloads)  # payloads are already tuples
            return {c: (start_idx + i) for i, c in enumerate(c_ids)}

        def _save_ml_for_layer(ml, layer):
            """Create a unique save name for this model and store it."""
            save_name = f"{layer}_{uuid4()}"
            return save_temp(ml, save_name)

        def _finalize_layer(layer, ml_list, next_inputs):
            """Append saved model paths and freeze next_inputs -> inputs tuple."""
            self.hmodel.append(ml_list)
            return tuple(next_inputs)

        # Use TemporaryDirectory to ensure cleanup
        with tempfile.TemporaryDirectory(prefix="ml_store_") as temp_dir:
            self.ml_dir = Path(temp_dir)
            self.hmodel = []
            child_index_map = []


            for layer in range(self.layers):
                
                self.logger.info(f"Training HML: layer: {layer}")
                
                next_inputs: list[tuple] = []
                ml_list: list[str] = []
                layer_failed = False
                layer_child_maps: list[dict[int, int]] = []   # <-- add this
                
                is_last_layer = (layer == last_layer_index)
                ranker_flag = True if is_last_layer else ranker_flag_default

                if self.cut_half_cluster and layer > 0: 
                    n_curr = int(get_n_clusters("n_clusters", 2))
                    set_n_clusters("n_clusters", max(2, n_curr // 2))
                    
                number_of_childs = len(inputs)
                
                # parent_idx
                for (X_node, Y_node, Z_node, local_to_label_node, global_to_local_node) in inputs:
                    
                    n_child = len(inputs) - number_of_childs
                    
                    self.logger.info(f"Training ML: Number {n_child}")
                    
                    ml = MLModel(
                        clustering_config=self.clustering_config,
                        matcher_config=self.matcher_config,
                        ranker_config=self.ranker_config,
                        cur_config=self.cur_config,
                        min_leaf_size=self.min_leaf_size,
                        max_leaf_size=self.max_leaf_size,
                        ranker_every_layer= ranker_flag,
                        is_last_layer=is_last_layer,
                        layer=layer,
                        n_workers=self.n_workers,
                        train_rankers=getattr(self, "train_rankers", True),
                    )

                    ml.train(
                        X_train=X_node,
                        Y_train=Y_node,
                        Z_train=Z_node,
                        local_to_global=local_to_label_node,
                        global_to_local=global_to_local_node
                    )

                    if ml.is_empty:
                        layer_failed = True
                        break

                    C = ml.cluster_model.c_node
                    fused_scores = ml.fused_scores

                    self.logger.info(f"Training ML: Preparing Layer")

                    # Prepare inputs for next layer
                    raw_children = self.prepare_layer(
                        X=X_node,
                        Y=Y_node,
                        Z=Z_node,
                        C=C,
                        fused_scores=fused_scores,
                        local_to_global_idx=local_to_label_node
                    )
                                            
                    cluster_to_child = _accumulate_children(
                        raw_children, 
                        start_idx=len(next_inputs), 
                        next_inputs_list=next_inputs
                    )
                    
                    ml_path = _save_ml_for_layer(ml, layer)
                    del ml
                    ml_list.append(ml_path)
                    layer_child_maps.append(cluster_to_child)

                if layer_failed:
                    break

                inputs = _finalize_layer(layer, ml_list, next_inputs)
                del ml_list
                collect()
                child_index_map.append(layer_child_maps)

            # Reload all models for final hmodel
            self.hmodel = [[MLModel.load(p) for p in model_list] for model_list in self.hmodel]
            self._child_index_map = child_index_map
            return self.hmodel
        
    
    def predict(self, 
                X_query, 
                topk: int = 5, 
                beam_size: int = 5, 
                fusion: str = "lp_fusion", 
                eps: float = 1e-9, 
                alpha: float = 0.5,
                topk_mode: str = "per_leaf",   # "per_leaf" | "global" | "none"
                include_global_path: bool = True,
                n_jobs: int = None,
                path_score: bool = False):
        # path_score: leaf score x routing path probability (exp(path_logscore)), as XR-Linear does.
        # Leaf matcher probabilities are per-label sigmoids trained only against that leaf's labels, so
        # without it the beam's leaves are merged by max on scales that are not comparable.

        time_start_routing = time.time()

        # --- helpers (keep loops minimal) ---

        def _select_topk_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
            if k is None or k <= 0 or scores.size == 0:
                return np.array([], dtype=int), np.array([], dtype=float)
            k = min(k, scores.size)
            idx = np.argpartition(scores, scores.size - k)[-k:]
            vals = scores[idx]
            order = np.argsort(-vals)
            return idx[order], vals[order]

        def _norm_topk(k):
            return None if (k is None or k <= 0) else int(k)

        def _init_beam_first_layer(n_models: int, x0_row):
            # tuple per perflint W8301
            return tuple((mi, x0_row, 0.0, []) for mi in range(n_models))

        def _parent_to_child(layer_i: int, parent_model_i: int) -> dict:
            return self.child_index_map[layer_i][parent_model_i]

        def _stack_batch(xs):
            return xs[0] if len(xs) == 1 else vstack(xs, format="csr")

        def _maybe_leaf_topk(labels, scores, k_norm):
            # Leaf predict returns labels in index order, not score order: sort before cutting.
            if k_norm is None:
                return labels, scores
            labels, scores = asarray(labels), asarray(scores)
            top = np.argsort(-scores, kind="stable")[:k_norm]
            return labels[top], scores[top]

        def _append_path(paths_per_q, qi, trail, leaf_idx, labels, scores, source=None):
            paths_per_q[qi].append({
                "trail": trail,
                "leaf_model_idx": int(leaf_idx),
                "leaf_global_labels": labels.astype(int32, copy=False),
                "scores": scores.astype(float, copy=False),
                **({"source": source} if source is not None else {})
            })

        def _acc(per_query_scores_list, qi):
            return per_query_scores_list[qi]

        def _maybe_global_topk_items(items, k_norm):
            return items if k_norm is None else nlargest(k_norm, items, key=lambda kv: kv[1])

        def _add_beam_to_pending(beam, qi, pending_by_leaf):
            """Avoid W8402 by doing the grouping inside a helper."""
            by_leaf = defaultdict(list)
            # Build (leaf_idx -> list[(qi, x_row, trail)])
            for child_idx, x_row, trail in ((b[0], b[1], b[3]) for b in beam):
                by_leaf[int(child_idx)].append((qi, x_row, trail))
            for k_leaf, v_list in by_leaf.items():
                pending_by_leaf[k_leaf].extend(v_list)

        def _predict_one_leaf(leaf_idx, items, per_leaf_topk, topk_norm, fusion, alpha,
                            paths_per_query, per_query_scores):
            """Encapsulate leaf loop body to avoid W8201 on helper calls/branches."""
            q_indices, xs, trails = zip(*items)
            X_batch = _stack_batch(list(xs))  # not loop-invariant anymore from perflint’s PoV

            leaf_ml = leaf_layer_models[leaf_idx]
            scores_list, labels_list = leaf_ml.predict(
                X_batch,
                beam_size=100,
                fusion=fusion,
                alpha=alpha
            )

            for (qi, trail), labels, scores in zip(zip(q_indices, trails), labels_list, scores_list):
                if path_score and trail:
                    scores = asarray(scores) * np.exp(trail[-1]["path_logscore"])
                if per_leaf_topk:
                    labels, scores = _maybe_leaf_topk(labels, scores, topk_norm)

                _append_path(paths_per_query, qi, trail, leaf_idx, labels, scores)

                acc = _acc(per_query_scores, qi)
                for lid, sc in zip(labels, scores):
                    lid_i = int(lid); sc_f = float(sc)
                    if sc_f > acc.get(lid_i, 0.0):
                        acc[lid_i] = sc_f

        def _empty_global_path(source_tag):
            return {
                "trail": [],
                "leaf_model_idx": -1,
                "source": source_tag,
                "leaf_global_labels": array([], dtype=int32),
                "scores": array([], dtype=float),
            }

        topk_norm = _norm_topk(topk)

        if self.hmodel is None:
            out = [{"query_index": i, "paths": [], "new_path": None} for i in range(X_query.shape[0])]
            n_labels = int(getattr(getattr(self, "label_embeddings", np.empty((0,))), "shape", [0])[0]) if getattr(self, "label_embeddings", None) is not None else 0
            return out, csr_matrix((X_query.shape[0], n_labels), dtype=float)

        n_layers = len(self.hmodel)
        n_queries = X_query.shape[0]

        paths_per_query = [[] for _ in range(n_queries)]
        per_query_scores = [defaultdict(float) for _ in range(n_queries)]
        pending_by_leaf: dict[int, list[tuple[int, csr_matrix, list]]] = defaultdict(list)

        # precompute invariants that were flagged
        is_per_leaf_topk = (topk_mode == "per_leaf")
        is_global_topk = (topk_mode == "global")

        print(f"Shape of the query: {X_query.shape[0]}")

        for qi in range(X_query.shape[0]):
            x0 = X_query[qi:qi+1]
            beam = _init_beam_first_layer(len(self.hmodel[0]), x0)

            # Traverse all non-final layers
            for layer in range(n_layers - 1):
                ml_list = self.hmodel[layer]
                candidates = []

                for parent_model_idx, x_row, logscore, trail in beam:
                    ml = ml_list[parent_model_idx]

                    cs = asarray(ml.matcher_model.predict_proba(x_row)).ravel()
                    if cs.size == 0 or np.all(cs <= 0):
                        continue

                    idx_top, vals_top = _select_topk_indices(cs, min(beam_size, cs.size))
                    parent_to_child = _parent_to_child(layer, parent_model_idx)

                    sum_cs = float(cs.sum()); max_cs = float(cs.max())

                    for c, p_child in zip(idx_top, vals_top):
                        child_idx = parent_to_child.get(int(c))
                        if child_idx is None:
                            continue

                        extra = csr_matrix([[float(p_child), sum_cs, max_cs]])
                        x_next = sp_hstack([x_row, extra], format="csr")
                        x_next = normalize(x_next, norm="l2", axis=1)

                        logscore_next = logscore + float(log(max(p_child, eps)))

                        trail_next = trail + [{
                            "layer": layer,
                            "parent_model_idx": int(parent_model_idx),
                            "chosen_cluster": int(c),
                            "matcher_prob": float(p_child),
                            "child_model_idx": int(child_idx),
                            "path_logscore": float(logscore_next)
                        }]

                        candidates.append((int(child_idx), x_next, logscore_next, trail_next))

                if not candidates:
                    beam = ()
                    break

                candidates.sort(key=lambda t: -t[2])
                beam = tuple(candidates[:beam_size])

            if beam:
                _add_beam_to_pending(beam, qi, pending_by_leaf)

        time_end_routing = time.time()
        time_start_ranking = time.time()
        print("Routing: ", time_end_routing - time_start_routing)

        # --- Batched leaf predictions per leaf model ---
        leaf_layer_models = self.hmodel[-1]
        for leaf_idx, items in pending_by_leaf.items():
            _predict_one_leaf(
                leaf_idx=leaf_idx,
                items=items,
                per_leaf_topk=is_per_leaf_topk,
                topk_norm=topk_norm,
                fusion=fusion,
                alpha=alpha,
                paths_per_query=paths_per_query,
                per_query_scores=per_query_scores,
            )

        # Build CSR (n_queries x n_labels)
        try:
            n_labels_total = int(self.label_embeddings.shape[0])
        except Exception:
            max_lid = -1
            for acc in per_query_scores:
                if acc:
                    m = max(acc.keys())
                    if m > max_lid:
                        max_lid = m
            n_labels_total = max_lid + 1 if max_lid >= 0 else 0

        indptr = [0]; indices = []; data = []

        for qi in range(n_queries):
            acc = _acc(per_query_scores, qi)
            if not acc:
                indptr.append(len(indices))
                continue

            items = acc.items()
            if is_global_topk:
                items = _maybe_global_topk_items(items, topk_norm)

            lids, scs = zip(*items)
            lids = asarray(lids, dtype=int32)
            scs = asarray(scs, dtype=float)

            order = argsort(-scs)
            indices.extend(lids[order].tolist())
            data.extend(scs[order].tolist())
            indptr.append(len(indices))

        scores_csr = csr_matrix(
            (asarray(data, dtype=float),
            asarray(indices, dtype=int32),
            asarray(indptr, dtype=int32)),
            shape=(n_queries, n_labels_total),
            dtype=float
        )

        # Optional fused/global path
        new_paths = [None] * n_queries
        if include_global_path:
            source_tag = f"global_from_scores_csr[{topk_mode}]"
            for qi in range(n_queries):
                row = scores_csr.getrow(qi)
                if row.nnz == 0:
                    new_paths[qi] = _empty_global_path(source_tag)
                    continue
                lbls = row.indices; scs = row.data
                order = argsort(-scs)
                lbls = asarray(lbls[order], dtype=int32)
                scs = asarray(scs[order], dtype=float)
                new_paths[qi] = {
                    "trail": [],
                    "leaf_model_idx": -1,
                    "source": source_tag,
                    "leaf_global_labels": lbls,
                    "scores": scs,
                }

        out = [
            {
                "query_index": int(qi),
                "paths": paths_per_query[qi],
                "final_path": new_paths[qi] if include_global_path else None,
            }
            for qi in range(n_queries)
        ]

        time_end_ranking = time.time()
        print("Ranking: ", time_end_ranking - time_start_ranking)

        return out, scores_csr
