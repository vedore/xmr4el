import os
import re
import tempfile
import logging
import time

import numpy as np

from gc import collect
from copy import deepcopy
from os.path import join as pjoin, exists as pexists, isdir as pisdir
from os import makedirs as pmakedirs, listdir as plistdir
from pickle import dump as pkl_dump, load as pkl_load
from joblib import dump as jdump, load as jload
from scipy.sparse import hstack as sp_hstack, vstack, csr_matrix
from numpy import (
    asarray, array, int32, argsort, log, hstack as np_hstack,
    vstack as np_vstack, zeros
)
from sklearn.preprocessing import normalize
from collections import defaultdict
from typing import Tuple
from uuid import uuid4
from heapq import nlargest
from pathlib import Path
from xmr4el.hierarchy.node import MLModel

def augment_features(X, selected, total, maximum):
    """Append routing probabilities and normalize identically during train and predict."""
    extra = csr_matrix(np_vstack([selected, total, maximum]).T)
    return normalize(sp_hstack([X, extra], format="csr"), norm="l2", axis=1)


class HierarchicalMLModel():
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
        
        self.ml_dir = None
        
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
            
            X_aug = augment_features(X_node, feat_c, feat_sum, feat_max)

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
        clustering_config = deepcopy(self.clustering_config)
        last_layer_index = self.layers - 1
        ranker_flag_default = bool(self.ranker_every_layer)

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
                layer_start = time.perf_counter()
                self.logger.info("Layer started: layer=%d/%d nodes=%d", layer + 1, self.layers, len(inputs))
                
                next_inputs: list[tuple] = []
                ml_list: list[str] = []
                layer_child_maps: list[dict[int, int]] = []   # <-- add this
                
                is_last_layer = (layer == last_layer_index)
                ranker_flag = True if is_last_layer else ranker_flag_default

                if self.cut_half_cluster and layer > 0:
                    # A new dict per layer: ClusteringModel.train replaces config["kwargs"], and the
                    # caller's config (saved with the tree) must keep the configured n_clusters
                    kw = clustering_config["kwargs"]
                    clustering_config = {**clustering_config,
                                         "kwargs": {**kw, "n_clusters": max(2, int(kw.get("n_clusters", 2)) // 2)}}
                    
                for node_idx, (X_node, Y_node, Z_node, local_to_label_node, global_to_local_node) in enumerate(inputs):
                    node_start = time.perf_counter()
                    self.logger.info("Node started: layer=%d node=%d/%d rows=%d labels=%d leaf=%s",
                                     layer + 1, node_idx + 1, len(inputs), X_node.shape[0], Z_node.shape[0], is_last_layer)
                    
                    ml = MLModel(
                        clustering_config=clustering_config,
                        matcher_config=self.matcher_config,
                        ranker_config=self.ranker_config,
                        cur_config=self.cur_config,
                        min_leaf_size=self.min_leaf_size,
                        max_leaf_size=self.max_leaf_size,
                        ranker_every_layer= ranker_flag,
                        is_last_layer=is_last_layer,
                        layer=layer,
                        n_workers=self.n_workers,
                        train_rankers=self.train_rankers,
                    )

                    ml.train(
                        X_train=X_node,
                        Y_train=Y_node,
                        Z_train=Z_node,
                        local_to_global=local_to_label_node,
                        global_to_local=global_to_local_node
                    )

                    if ml.is_empty:  # only internal nodes can be empty; leaves use identity C
                        raise ValueError(
                            f"layer {layer} node with {Z_node.shape[0]} labels cannot be split into clusters of "
                            f">= min_leaf_size={self.min_leaf_size}; lower depth or min_leaf_size")

                    cluster_to_child = {}
                    if not is_last_layer:
                        self.logger.debug("Preparing child inputs: layer=%d node=%d", layer + 1, node_idx + 1)
                        raw_children = self.prepare_layer(
                            X=X_node,
                            Y=Y_node,
                            Z=Z_node,
                            C=ml.cluster_model.c_node,
                            fused_scores=ml.fused_scores,
                            local_to_global_idx=local_to_label_node
                        )
                        cluster_to_child = _accumulate_children(
                            raw_children,
                            start_idx=len(next_inputs),
                            next_inputs_list=next_inputs
                        )
                    
                    ml_path = _save_ml_for_layer(ml, layer)
                    self.logger.info("Node completed: layer=%d node=%d/%d elapsed=%.1fs",
                                     layer + 1, node_idx + 1, len(inputs), time.perf_counter() - node_start)
                    del ml
                    ml_list.append(ml_path)
                    layer_child_maps.append(cluster_to_child)

                inputs = _finalize_layer(layer, ml_list, next_inputs)
                del ml_list
                collect()
                child_index_map.append(layer_child_maps)
                self.logger.info("Layer completed: layer=%d/%d next_nodes=%d elapsed=%.1fs",
                                 layer + 1, self.layers, len(inputs), time.perf_counter() - layer_start)

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
                path_score: bool = False,
                scorer: str | None = None):
        if scorer not in (None, "ranker", "cosine"):
            raise ValueError(f"Unknown scorer: {scorer}")
        # path_score: leaf score x routing path probability (exp(path_logscore)), as XR-Linear does.
        # Leaf matcher probabilities are per-label sigmoids trained only against that leaf's labels, so
        # without it the beam's leaves are merged by max on scales that are not comparable.

        time_start_routing = time.perf_counter()

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
                alpha=alpha,
                scorer=scorer
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

        self.logger.info("Routing started: rows=%d beam=%d topk=%s topk_mode=%s scorer=%s alpha=%s path_score=%s",
                         n_queries, beam_size, topk, topk_mode, scorer or "auto", alpha, path_score)

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

                        x_next = augment_features(x_row, [float(p_child)], [sum_cs], [max_cs])

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

        time_start_ranking = time.perf_counter()
        self.logger.info("Routing completed: visited_leaves=%d elapsed=%.1fs",
                         len(pending_by_leaf), time_start_ranking - time_start_routing)

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
        n_labels_total = int(self.hmodel[0][0].label_embeddings.shape[0])  # root holds every label

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

        self.logger.info("Ranking completed: rows=%d scores=%d elapsed=%.1fs",
                         n_queries, scores_csr.nnz, time.perf_counter() - time_start_ranking)

        return out, scores_csr
