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
from scipy.sparse import hstack as sp_hstack, csr_matrix
from numpy import (
    asarray, int32, log, vstack as np_vstack
)
from sklearn.preprocessing import normalize
from collections import defaultdict
from uuid import uuid4
from pathlib import Path
from xmr4el.hierarchy.node import MLModel

def augment_features(X, selected, total, maximum):
    """Append routing probabilities and normalize identically during train and predict."""
    extra = csr_matrix(np_vstack([selected, total, maximum]).T)
    return normalize(sp_hstack([X, extra], format="csr"), norm="l2", axis=1)


class HierarchicalMLModel():
    """Loops MLModel"""
    LEAF_CANDIDATES = 100  # labels each visited leaf contributes (the cap on @cand)

    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 min_leaf_size=20,
                 max_leaf_size=None,
                 cut_half_cluster=False,
                 layer=1):
        
        self.logger = logging.getLogger(__name__)
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster = cut_half_cluster
        
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
        (X_aug, Y_node, Z_node, local_to_global_next, global_to_local_next, c)
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

            fused_c = fused_dense[mention_mask, :]
            feat_c = fused_c[:, c].ravel()
            feat_sum = fused_c.sum(axis=1).ravel()
            feat_max = fused_c.max(axis=1).ravel()
            
            X_aug = augment_features(X_node, feat_c, feat_sum, feat_max)

            Z_node = normalize(Z[local_idxs, :], norm="l2", axis=1)
            inputs.append((X_aug, Y_node, Z_node, local_to_global_next, global_to_local_next, c))

        return inputs
            
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
        Train multiple layers of MLModel; intermediate models are saved in a
        temporary folder which is automatically deleted at the end of training.
        """
        inputs = ((X_train, Y_train, Z_train, local_to_global, global_to_local),)
        clustering_config = deepcopy(self.clustering_config)
        last_layer_index = self.layers - 1

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
                        min_leaf_size=self.min_leaf_size,
                        max_leaf_size=self.max_leaf_size,
                        is_last_layer=is_last_layer,
                        layer=layer,
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
        
    
    def predict(self, X_query, beam_size: int = 5, topk: int = 0):
        """Score CSR (n_queries x n_labels), each row sorted by descending score.

        Beam search over the internal layers; every visited leaf contributes its LEAF_CANDIDATES labels with the
        highest matcher probability, scored matcher probability x routing path probability (XR-Linear). Leaf
        matcher probabilities are per-label sigmoids trained only against that leaf's labels, so the path
        probability is what makes leaves in one beam comparable. topk > 0 keeps each row's topk best labels."""
        time_start_routing = time.perf_counter()
        n_queries = X_query.shape[0]
        self.logger.info("Routing started: rows=%d beam=%d topk=%d", n_queries, beam_size, topk)

        # Beam entries of all queries, in query order then beam order; row i of X_beam belongs to entry i.
        # One matcher call and one augment_features call per layer instead of per query and beam entry.
        n_roots = len(self.hmodel[0])
        beam = [(qi, mi, 0.0) for qi in range(n_queries) for mi in range(n_roots)]
        X_beam = X_query[np.repeat(np.arange(n_queries), n_roots)]

        for layer in range(len(self.hmodel) - 1):
            ml_list = self.hmodel[layer]
            parents = asarray([e[1] for e in beam])
            cs_rows = [None] * len(beam)
            for parent_model_idx in np.unique(parents):
                rows = np.flatnonzero(parents == parent_model_idx)
                cs_batch = asarray(ml_list[parent_model_idx].matcher_model.predict_proba(X_beam[rows]))
                for r, cs in zip(rows, cs_batch):
                    cs_rows[r] = cs

            candidates_by_query = defaultdict(list)
            for row, ((qi, parent_model_idx, logscore), cs) in enumerate(zip(beam, cs_rows)):
                if cs.size == 0 or np.all(cs <= 0):
                    continue
                k = min(beam_size, cs.size)
                idx_top = np.argpartition(cs, cs.size - k)[-k:]
                idx_top = idx_top[np.argsort(-cs[idx_top])]
                parent_to_child = self.child_index_map[layer][parent_model_idx]
                sum_cs = float(cs.sum()); max_cs = float(cs.max())
                for c in idx_top:
                    child_idx = parent_to_child.get(int(c))
                    if child_idx is None:
                        continue
                    p_child = float(cs[c])
                    candidates_by_query[qi].append((int(child_idx), logscore + float(log(max(p_child, 1e-9))),
                                                    row, p_child, sum_cs, max_cs))

            beam, rows, selected, total, maximum = [], [], [], [], []
            for qi, candidates in candidates_by_query.items():
                candidates.sort(key=lambda t: -t[1])
                for child_idx, logscore_next, row, p_child, sum_cs, max_cs in candidates[:beam_size]:
                    beam.append((qi, child_idx, logscore_next))
                    rows.append(row); selected.append(p_child); total.append(sum_cs); maximum.append(max_cs)
            if not beam:
                break
            X_beam = augment_features(X_beam[rows], selected, total, maximum)

        time_start_ranking = time.perf_counter()
        leaves = asarray([e[1] for e in beam], dtype=int)
        self.logger.info("Routing completed: visited_leaves=%d elapsed=%.1fs",
                         np.unique(leaves).size, time_start_ranking - time_start_routing)

        qis, cols, vals = [np.empty(0, dtype=int)], [np.empty(0, dtype=int)], [np.empty(0)]
        for leaf_idx in np.unique(leaves):
            rows = np.flatnonzero(leaves == leaf_idx)
            leaf = self.hmodel[-1][leaf_idx]
            P = asarray(leaf.matcher_model.predict_proba(X_beam[rows]), dtype=float)  # column j = local label j
            k = min(self.LEAF_CANDIDATES, P.shape[1])
            top = np.argpartition(P, P.shape[1] - k, axis=1)[:, -k:]
            path = np.exp([beam[r][2] for r in rows])
            qis.append(np.repeat([beam[r][0] for r in rows], k))
            cols.append(asarray(leaf.local_to_global_idx)[top].ravel())
            vals.append((np.take_along_axis(P, top, axis=1) * path[:, None]).ravel())
        qis, cols, vals = np.concatenate(qis), np.concatenate(cols), np.concatenate(vals)

        # Each label sits in one leaf and a query reaches each leaf by one path: no (query, label) repeats
        order = np.lexsort((-vals, qis))
        qis, cols, vals = qis[order], cols[order], vals[order]
        indptr = np.concatenate([[0], np.cumsum(np.bincount(qis, minlength=n_queries))])
        if topk > 0:
            keep = np.arange(qis.size) - indptr[qis] < topk
            qis, cols, vals = qis[keep], cols[keep], vals[keep]
            indptr = np.concatenate([[0], np.cumsum(np.bincount(qis, minlength=n_queries))])
        n_labels = len(self.hmodel[0][0].local_to_global_idx)  # root holds every label
        scores = csr_matrix((vals, cols.astype(int32), indptr), shape=(n_queries, n_labels))

        self.logger.info("Ranking completed: rows=%d scores=%d elapsed=%.1fs",
                         n_queries, scores.nnz, time.perf_counter() - time_start_ranking)
        return scores
