import os
import logging
import time
import numpy as np
import pickle
from numpy import array
from joblib import Parallel, delayed
from scipy.sparse import hstack, csr_matrix, issparse
from typing import Dict, Optional, Tuple
from xmr4el.learning.classifiers import ClassifierModel


random_seed = 0

logger = logging.getLogger(__name__)

def ranker_input(X, label_embeddings):
    """Append label features, preserving sparse training and dense scoring inputs."""
    labels = np.asarray(label_embeddings)
    if labels.ndim == 1:
        if issparse(X):
            n = X.shape[0]
            labels = csr_matrix((np.ones(n), (np.arange(n), np.zeros(n))), shape=(n, 1)).dot(
                csr_matrix(labels.reshape(1, -1))
            )
        else:
            labels = np.tile(labels, (X.shape[0], 1))
    return hstack([X, labels], format="csr") if issparse(X) else np.hstack([X, labels])

class RankerTrainer:
    """Training routines for per-label rankers."""
    

        
    @staticmethod
    def _width_match(vec: np.ndarray, D: int) -> np.ndarray:
        if vec.shape[0] >= D:
            return vec[:D]
        out = np.zeros(D, dtype=vec.dtype)
        out[:vec.shape[0]] = vec
        return out
    
    @staticmethod
    def _row_norms(X) -> np.ndarray:
        """Sparse/dense row L2 norms, as 1-D np.array of shape [n_rows]."""
        try:
            if issparse(X):
                return np.sqrt(X.multiply(X).sum(axis=1)).A1
        except Exception:
            pass
        # dense fallback
        return np.linalg.norm(X, axis=1)


    
    @staticmethod
    def _select_negatives_curriculum(
        *,
        epoch: int,
        positive_positions: np.ndarray,   # positions within cluster
        neg_positions_all: np.ndarray,    # positions within cluster
        cos_scores_neg: np.ndarray,       # cosine scores aligned with neg_positions_all
        ip_scores_neg: np.ndarray,        # inner-product scores aligned with neg_positions_all
        E_warm: int,
        ratios_warm: tuple,               # (rand, proto)
        ratios_hard: tuple,               # (ip, proto, rand)
        neg_mult: int,
        rng: np.random.RandomState,
        cluster_size: int,
    ) -> np.ndarray:
        """
        Returns selected NEGATIVE positions (indices into X_cluster).
        Never touches X_cluster; operates purely on positions & scores.
        """
        
        n_pos = positive_positions.size
        n_neg_avail = neg_positions_all.size
        if n_pos == 0 or n_neg_avail == 0:
            return array([], dtype=int)

        N_neg = min(neg_mult * n_pos, n_neg_avail)
        if N_neg <= 0:
            return array([], dtype=int)

        def topk_from(scores: np.ndarray, pool: np.ndarray, want: int) -> np.ndarray:
            if want <= 0:
                return array([], dtype=int)
            k_pre = min(max(1, 3 * want), pool.size)
            idx = np.argpartition(-scores, k_pre - 1)[:k_pre]
            idx = idx[np.argsort(-scores[idx])]
            return pool[idx][:want]

        if epoch <= E_warm:
            r_rand, r_proto = ratios_warm
            n_rand  = int(round(r_rand  * N_neg))
            n_proto = N_neg - n_rand

            rand_pick  = rng.choice(neg_positions_all, size=min(n_rand, n_neg_avail), replace=False) if n_rand > 0 else array([], dtype=int)
            proto_pick = topk_from(cos_scores_neg, neg_positions_all, n_proto)
            neg_positions = np.unique(np.concatenate([rand_pick, proto_pick]))
            return neg_positions[:N_neg]

        # hard phase
        r_ip, r_proto, r_rand = ratios_hard
        n_ip   = int(round(r_ip   * N_neg))
        n_proto= int(round(r_proto* N_neg))
        n_rand = N_neg - n_ip - n_proto

        ip_pick    = topk_from(ip_scores_neg,  neg_positions_all, n_ip)
        proto_pick = topk_from(cos_scores_neg, neg_positions_all, n_proto)

        taken = np.zeros(cluster_size, dtype=bool)
        taken[ip_pick] = True
        taken[proto_pick] = True
        # rest negatives = all negatives not yet taken
        rest_mask = ~taken
        # ensure positives not considered in rest
        pos_mask = np.zeros(cluster_size, dtype=bool)
        pos_mask[positive_positions] = True
        rest = np.where(rest_mask & (~pos_mask))[0]
        # intersect rest with neg_positions_all to avoid pulling out-of-cluster ids
        # (rest already cluster-local; intersect for safety)
        if rest.size > 0:
            # keep order from 'rest'
            in_neg = np.intersect1d(rest, neg_positions_all, assume_unique=False)
        else:
            in_neg = array([], dtype=int)

        rand_pick = rng.choice(in_neg, size=min(max(0, n_rand), in_neg.size), replace=False) if (n_rand > 0 and in_neg.size > 0) else array([], dtype=int)

        neg_positions = np.unique(np.concatenate([ip_pick, proto_pick, rand_pick]))
        return neg_positions[:N_neg]

    @staticmethod
    def process_label_incremental(
    global_idx: int,
    X_cluster,                      # dense np.ndarray or csr_matrix
    Y_col: np.ndarray,
    label_emb: np.ndarray,
    candidate_indices: np.ndarray,
    config: Dict,
    cur_config: Dict,
    epoch: int,
    existing: Optional["ClassifierModel"],
    ) -> Tuple[int, Optional["ClassifierModel"]]:

        positive_global = Y_col.nonzero()[0]
        if positive_global.size == 0:
            return (global_idx, existing)

        pos_mask = np.isin(candidate_indices, positive_global)
        positive_positions = np.where(pos_mask)[0]
        n_pos = positive_positions.size
        if n_pos < 2:
            return (global_idx, existing)

        neg_mask = ~pos_mask
        neg_positions_all = np.where(neg_mask)[0]
        if neg_positions_all.size == 0:
            return (global_idx, existing)

        # width-match label embedding
        n_feats = X_cluster.shape[1]
        base_emb = RankerTrainer._width_match(label_emb, n_feats)

        # compute scores ONCE over all rows (no copies of X_cluster)
        # inner-product scores:
        dot_scores_all = (X_cluster @ base_emb)
        dot_scores_all = np.asarray(dot_scores_all).ravel()

        # cosine scores:
        norms = RankerTrainer._row_norms(X_cluster) + 1e-12
        cos_scores_all = dot_scores_all / norms

        # slice negative scores aligned with neg_positions_all
        cos_scores_neg = cos_scores_all[neg_positions_all]
        ip_scores_neg  = dot_scores_all[neg_positions_all]

        if cur_config is None:
            cur_config = {
                "E_warm": 3,
                "ratios_warm": (0.70, 0.30),
                "ratios_hard": (0.60, 0.30, 0.10),
                "neg_mult": 15,
                "seed": config["kwargs"]["random_state"],
            }
        
        rng = np.random.RandomState(cur_config["seed"] + epoch)

        # select negatives purely by positions & scores (no X_cluster)
        neg_positions = RankerTrainer._select_negatives_curriculum(
            epoch=epoch,
            positive_positions=positive_positions,
            neg_positions_all=neg_positions_all,
            cos_scores_neg=cos_scores_neg,
            ip_scores_neg=ip_scores_neg,
            E_warm=cur_config["E_warm"],
            ratios_warm=tuple(cur_config["ratios_warm"]),
            ratios_hard=tuple(cur_config["ratios_hard"]),
            neg_mult=cur_config["neg_mult"],
            rng=rng,
            cluster_size=X_cluster.shape[0],
        )

        n_neg = neg_positions.size
        if n_neg < 2:
            return (global_idx, existing)

        # build training slice
        selected_positions = np.concatenate([positive_positions, neg_positions])
        y = np.zeros(selected_positions.size, dtype=np.int8)
        y[:n_pos] = 1
        X_valid = X_cluster[selected_positions]

        X_combined = ranker_input(X_valid, label_emb)

        # light shuffle helps SGD
        perm = np.arange(X_combined.shape[0])
        np.random.shuffle(perm)
        X_combined = X_combined[perm]
        y = y[perm]

        # ensure/init incremental model
        model = existing
        if model is None:
            mtype = config.get("type", "").lower()
            if mtype not in ("sklearnsgdclassifier",):
                raise RuntimeError(f"Incremental ranker requires partial_fit model; got type={mtype}")
            model = ClassifierModel.init_model(config)

        if not model.supports_partial_fit():
            raise RuntimeError(f"Model {type(model.model).__name__} does not support partial_fit")

        model.partial_fit(X=X_combined, Y=y, classes=array([0, 1]), dtype=np.int32)

        logger.debug("Ranker label: epoch=%d label=%d pos=%d neg=%d cluster_rows=%d",
                     epoch, global_idx, n_pos, n_neg, X_cluster.shape[0])

        return (global_idx, model)

    @staticmethod
    def train(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        M_TFN: np.ndarray,
        M_MAN: Optional[np.ndarray],
        cluster_labels: np.ndarray,
        config: Dict,
        cur_config: Dict,
        local_to_global_idx: np.ndarray,
        n_label_workers: int = -1,
        parallel_backend: str = "threading",
        n_epochs: int = 3,
        
    ) -> Dict[int, "ClassifierModel"]:
        """Train/Update ranker models for all labels with a curriculum using partial_fit."""
        
        if M_MAN is None:
            M_bar = (M_TFN > 0).astype(int)
        else:
            M_bar = ((M_TFN + M_MAN) > 0).astype(int)

        unique_clusters = np.unique(cluster_labels)
        labels_by_cluster: Dict[int, list] = {
            int(c): np.where(cluster_labels == c)[0].tolist()
            for c in unique_clusters
        }

        cluster_mentions = {c: M_bar[:, c].nonzero()[0] for c in labels_by_cluster.keys()}

        # model registry (kept in memory across epochs)
        ranker_models: Dict[int, "ClassifierModel"] = {}

        for epoch in range(1, n_epochs + 1):
            epoch_start = time.perf_counter()
            tasks = []
            for cluster_idx, label_list in labels_by_cluster.items():
                candidate_indices = cluster_mentions.get(cluster_idx, array([], dtype=int))
                if candidate_indices.size == 0:
                    continue
                X_cluster = X[candidate_indices]
                for local_idx in label_list:
                    gid = int(local_to_global_idx[local_idx])
                    label_emb = Z[local_idx]
                    Y_col = Y[:, local_idx]
                    existing = ranker_models.get(gid, None)
                    tasks.append((gid, X_cluster, Y_col, label_emb, candidate_indices, existing))

            results = Parallel(n_jobs=n_label_workers, backend=parallel_backend, verbose=0)(
                delayed(RankerTrainer.process_label_incremental)(
                    gid, X_c, Y_col, z, cand_idx, config, cur_config, epoch, existing
                )
                for (gid, X_c, Y_col, z, cand_idx, existing) in tasks
            )

            ranker_models.update({gid: model for gid, model in results if model is not None})
            logger.info("Ranker epoch completed: epoch=%d/%d trained_labels=%d skipped_labels=%d elapsed=%.1fs",
                        epoch, n_epochs, len(ranker_models), sum(model is None for _, model in results),
                        time.perf_counter() - epoch_start)

        return ranker_models



class Ranker:
    """Wrapper around a collection of per-label ranker models."""

    def __init__(
        self,
        temp_dir: str = "./tmp",
    ) -> None:        
        self._ranker_models: Optional[Dict[int, ClassifierModel]] = None
        
    @property
    def model_dict(self) -> Optional[Dict[int, ClassifierModel]]:
        """Return the dictionary of trained ranker models."""
        return self._ranker_models
    
    @model_dict.setter
    def model_dict(self, value: Dict[int, ClassifierModel]) -> None:
        """Set the internal ranker model dictionary."""
        self._ranker_models = value
        
    def save(self, save_dir: str) -> None:
        """Persist ranker state and models to ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)
        
        state = self.__dict__.copy()
            
        model_dict = self.model_dict or {}
        
        for idx, model in model_dict.items():
            idx_folder = os.path.join(save_dir, f"{idx}")
            model.save(idx_folder)
        
        state.pop("_ranker_models", None)
        state.pop("memory", None)
        state.pop("_cached_process_label", None)
        
        with open(os.path.join(save_dir, "ranker.pkl"), "wb") as fout:
            pickle.dump(state, fout)
        
    @classmethod
        # Load the pickled state
    def load(cls, load_dir: str) -> "Ranker":
        """Load ranker state and model dictionary from ``load_dir``."""
        with open(os.path.join(load_dir, "ranker.pkl"), "rb") as fin:
            state = pickle.load(fin)
        
        state.pop("model_dict", None)
        
        model = cls()
        model.__dict__.update(state)

        # Load each individual model from its directory
        model_dict: Dict[int, ClassifierModel] = {}
        for folder in os.listdir(load_dir):
            folder_path = os.path.join(load_dir, folder)
            if folder.isdigit():
                ranker_model = ClassifierModel.load(folder_path)
                model_dict[int(folder)] = ranker_model

        setattr(model, "_ranker_models", model_dict)
        return model
        
    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        M_TFN: np.ndarray,
        M_MAN: Optional[np.ndarray],
        cluster_labels: np.ndarray,
        local_to_global_idx: np.ndarray,
        layer: int,
        n_label_workers: int = 8,
        ranker_config: Optional[Dict] = None,
        cur_config: Optional[Dict] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Train ranker models for each label cluster."""
        n_epochs = (ranker_config or {}).get("n_epochs", 3)
        
        ranker_models = RankerTrainer.train(
            X=X,
            Y=Y,
            Z=Z,
            M_TFN=M_TFN,
            M_MAN=M_MAN,
            cluster_labels=cluster_labels,
            config=ranker_config,
            cur_config=cur_config,
            local_to_global_idx=local_to_global_idx,
            n_label_workers=n_label_workers,
            parallel_backend="threading",
            n_epochs=n_epochs
        )

        self.model_dict = ranker_models
