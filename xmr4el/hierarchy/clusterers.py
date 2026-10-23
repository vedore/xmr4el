import os
import json
import numbers
import pickle
import numpy as np
import logging
import torch
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Counter
from numpy import ones
from scipy.sparse import csr_matrix


class ClusteringModel:
    """Configured balanced k-means assignments; no runtime backend registry."""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    @staticmethod
    def check_config(config):
        if config.get("type") != "balancedkmeans":
            raise ValueError(f"invalid cluster type {config.get('type')!r}; expected 'balancedkmeans'")
        BalancedKMeans.check_config(config.get("kwargs", {}))

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "cluster_config.json"), "w", encoding="utf-8") as fout:
            json.dump(self.config, fout)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, "cluster_config.json"), encoding="utf-8") as fin:
            config = json.load(fin)
        cls.check_config(config)
        with open(os.path.join(load_dir, "clustering.pkl"), "rb") as fin:
            return cls(config, pickle.load(fin))

    @classmethod
    def train(cls, trn_corpus, config):
        cls.check_config(config)
        config = deepcopy(config)
        config["kwargs"] = {**BalancedKMeans.DEFAULTS, **config.get("kwargs", {})}
        return cls(config, BalancedKMeans.train(trn_corpus, config["kwargs"]))

    def labels(self):
        return self.model


class BalancedKMeans:
    """Balanced spherical k-means: recursive 2-means followed by joint balanced refinement."""

    DEFAULTS = {"n_clusters": 8, "iter_limit": 20, "refine_iter": 20, "seed": 0}

    @staticmethod
    def check_config(config):
        """Raise ValueError unless n_clusters and iter_limit are ints >= 1, refine_iter an int >= 0 and seed an int
        (keys left out take the defaults)."""
        lows = {"n_clusters": 1, "iter_limit": 1, "refine_iter": 0, "seed": 0}
        bad = [f"{k} {config[k]!r} (int >= {lo})" for k, lo in lows.items() if k in config and not (
            isinstance(config[k], numbers.Integral) and not isinstance(config[k], bool) and config[k] >= lo)]
        if bad:
            raise ValueError("balancedkmeans: " + "; ".join(bad))

    @classmethod
    def train(cls, trn_corpus, config=None):
        config = config or {}
        cls.check_config(config)
        config = {**cls.DEFAULTS, **config}
        Z = np.asarray(trn_corpus, dtype=np.float32)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)  # zero rows stay (one label per row)
        rs = np.random.RandomState(config["seed"])
        labels = np.empty(len(Z), dtype=np.int64)
        stack, next_id = [(np.arange(len(Z)), config["n_clusters"])], 0
        while stack:
            idx, k = stack.pop()
            if k == 1 or len(idx) < 2:
                labels[idx] = next_id
                next_id += 1
                continue
            k_l = k // 2
            n_l = round(len(idx) * k_l / k)
            X = Z[idx]
            c = X[rs.choice(len(idx), 2, replace=False)]
            left = None
            for _ in range(config["iter_limit"]):
                new_left = np.zeros(len(idx), dtype=bool)
                new_left[np.argsort(-(X @ (c[0] - c[1])), kind="stable")[:n_l]] = True
                if left is not None and (new_left == left).all():
                    break
                left = new_left
                c = np.vstack([X[left].mean(axis=0), X[~left].mean(axis=0)])
                c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
            stack += [(idx[~left], k - k_l), (idx[left], k_l)]

        # Joint refinement: the greedy splits never move a point across an earlier split. Each round reassigns
        # all points to the k centroids under the bisect sizes (balanced assignment), until nothing changes.
        cap = np.bincount(labels)
        for _ in range(config["refine_iter"]):
            c = np.vstack([Z[labels == j].mean(axis=0) for j in range(len(cap))])
            c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
            new = _balanced_assign(Z @ c.T, cap)
            if (new == labels).all():
                break
            labels = new
        return labels


def _balanced_assign(S, cap, steps=300):
    """Assign row i to a column of score matrix S with exactly cap[j] rows per column, near the max-score assignment.
    Dual prices p are raised on over-full columns, then rows are taken greedily by S - p in descending order;
    on the BC5CDR root this matched (float64 prices) scipy linprog's exact optimum (0.5584 mean cosine) at ~0.15 s vs ~38 s."""
    n, k = S.shape
    # Price loop in torch float32 (multithreaded, same values as numpy's float32 S - p).
    St, p, capt = torch.from_numpy(S).float(), torch.zeros(k, dtype=torch.float32), torch.from_numpy(cap).float()
    lr = float(S.max() - S.min()) / n
    for _ in range(steps):
        p += lr * (torch.bincount((St - p).argmax(dim=1), minlength=k) - capt)
    V = (St - p).numpy()
    # The greedy over all (row, column) entries by V descending equals row-proposing deferred acceptance with
    # columns keeping their top-V rows (common weights -> unique stable matching), vectorized per round.
    pref = np.argsort(-V, axis=1, kind="stable")
    nxt, labels, free = np.zeros(n, dtype=np.int64), np.full(n, -1), np.arange(n)
    while free.size:
        labels[free] = pref[free, nxt[free]]
        nxt[free] += 1
        rows = np.flatnonzero(np.isin(labels, np.unique(labels[free])))
        o = np.lexsort((rows, -V[rows, labels[rows]], labels[rows]))  # column, V desc, row (the greedy's tie order)
        r, c = rows[o], labels[rows[o]]
        free = r[np.arange(len(r)) - np.searchsorted(c, c) >= cap[c]]
        labels[free] = -1
    return labels


logger = logging.getLogger(__name__)


class ClusteringTrainer:
    """Utility class providing the clustering training routine."""
    
    @staticmethod
    def train(
        Z: np.ndarray,
        config: Dict[str, Any],
        min_leaf_size: int = 20,
        dtype: Any = np.float32,
    ) -> Tuple[Optional[csr_matrix], Optional[ClusteringModel]]:
        """Train clustering without recursive partitioning."""
        
        n_points = Z.shape[0]

        if n_points <= min_leaf_size:
            return None, None

        # Train clustering model
        clustering_model = ClusteringModel.train(Z, config)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)

        # Filter out invalid clusters
        valid_clusters = [cid for cid, cnt in cluster_counts.items() if cnt >= min_leaf_size]
        if len(valid_clusters) <= 1:
            return None, None

        assert len(cluster_labels) == n_points, (
            f"clustering returned {len(cluster_labels)} labels for {n_points} points; "
            "C_node rows would be misaligned with Z"
        )

        # Points in undersized clusters are reassigned to the nearest valid centroid.
        # Dropping them would leave an all-zero C_node row, i.e. a label that can never be
        # proposed (tree.py) and never reaches the next layer (tree.py).
        cluster_to_col = {cid: i for i, cid in enumerate(valid_clusters)}
        cols = np.array([cluster_to_col.get(int(c), -1) for c in cluster_labels])

        orphans = cols < 0
        n_orphans = int(orphans.sum())
        if n_orphans:
            Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
            centroids = np.vstack([Zn[cluster_labels == cid].mean(axis=0) for cid in valid_clusters])
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
            cols[orphans] = np.argmax(Zn[orphans] @ centroids.T, axis=1)

        sizes = Counter(cols.tolist()).values()
        logger.info("Clustering completed: raw_clusters=%d valid_clusters=%d labels=%d reassigned=%d size_min=%d size_max=%d",
                    len(cluster_counts), len(valid_clusters), n_points, n_orphans, min(sizes), max(sizes))
        logger.debug("Cluster sizes: %s", sorted(sizes, reverse=True))

        C_node = csr_matrix(
            (ones(n_points, dtype=dtype), (np.arange(n_points), cols)),
            shape=(n_points, len(valid_clusters)),
        )
        return C_node, clustering_model




class Clustering:
    """Pipeline that encapsulates label clustering."""

    def __init__(
        self,
    ) -> None:
        """Initialize the clustering pipeline."""
        self.c_node: Optional[csr_matrix] = None  # C, L x K (one 1 per row)
        self.model: Optional[ClusteringModel] = None

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if clustering was not trained."""
        return self.c_node is None

    def save(self, save_dir: str) -> None:
        """Persist the clustering object to disk."""
        model = self.model
        if model is not None and not callable(getattr(model, "save", None)):
            raise TypeError(f"clustering model ({type(model).__name__}) has no save()")
        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        if model is not None:
            model.save(os.path.join(save_dir, "clustering"))
            state.pop("model", None)

        # Save remaining state
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir: str) -> "Clustering":
        """Load a clustering object from ``load_dir``."""
        cluster_path = os.path.join(load_dir, "clustering.pkl")
        if not os.path.exists(cluster_path):
            raise FileNotFoundError(f"Clustering path {cluster_path} does not exist")

        with open(cluster_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "clustering")
        # A leaf too small to cluster saves no model (identity C only)
        if os.path.exists(model_path):
            model.model = ClusteringModel.load(model_path)
        
        return model
    
    def train(
        self,
        Z: np.ndarray,
        min_leaf_size: int,
        clustering_config: Optional[Dict[str, any]],
        dtype: float
    ) -> None:
        """Cluster Z (n_labels x d) into `c_node` (n_labels x K); leaves both None when Z is too small."""
        
        C_node, model = ClusteringTrainer.train(
            Z=Z,
            config=clustering_config,
            min_leaf_size=min_leaf_size,
            dtype=dtype,
        )
        
        if C_node is None or model is None:
            return
        
        self.c_node = C_node
        self.model = model
