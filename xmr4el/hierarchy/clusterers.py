import os
import json
import pickle
import numpy as np
import logging
import torch
from abc import ABCMeta
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Counter
from numpy import ones
from scipy.sparse import csr_matrix


cluster_dict = {}


class ClusterMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of ClusteringModel.
    Automatically registers each subclass in the cluster_dict.
    """

    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "ClusteringModel":
            cluster_dict[name.lower()] = new_cls
        return new_cls


class ClusteringModel(metaclass=ClusterMeta):
    """Wrapper to all clustering models"""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def save(self, clustering_folder):
        """Save clustering model to disk.

        Args:
            clustering_folder (str): Folder to save to.
        """

        os.makedirs(clustering_folder, exist_ok=True)
        with open(
            os.path.join(clustering_folder, "cluster_config.json"),
            "w",
            encoding="utf-8",
        ) as fout:
            fout.write(json.dumps(self.config))
        self.model.save(clustering_folder)

    @classmethod
    def load(cls, clustering_folder):
        """Load a saved clustering model from disk.

        Args:
            clustering_folder (str): Folder where `ClusteringModel` was saved to using `ClusteringModel.save`.

        Returns:
            ClusteringModel: The loaded object.
        """

        with open(os.path.join(clustering_folder, "cluster_config.json"), "r", encoding="utf-8") as fin:
            config = json.loads(fin.read())

        cluster_type = config.get("type", None)
        if cluster_type not in cluster_dict:
            raise ValueError(f"{clustering_folder}: invalid cluster type {cluster_type}")
        model = cluster_dict[cluster_type].load(clustering_folder, config["kwargs"])
        return cls(config, model)

    @classmethod
    def train(cls, trn_corpus, config, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific cluster
            class to use.
                Also contains keyword arguments to pass to the specified cluster. Default behavior is to use kmeans cluster with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            ClusteringModel: Trained cluster model.
        """

        config = deepcopy(config)
        cluster_type = config.get("type", None)
        assert (
            cluster_type is not None
        ), f"config {config} should contain a key 'type' for the cluster type"
        model = cluster_dict[cluster_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        config["kwargs"] = model.config
        return cls(config, model)

    def labels(self):
        return self.model.labels()



class BalancedKMeans(ClusteringModel):
    """PECOS-style balanced spherical k-means: recursive 2-means whose split is the n_l best-scoring points
    (sizes differ by at most 1 at each split), then joint balanced refinement over all k clusters."""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        """Save trained Balanced KMeans model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        """Load a saved Balanced KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            BalancedKMeans: The loaded object.
        """

        clustering_path = os.path.join(load_dir, "clustering.pkl")
        if not os.path.exists(clustering_path):
            raise FileNotFoundError(f"clustering path {clustering_path} does not exist")

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
        return model

    def labels(self):
        return self.model["cluster_labels"]

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        config = {"n_clusters": 8, "iter_limit": 20, "refine_iter": 20, "seed": 0, **config}
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
        return cls(config, {"cluster_labels": labels})


def _balanced_assign(S, cap, steps=300):
    """Assign row i to a column of score matrix S with exactly cap[j] rows per column, near the max-score assignment.
    Dual prices p are raised on over-full columns, then rows are taken greedily by S - p in descending order;
    on the BC5CDR root this matches scipy linprog's exact optimum (0.5584 mean cosine) at ~0.15 s vs ~38 s."""
    n, k = S.shape
    # Price loop in torch float64 (multithreaded, same values as numpy's float64 S - p).
    St, p, capt = torch.from_numpy(S).double(), torch.zeros(k, dtype=torch.float64), torch.from_numpy(cap).double()
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
        n_clusters = config["kwargs"]["n_clusters"]

        if n_points <= min_leaf_size:
            return None, None

        # Train clustering model
        clustering_model = ClusteringModel.train(Z, config, dtype)
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
