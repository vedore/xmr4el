import os
import json
import pickle
import numpy as np
import logging
import joblib
import torch
from abc import ABCMeta
from copy import deepcopy
from joblib import parallel_backend
from sklearn.cluster import KMeans
from typing import Any, Dict, Optional, Tuple, Counter, List
from numpy import ones, ndarray, asarray, argmax
from scipy.sparse import csr_matrix
from collections import defaultdict


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

        config_path = os.path.join(clustering_folder, "cluster_config.json")

        if not os.path.exists(config_path):
            config = {"type": "sklearnkmeans", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())

        cluster_type = config.get("type", None)
        assert (
            cluster_type is not None
        ), f"{clustering_folder} is not a valid clustering folder"
        assert cluster_type in cluster_dict, f"invalid cluster type {config['type']}"
        model = cluster_dict[cluster_type].load(clustering_folder, config["kwargs"])
        return cls(config, model)

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
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

        config = (
            deepcopy(config) if config is not None else {"type": "sklearnkmeans", "kwargs": {}}
        )
        # LOGGER.debug(f"Train Clustering with config: {json.dumps(config, indent=True)}")
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



class SklearnKMeans(ClusteringModel):
    """Simple KMeans"""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        """Save trained sklearn KMeans model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        """Load a saved sklearn KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnKMeans: The loaded object.
        """

        # LOGGER.info(f"Loading Sklearn Kmeans Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
        return model

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's KMeans Clustering.

        Returns:
            KMeans: Trained clustering.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnKMeans does not accept.
        """

        defaults = {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": "auto",
            "max_iter": 300,
            "tol": 0.0001,
            "verbose": 0,
            "random_state": None,
            "copy_x": True,
            "algorithm": "lloyd",
        }

        try:
            config = {**defaults, **config}
            model = KMeans(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnKMeans Clustering"
            )
        with parallel_backend("threading", n_jobs=-1):
            model.fit(trn_corpus)
        return cls(config, model)

    def predict(self, predict_input):
        """Predict an input.

        Args:
            corpus (str, list): List of strings to predict.

        Returns:
            numpy.ndarray: Matrix of features.
        """

        return self.model.predict(predict_input)

    def get_params(self):
        return self.model.get_params()

    def labels(self):
        return self.model.labels_


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

        # LOGGER.info(f"Loading Balanced KMeans Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

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
        max_leaf_size: Optional[int] = None,
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
        self._C_node: Optional[ndarray] = None
        self._model: Optional[ClusteringModel] = None
        self._cluster_to_labels: Optional[Dict[int, List[int]]] = None

    @property
    def c_node(self) -> Optional[ndarray]:
        """Sparse cluster assignment matrix."""
        return self._C_node
    
    @c_node.setter
    def c_node(self, value: ndarray) -> None:
        """Set the cluster assignment matrix."""
        self._C_node = value
        
    @property
    def model(self) -> Optional[ClusteringModel]:
        """Return the underlying clustering model."""
        return self._model
    
    @model.setter
    def model(self, value: ClusteringModel) -> None:
        """Set the underlying clustering model."""
        self._model = value
        
    @property
    def cluster_to_labels(self) -> Optional[Dict[int, List[int]]]:
        """Mapping from cluster id to list of label indices."""
        return self._cluster_to_labels
    
    @cluster_to_labels.setter
    def cluster_to_labels(self, value: Dict[int, List[int]]) -> None:
        """Set the cluster to label mapping."""
        self._cluster_to_labels = value
    
    @property
    def is_empty(self) -> bool:
        """Return ``True`` if clustering was not trained."""
        return self.c_node is None

    def save(self, save_dir: str) -> None:
        """Persist the clustering object to disk."""
        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        model = self.model

        if model is not None:
            model_path = os.path.join(save_dir, "clustering")

            if hasattr(model, "save") and callable(model.save):
                model.save(model_path)
            else:
                joblib.dump(model, f"{model_path}.joblib")

            state.pop("_model", None)

        # Save remaining state
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir: str) -> "Clustering":
        """Load a clustering object from ``load_dir``."""
        cluster_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(cluster_path), f"Clustering path {cluster_path} does not exist"

        with open(cluster_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "clustering")
        # A leaf too small to cluster saves no model (identity C only)
        if os.path.exists(model_path):
            setattr(model, "_model", ClusteringModel.load(model_path))
        
        return model
    
    def train(
        self,
        Z: np.ndarray,
        local_to_global_idx: List[int],
        min_leaf_size: int,
        max_leaf_size: Optional[int],
        clustering_config: Optional[Dict[str, any]],
        dtype: float
    ) -> None:
        """Train the clustering model and populate cluster assignments."""
        
        C_node, model = ClusteringTrainer.train(
            Z=Z,
            config=clustering_config,
            min_leaf_size=min_leaf_size,
            max_leaf_size=max_leaf_size,
            dtype=dtype,
        )
        
        if C_node is None or model is None:
            return
        
        self.c_node = C_node
        self.model = model
        
        self.cluster_to_labels = defaultdict(list)
        for local_idx in range(C_node.shape[0]):
            cluster_vector = (
                C_node[local_idx].toarray().ravel()
                if hasattr(C_node[local_idx], "toarray")
                else asarray(C_node[local_idx]).ravel()
            )
            cid = int(argmax(cluster_vector))
        
            gidx = local_to_global_idx[local_idx]
            self.cluster_to_labels[cid].append(int(gidx))
