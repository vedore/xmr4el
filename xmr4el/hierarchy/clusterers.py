import os
import json
import pickle
import numpy as np
import logging
import joblib
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
    """Balanced KMeans with gpu support"""

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

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to balanced KMeans.

        Returns:
            BalancedKMeans: Trained clustering.

        Raises:
            Exception: If `config` contains keyword arguments that the BalancedKMeans does not accept.
        """
        import torch
        from kmeans_pytorch import KMeans as PyTorchBalancedKMeans

        defaults = {
            "n_clusters": 8,
            "distance": "cosine",
            "tol": 1e-4,
            "tqdm_flag": logger.isEnabledFor(logging.DEBUG),
            "iter_limit": 400,
            "iter_k": None,
            "device": None,
            "gamma_for_soft_dtw": 0.001,
            "seed": 0,
        }

        try:
            config = {**defaults, **config}
            device = torch.device("cuda" if config["device"] == "gpu" and torch.cuda.is_available() else "cpu")
            model = PyTorchBalancedKMeans(n_clusters=config["n_clusters"], balanced=True, device=device)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for BalancedKMeans Clustering"
            )

        trn_corpus = torch.from_numpy(trn_corpus)
        
        # Check for zero vectors (norm == 0)
        norms = torch.norm(trn_corpus, dim=1)
        if (norms == 0).any():
            # You might want to remove or fix these vectors, e.g.:
            trn_corpus = trn_corpus[norms > 0]
        
        # kmeans_pytorch picks its initial centroids with np.random.choice; without this the
        # whole tree differs run to run and ablation deltas are unreadable.
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])

        cluster_labels = model.fit(X=trn_corpus, 
                                   distance=config["distance"], 
                                   tol=config["tol"], 
                                   tqdm_flag=config["tqdm_flag"],
                                   iter_limit=config["iter_limit"], 
                                   gamma_for_soft_dtw=config["gamma_for_soft_dtw"],
                                   iter_k=config["iter_k"]
                                   )
        cluster_labels = cluster_labels.cpu().numpy()
        model = {"cluster_labels": cluster_labels}
        return cls(config, model)

    def predict(self, predict_input):
        """Predict an input.

        Args:
            corpus (str, list): List of strings to predict.

        Returns:
            numpy.ndarray: Matrix of features.
        """

        return self.model.predict(predict_input)

    def labels(self):
        return self.model["cluster_labels"]
    
    # def centroids(self):
    #     return self.model["cluster_centers"]


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
