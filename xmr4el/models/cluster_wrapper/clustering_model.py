import importlib
import os
import json
import pickle
import pkgutil
import sys
import torch

import numpy as np

from abc import ABCMeta
from joblib import parallel_backend
from sklearn.cluster import KMeans
from kmeans_pytorch import KMeans as PyTorchBalancedKMeans


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

    @classmethod
    def load_subclasses(cls, package_name):
        """Dynamically imports all modules in the package to register subclasses."""

        package = sys.modules[package_name]
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package_name}.{modname}")


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
            config if config is not None else {"type": "sklearnkmeans", "kwargs": {}}
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

    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `cluster_config_path` (path to a json file) or `cluster_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """

        if args.cluster_config_path is not None:
            with open(args.cluster_config_path, "r", encoding="utf-8") as fin:
                cluster_config_json = fin.read()
        else:
            cluster_config_json = args.cluster_config_json

        try:
            cluster_config = json.loads(cluster_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                f"Failed to load clustering config json from {cluster_config_json} ({jex})"
            )
        return cluster_config


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
        defaults = {
            "n_clusters": 8,
            "distance": "cosine",
            "tol": 1e-4,
            "tqdm_flag": True,
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

        # print(config["n_clusters"])
        # print(type(trn_corpus))
        trn_corpus = torch.from_numpy(trn_corpus)
        
        # Check for zero vectors (norm == 0)
        norms = torch.norm(trn_corpus, dim=1)
        if (norms == 0).any():
            # print(f"Warning: Found {torch.sum(norms == 0).item()} zero vectors in input!")
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