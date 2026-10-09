import json
import os
import pickle
import logging
import numpy as np
from abc import ABCMeta
from copy import deepcopy
from sklearn.decomposition import TruncatedSVD


dimension_dict = {}


class DimensionModelMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of Dimension Models.
    Automatically registers each subclass in the dimension_dict.
    """

    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "DimensionModel":
            dimension_dict[name.lower()] = new_cls
        return new_cls

class DimensionModel(metaclass=DimensionModelMeta):
    
    def __init__(self, config, model):
        
        self.config = config
        self.model = model
        
    def save(self, dimension_folder):
        """Save trained dimension model to disk.

        Args:
            dimension_folder (str): Folder to save to.
        """

        os.makedirs(dimension_folder, exist_ok=True)
        with open(
            os.path.join(dimension_folder, "dim_config.json"), "w", encoding="utf-8"
        ) as fout:
            fout.write(json.dumps(self.config))
        self.model.save(dimension_folder)

    @classmethod
    def load(cls, dimension_folder):
        """Load a saved dimension model from disk.

        Args:
            dimension_folder (str): Folder where `DimensionModel` was saved to using `DimensionModel.save`.

        Returns:
            DimensionModel: The loaded object.
        """

        config_path = os.path.join(dimension_folder, "dim_config.json")

        if not os.path.exists(config_path):
            config = {"type": "sklearntruncatedsvd", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())

        dimension_type = config.get("type", None)
        if dimension_type not in dimension_dict:
            raise ValueError(f"{dimension_folder}: invalid dimension type {dimension_type}")
        model = dimension_dict[dimension_type].load(dimension_folder)
        return cls(config, model)
    
    @classmethod
    def fit(cls, X_emb, config=None, dtype=np.float32):
        """Fit a dimension model on X_emb (n x d, sparse or dense).

        Args:
            X_emb: Features to reduce.
            config (dict, optional): `"type"` (lower-cased class name) and `"kwargs"` for it. Default: truncated SVD
                with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            DimensionModel: Fitted model.
        """

        config = deepcopy(config) if config is not None else {"type": "sklearntruncatedsvd", "kwargs": {}}
        dimension_type = config.get("type", None)
        assert (
            dimension_type is not None
        ), f"config {config} should contain a key 'type' for the dimension type"
        model = dimension_dict[dimension_type].fit(
            X_emb, config=config["kwargs"], dtype=dtype
        )
        config["kwargs"] = model.config
        return cls(config, model)
    
    def transform(self, x_emb, **kwargs):
        """Reduce x_emb (n x d) to (n x n_components); unchanged when the fit skipped SVD.

        Returns:
            numpy.ndarray, or x_emb itself when there is no model.
        """

        if isinstance(x_emb, str) and self.config["type"] != "tfidf":
            raise ValueError(
                "Iterable over raw text expected for vectorizer other than tfidf."
            )
        
        if self.model is None:
            return x_emb
            
        return self.model.transform(x_emb, **kwargs)
    
class SklearnTruncatedSVD(DimensionModel):
    
    def __init__(self, config=None, model=None):
        """Initialization

        Args:
            config (dict): TruncatedSVD keyword arguments.
            model (sklearn.decomposition.TruncatedSVD, optional): The fitted SVD; None = features kept unreduced.
        """

        self.config = config
        self.model = model

    def __del__(self):
        """Destruct self model instance"""
        self.model = None

    def save(self, save_dir):
        """Save the fitted SVD to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "dimension_model.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a saved SVD from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnTruncatedSVD: The loaded object.
        """

        dimension_path = os.path.join(load_dir, "dimension_model.pkl")
        if not os.path.exists(dimension_path):
            raise FileNotFoundError(f"dimension model path {dimension_path} does not exist")

        with open(dimension_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def fit(cls, X_emb, config={}, dtype=np.float32):
        """Fit TruncatedSVD on X_emb (n x d, sparse or dense).

        Args:
            X_emb: Features to reduce.
            config (dict): Keyword arguments for sklearn's TruncatedSVD (defaults below).
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            SklearnTruncatedSVD: Fitted model; model None (no reduction) when n_components > d.

        Raises:
            Exception: If `config` contains keyword arguments that TruncatedSVD does not accept.
        """
        defaults = {
            "n_components": 1000, 
            "algorithm": 'randomized', 
            "n_iter": 5, 
            "n_oversamples": 10, 
            "power_iteration_normalizer": 'auto', 
            "random_state": 0,  # randomized SVD: seeded so an unset config is still repeatable
            "tol": 0.0
        }

        try:
            config = {**defaults, **config}
            
            X_n_features = X_emb.shape[1]
            config_n_components = config["n_components"]
            
            if X_n_features < config_n_components:
                logging.getLogger(__name__).warning(
                    "Skipping SVD: n_components=%d exceeds n_features=%d; retaining unreduced features",
                    config_n_components, X_n_features)
                return cls(config, None)
            
            model = TruncatedSVD(**config)
        except TypeError:
            raise Exception(
                f"dimension config {config} contains unexpected keyword arguments for TruncatedSVD"
            )
        model.fit(X_emb)
        return cls(config, model)

    def transform(self, corpus):
        """Reduce corpus (n x d features) to (n x n_components), or return it unchanged without a model.

        Returns:
            numpy.ndarray: Reduced features.
        """
        
        if self.model is None:
            return corpus
        
        return self.model.transform(corpus)
