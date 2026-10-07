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
        assert (
            dimension_type is not None
        ), f"{dimension_folder} is not a valid vectorizer folder"
        assert (
            dimension_type in dimension_dict
        ), f"invalid vectorizer type {config['type']}"
        model = dimension_dict[dimension_type].load(dimension_folder)
        return cls(config, model)
    
    @classmethod
    def fit(cls, X_emb, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer. Default behavior is to use tfidf vectorizer with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Vectorizer: Trained vectorizer.
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
        """Reduce an corpus.

        Args:
            corpus (list or str): List of strings to vectorize or path to text file.
            **kwargs: Keyword arguments to pass to the trained vectorizer.

        Returns:
            numpy.ndarray or scipy.sparse.csr.csr_matrix: Matrix of features.
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
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer.
            model (sklearn.feature_extraction.text.TfidfVectorizer, optional): The trained tfidf vectorizer. Default is `None`.
        """

        self.config = config
        self.model = model

    def __del__(self):
        """Destruct self model instance"""
        self.model = None

    def save(self, save_dir):
        """Save trained sklearn Tfidf vectorizer to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "dimension_model.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a saved sklearn Tfidf vectorizer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            Tfidf: The loaded object.
        """

        dimension_path = os.path.join(load_dir, "dimension_model.pkl")
        assert os.path.exists(
            dimension_path
        ), f"vectorizer path {dimension_path} does not exist"

        with open(dimension_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def fit(cls, X_emb, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's TfidfVectorizer.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Tfidf: Trained vectorizer.

        Raises:
            Exception: If `config` contains keyword arguments that the tfidf vectorizer does not accept.
        """
        defaults = {
            "n_components": 1000, 
            "algorithm": 'randomized', 
            "n_iter": 5, 
            "n_oversamples": 10, 
            "power_iteration_normalizer": 'auto', 
            "random_state": None, 
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
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(X_emb)
        return cls(config, model)

    def transform(self, corpus):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        
        if self.model is None:
            return corpus
        
        return self.model.transform(corpus)
