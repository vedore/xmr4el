"""Truncated SVD, retaining sparse features when reduction is unnecessary."""
import json
import logging
import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD


class DimensionModel:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    @staticmethod
    def check_config(config):
        if config.get("type") != "sklearntruncatedsvd":
            raise ValueError(f"invalid dimension type {config.get('type')!r}; expected 'sklearntruncatedsvd'")

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "dim_config.json"), "w", encoding="utf-8") as fout:
            json.dump(self.config, fout)
        with open(os.path.join(save_dir, "dimension_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, "dim_config.json"), encoding="utf-8") as fin:
            config = json.load(fin)
        cls.check_config(config)
        with open(os.path.join(load_dir, "dimension_model.pkl"), "rb") as fin:
            return cls(config, pickle.load(fin))

    @classmethod
    def fit(cls, X_emb, config=None):
        config = config if config is not None else {"type": "sklearntruncatedsvd", "kwargs": {}}
        cls.check_config(config)
        defaults = {
            "n_components": 1000,
            "algorithm": 'randomized',
            "n_iter": 5,
            "n_oversamples": 10,
            "power_iteration_normalizer": 'auto',
            "random_state": 0,  # randomized SVD: seeded so an unset config is still repeatable
            "tol": 0.0
        }

        kwargs = {**defaults, **config.get("kwargs", {})}
        model = TruncatedSVD(**kwargs)
        if X_emb.shape[1] < kwargs["n_components"]:
            logging.getLogger(__name__).warning(
                "Skipping SVD: n_components=%d exceeds n_features=%d; retaining unreduced features",
                kwargs["n_components"], X_emb.shape[1])
            model = None
        else:
            model.fit(X_emb)
        return cls({"type": "sklearntruncatedsvd", "kwargs": kwargs}, model)

    def transform(self, corpus):
        """Reduce corpus (n x d features) to (n x n_components), or return it unchanged without a model.

        Returns:
            numpy.ndarray: Reduced features.
        """
        
        if self.model is None:
            return corpus
        if not issparse(corpus):
            return self.model.transform(corpus)
        # = model.transform (bit-identical) on the used columns only: csr @ components_.T copies the whole
        # n_components x d matrix per call (~0.4-1.5 s for one query at 150k+ char n-grams)
        X = csr_matrix(corpus)
        if X.shape[1] != self.model.n_features_in_:  # model.transform's check, skipped by the shortcut
            raise ValueError(f"X has {X.shape[1]} features, but the SVD was fitted on {self.model.n_features_in_}")
        cols = np.unique(X.indices)
        return X[:, cols] @ self.model.components_[:, cols].T
