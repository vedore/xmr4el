"""TF-IDF features with explicit configuration and persistence."""
import json
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    """The pipeline's TF-IDF adapter; config uses type/kwargs like the other stages."""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    @staticmethod
    def check_config(config):
        if config.get("type") != "tfidf":
            raise ValueError(f"invalid vectorizer type {config.get('type')!r}; expected 'tfidf'")

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vec_config.json"), "w", encoding="utf-8") as fout:
            json.dump(self.config, fout)
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, "vec_config.json"), encoding="utf-8") as fin:
            config = json.load(fin)
        cls.check_config(config)
        with open(os.path.join(load_dir, "vectorizer.pkl"), "rb") as fin:
            return cls(config, pickle.load(fin))

    @classmethod
    def fit(cls, trn_corpus, config=None, dtype=np.float32):
        config = config if config is not None else {"type": "tfidf", "kwargs": {}}
        cls.check_config(config)
        defaults = {
            "ngram_range": (1, 2),  # n-grams from 1 to 2
            "max_features": None,  # No max feature limit
            "min_df": 0.0,  # Minimum document frequency ratio
            "max_df": 0.98,  # Maximum document frequency ratio
            "binary": False,  # Term frequency is not binary
            "use_idf": True,  # Use inverse document frequency
            "smooth_idf": True,  # Apply smoothing to idf
            "sublinear_tf": False,  # Use raw term frequency
            "norm": "l2",  # Apply L2 normalization
            "analyzer": "word",  # Tokenizes by word
            "stop_words": None,  # No stop words used
        }

        kwargs = {**defaults, **config.get("kwargs", {})}
        kwargs["ngram_range"] = tuple(kwargs["ngram_range"])  # JSON gives a list
        model = TfidfVectorizer(**kwargs, dtype=dtype).fit(trn_corpus)
        return cls({"type": "tfidf", "kwargs": kwargs}, model)

    def transform(self, corpus):
        return self.model.transform(corpus)
