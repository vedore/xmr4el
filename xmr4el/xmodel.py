import json
import os
import joblib
import pickle
import time 
import logging
from copy import deepcopy

import numpy as np

from xmr4el import get_logger, set_verbosity
from pathlib import Path
from datetime import datetime
from typing import Optional
from xmr4el.features.label_embeddings import LabelEmbeddingFactory
from xmr4el.data.readers import Preprocessor
from xmr4el.features.encoder import TextEncoder
from xmr4el.hierarchy.tree import HierarchicalMLModel
from xmr4el.learning.scoring import label_max_cos




class XModel:
    
    def __init__(self, 
                 vectorizer_config: dict = None,
                 transformer_config: dict = None,
                 dimension_config: dict = None,
                 context_vectorizer_config: dict = None,
                 context_dimension_config: dict = None,
                 context_window: Optional[int] = None,
                 abbrev_expansion: Optional[str] = None,
                 clustering_config: dict = None,
                 matcher_config: dict = None,
                 min_leaf_size: int = 20,
                 max_leaf_size: int = None,
                 cut_half_cluster: bool = False,
                 depth: int = 1,
                 features: str = "sapbert_char_context",
                 predict_config: Optional[dict] = None,
                 verbose: Optional[int] = None,
                 logger: Optional[logging.Logger] = None
                 ):
        
        # 0 = WARNING, 1 = INFO, 2 = DEBUG
        if logger is not None:
            self.logger = logger
        else:
            if verbose is not None:
                set_verbosity(verbose)
            self.logger = get_logger("xmodel")
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        # features "sapbert_char_context" ("tfidf" = TF-IDF only): context block configs; PubTator loaders build the context from
        # `context_window` words around each mention (None = whole document)
        self.context_vectorizer_config = context_vectorizer_config
        self.context_dimension_config = context_dimension_config
        self.context_window = context_window
        # PubTator loaders expand in-document abbreviations: "append" | "replace" | None (off)
        self.abbrev_expansion = abbrev_expansion
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster = cut_half_cluster
        self.depth = depth
        self.features = features
        # Defaults of `predict`, stored with the tree so a bare evaluate.py reproduces the reported numbers
        self.predict_config = {"beam_size": 10, "topk": 0, "knn_beta": 0.0, **(predict_config or {})}
        
        self._text_encoder = None
        self._hml = None
        self._training_texts = None
        self._original_labels = None
        self._X = None
        self._Y = None
        self._Z = None
        
    
    def __str__(self) -> str:
        """Human-friendly multi-line summary of the model and its config/state."""
        def _short(obj, max_len=100):
            """Return a short representation for objects (dicts/lists/others)."""
            try:
                if obj is None:
                    return "None"
                
                if isinstance(obj, dict):
                    # show keys and count
                    keys = list(obj.keys())
                    k_display = ", ".join(map(str, keys[:6]))
                    more = f", ... (+{len(keys)-6})" if len(keys) > 6 else ""
                    return f"dict(keys=[{k_display}{more}])"
                    
                if isinstance(obj, (list, tuple, set)):
                    n = len(obj)
                    # preview = ", ".join(repr(x) for x in list(obj)[:6])
                    # more = f", ... (+{n-6})" if n > 6 else ""
                    return f"{type(obj).__name__}(len={n})"
                    
                # for numpy arrays / pandas objects show shape/len if possible
                if hasattr(obj, "shape"):
                    return f"{type(obj).__name__}(shape={getattr(obj, 'shape')})"
                
                if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
                    return f"{type(obj).__name__}(len={len(obj)})"
                
                r = repr(obj)
                return r if len(r) <= max_len else r[:max_len] + "..."
            
            except Exception:
                return f"<unrepr {type(obj).__name__}>"

        logger_name = getattr(self, "logger", None)
        if logger_name is not None:
            try:
                logger_info = f"{self.logger.name} (level={self.logger.level})"
            except Exception:
                logger_info = repr(self.logger)
        else:
            logger_info = "None"

        parts = [
            f"XModel summary:",
            f"  logger: {logger_info}",
            f"  depth={self.depth}, features={self.features}, predict={self.predict_config}",
            f"  cluster: min_leaf_size={self.min_leaf_size}, max_leaf_size={self.max_leaf_size}, cut_half_cluster={self.cut_half_cluster}",
            f"  configs:",
            f"    vectorizer: {_short(self.vectorizer_config)}",
            f"    transformer: {_short(self.transformer_config)}",
            f"    dimension: {_short(self.dimension_config)}",
            f"    clustering: {_short(self.clustering_config)}",
            f"    matcher: {_short(self.matcher_config)}",
            f"  internal state:",
            f"    text_encoder: {_short(self._text_encoder)}",
            f"    hml: {_short(self._hml)}",
            f"    training_texts: {_short(self._training_texts)}",
            f"    original_labels: {_short(self._original_labels)}",
            f"    X/Y/Z: {_short(self._X)}, {_short(self._Y)}, {_short(self._Z)}",
        ]
        return "\n".join(parts)

    def __repr__(self) -> str:
        # concise repr that can be used in containers / REPL
        try:
            return f"XModel(depth={self.depth}, features={self.features})"
        except Exception:
            return "<XModel (repr error)>"

    
    @property
    def text_encoder(self):
        return self._text_encoder
    
    @text_encoder.setter
    def text_encoder(self, value):
        self._text_encoder = value

    @property
    def model(self):
        return self._hml
    
    @model.setter
    def model(self, value):
        self._hml = value
        
    @property
    def training_set(self):
        return self._training_texts
    
    @training_set.setter
    def training_set(self, value):
        self._training_texts = value
        
    @property
    def initial_labels(self):
        return self._original_labels
    
    @initial_labels.setter
    def initial_labels(self, value):
        self._original_labels = value
        
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, value):
        self._X = value
        
    @property
    def Y(self):
        return self._Y
    
    @Y.setter
    def Y(self, value):
        self._Y = value
        
    @property 
    def Z(self):
        return self._Z
    
    @Z.setter
    def Z(self, value):
        self._Z = value        
        
    def save(self, save_dir):
        start = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_{timestamp}")
        os.makedirs(save_dir, exist_ok=False)
    
        state = self.__dict__.copy()
        
        model = self.model
        model_path = os.path.join(save_dir, "hml")
        
        if model is not None:
            if hasattr(model, "save"):
                model.save(model_path)
            else:
                try:
                    joblib.dump(model, f"{model_path}.joblib")
                except ImportError:
                    with open(f"{model_path}.pkl", "wb") as f:
                        pickle.dump(model, f)  
            
            state.pop("_hml", None) # Popped _hml from class
        
        text_encoder = self.text_encoder
        text_encoder_path = os.path.join(save_dir, "text_encoder")
        state.pop("_text_encoder", None)
        
        if text_encoder is not None:
            if hasattr(text_encoder, "save"):
                text_encoder.save(text_encoder_path)
            else:
                try:
                    joblib.dump(text_encoder, f"{text_encoder_path}.joblib")
                except ImportError:
                    with open(f"{text_encoder_path}.pkl", "wb") as f:
                        pickle.dump(text_encoder, f)  
                          
        with open(os.path.join(save_dir, "xmodel.pkl"), "wb") as fout:
            pickle.dump(state, fout)
        self.logger.info("Model saved: path=%s elapsed=%.1fs", save_dir, time.perf_counter() - start)
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = os.path.join(load_dir, "xmodel.pkl")
        assert os.path.exists(xmodel_path), f"XModel path {xmodel_path} does not exist"
        
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
        # Label index j is row j of Z: refuse a label list that does not match it
        if model.Z is not None:
            assert len(model.initial_labels) == model.Z.shape[0], (
                f"{len(model.initial_labels)} labels vs {model.Z.shape[0]} Z rows"
            )
        
        model_path = os.path.join(load_dir, "hml")
        hml = HierarchicalMLModel.load(model_path)
        setattr(model, "_hml", hml)
        
        text_encoder_path = os.path.join(load_dir, "text_encoder")
        text_encoder = TextEncoder.load(text_encoder_path)
        setattr(model, "_text_encoder", text_encoder)
        
        return model
    
    @classmethod
    def load_config(cls, path: str | Path) -> object:
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        return XModel(**data)
        
    
    def _fit(self, X_text, Y_text):
        """Returns embeddings: ndarray"""
        
        self.training_set = deepcopy(X_text)
        
        self.logger.info("Preparing data: groups=%d", len(Y_text))
        
        X_processed, Y_label_to_indices = Preprocessor.prepare_data(X_text, Y_text)
        
        start = time.perf_counter()
        self.logger.info("Encoding started: rows=%d features=%s", len(X_processed), self.features)
        
        # Encode X_processed
        text_encoder = TextEncoder(
            vectorizer_config=self.vectorizer_config,
            transformer_config=self.transformer_config,
            dimension_config=self.dimension_config, 
            features=self.features,
            context_vectorizer_config=self.context_vectorizer_config,
            context_dimension_config=self.context_dimension_config,
            )
        
        self.text_encoder = text_encoder
        
        X_emb = text_encoder.encode(X_processed)
        self.logger.info("Encoding completed: shape=%s elapsed=%.1fs", X_emb.shape, time.perf_counter() - start)

        start = time.perf_counter()
        self.logger.info("Label embeddings started: method=PIFA")
        
        Y_label_matrix = LabelEmbeddingFactory.generate_label_matrix(Y_label_to_indices)
        
        # Process Labels
        Y_binazer, classes = LabelEmbeddingFactory.label_binarizer(Y_label_matrix)
        # Label index j is column j of Y; empty label groups have no column
        self.initial_labels = classes.tolist()
        Z = LabelEmbeddingFactory.generate_PIFA(X_emb, Y_binazer)
        self.logger.info("Label embeddings completed: labels=%d shape=%s elapsed=%.1fs",
                         len(classes), Z.shape, time.perf_counter() - start)
        
        return X_emb, Y_binazer, Z 
    
    def train(self, X_text, Y_text):
        
        start = time.perf_counter()
        
        self.X, self.Y, self.Z = self._fit(X_text=X_text, Y_text=Y_text)

        n_labels = self.Z.shape[0]
        local_to_global = np.arange(n_labels, dtype=int)
        global_to_local = {g: i for i, g in enumerate(local_to_global)}

        hierarchy_start = time.perf_counter()
        self.logger.info("Hierarchy started: depth=%d labels=%d", self.depth, n_labels)

        hml = HierarchicalMLModel(
            clustering_config=self.clustering_config,
            matcher_config=self.matcher_config,
            min_leaf_size=self.min_leaf_size,
            max_leaf_size=self.max_leaf_size,
            cut_half_cluster=self.cut_half_cluster,
            layer=self.depth,
        )

        hml.train(
            X_train=self.X,
            Y_train=self.Y,
            Z_train=self.Z,
            local_to_global=local_to_global,
            global_to_local=global_to_local
        )

        self.model = hml
        self.logger.info("Hierarchy completed: elapsed=%.1fs", time.perf_counter() - hierarchy_start)
        self.logger.info("Training completed: elapsed=%.1fs", time.perf_counter() - start)
        
        
    def predict(self, X_text, beam_size: Optional[int] = None, topk: Optional[int] = None,
                knn_beta: Optional[float] = None):
        """Score CSR (n_queries x n_labels) for raw text queries (`HierarchicalMLModel.predict`).
        Arguments left None take the tree's `predict_config`.

        knn_beta > 0 multiplies each candidate's score by exp(knn_beta * knn), knn = max cosine of the query's
        mention block to that label's training rows (`mention_block`, `self.X`, `self.Y`). Only scores change:
        the candidates stay the tree's."""
        cfg = self.resolve_predict_config(beam_size=beam_size, topk=topk, knn_beta=knn_beta)
        knn_beta = cfg["knn_beta"]
        time_start_encoding = time.perf_counter()
        X_query = self.text_encoder.predict(X_text)
        self.logger.info("Prediction encoding completed: rows=%d shape=%s elapsed=%.1fs",
                         X_query.shape[0], X_query.shape, time.perf_counter() - time_start_encoding)

        scores = self.model.predict(X_query, beam_size=cfg["beam_size"], topk=cfg["topk"])
        if knn_beta:
            time_start_knn = time.perf_counter()
            Y = self.Y.tocsr()
            assert (Y.getnnz(axis=1) == 1).all(), "knn needs exactly one label per training row"
            block = self.mention_block()
            knn = label_max_cos(X_query[:, block], self.X[:, block], Y.indices, scores.shape[1])
            rows = np.repeat(np.arange(scores.shape[0]), np.diff(scores.indptr))
            scores.data *= np.exp(knn_beta * knn[rows, scores.indices].astype(np.float64))
            self.logger.info("Knn re-scoring completed: beta=%s elapsed=%.1fs", knn_beta,
                             time.perf_counter() - time_start_knn)
        return scores

    def resolve_predict_config(self, **overrides):
        """`predict_config` with the non-None overrides applied."""
        return {**self.predict_config, **{k: v for k, v in overrides.items() if v is not None}}

    def mention_block(self):
        """Columns of the mention encoder in X: "sapbert_char_context" is [transformer | char SVD | context SVD];
        else all."""
        if self.features != "sapbert_char_context":
            return slice(None)
        d = self.X.shape[1] - self.dimension_config["kwargs"]["n_components"] \
            - self.context_dimension_config["kwargs"]["n_components"]
        return slice(0, d)
