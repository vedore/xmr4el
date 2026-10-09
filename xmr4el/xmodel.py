import json
import math
import numbers
import os
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
from xmr4el.hierarchy.tree import HierarchicalMLModel, rank_rows
from xmr4el.learning.scoring import label_max_cos




def _json_state(state):
    """xmodel.json view of the pickled state: JSON values as is, arrays/sparse matrices as type/shape/dtype (+nnz),
    lists (labels, training texts) as type/len/first; json.dump's default=repr covers the rest."""
    out = {}
    for key, v in state.items():
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            out[key] = {"type": type(v).__name__, "shape": list(v.shape), "dtype": str(v.dtype)}
            if hasattr(v, "nnz"):
                out[key]["nnz"] = int(v.nnz)
        elif isinstance(v, (list, tuple)):
            first = v[0] if len(v) else None
            out[key] = {"type": type(v).__name__, "len": len(v),
                        "first": first if isinstance(first, (str, int, float, type(None))) else repr(first)[:200]}
        else:
            out[key] = v
    return out


def check_search(cfg):
    """Raise ValueError unless beam_size is an int >= 1, topk an int >= 0 and knn_beta a finite number
    (keys left out or None are not checked)."""
    is_int = lambda v: isinstance(v, numbers.Integral) and not isinstance(v, bool)
    bad = []
    if cfg.get("beam_size") is not None and not (is_int(cfg["beam_size"]) and cfg["beam_size"] >= 1):
        bad.append(f"beam_size {cfg['beam_size']!r} (int >= 1)")
    if cfg.get("topk") is not None and not (is_int(cfg["topk"]) and cfg["topk"] >= 0):
        bad.append(f"topk {cfg['topk']!r} (int >= 0)")
    beta = cfg.get("knn_beta")
    if beta is not None and not (isinstance(beta, numbers.Real) and not isinstance(beta, bool) and math.isfinite(beta)):
        bad.append(f"knn_beta {beta!r} (finite number)")
    if bad:
        raise ValueError("search settings: " + ", ".join(bad))


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
        self.cut_half_cluster = cut_half_cluster
        if not (isinstance(depth, numbers.Integral) and not isinstance(depth, bool) and depth >= 1):
            raise ValueError(f"depth must be an int >= 1, got {depth!r}")
        self.depth = depth
        self.features = features
        # Defaults of `predict`, stored with the tree so a bare evaluate.py reproduces the reported numbers
        self.predict_config = {"beam_size": 10, "topk": 0, "knn_beta": 0.0, **(predict_config or {})}
        
        self.text_encoder = None
        self.model = None  # HierarchicalMLModel
        self.training_set = None  # grouped training texts as given to train
        # Label index j = column j of Y = row j of Z = initial_labels[j] (binarizer classes_, sorted)
        self.initial_labels = None
        self.X = None  # train features, n x d (row i = i-th text after Preprocessor.prepare_data)
        self.Y = None  # train labels, n x L (sparse, one label per row)
        self.Z = None  # PIFA label embeddings, L x d
        
    
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
            f"  cluster: min_leaf_size={self.min_leaf_size}, cut_half_cluster={self.cut_half_cluster}",
            f"  configs:",
            f"    vectorizer: {_short(self.vectorizer_config)}",
            f"    transformer: {_short(self.transformer_config)}",
            f"    dimension: {_short(self.dimension_config)}",
            f"    clustering: {_short(self.clustering_config)}",
            f"    matcher: {_short(self.matcher_config)}",
            f"  internal state:",
            f"    text_encoder: {_short(self.text_encoder)}",
            f"    hml: {_short(self.model)}",
            f"    training_texts: {_short(self.training_set)}",
            f"    original_labels: {_short(self.initial_labels)}",
            f"    X/Y/Z: {_short(self.X)}, {_short(self.Y)}, {_short(self.Z)}",
        ]
        return "\n".join(parts)

    def __repr__(self) -> str:
        # concise repr that can be used in containers / REPL
        try:
            return f"XModel(depth={self.depth}, features={self.features})"
        except Exception:
            return "<XModel (repr error)>"

    
    def save(self, save_dir):
        start = time.perf_counter()
        parts = {"hml": self.model, "text_encoder": self.text_encoder}
        for name, part in parts.items():
            if part is not None and not callable(getattr(part, "save", None)):
                raise TypeError(f"{name} ({type(part).__name__}) has no save()")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_{timestamp}")
        save_dir, n = base, 1
        while True:  # two saves in one second: suffix _1, _2, ...
            try:
                os.makedirs(save_dir, exist_ok=False)
                break
            except FileExistsError:
                save_dir, n = f"{base}_{n}", n + 1
    
        state = self.__dict__.copy()
        state.pop("model"), state.pop("text_encoder")
        for name, part in parts.items():
            if part is not None:
                part.save(os.path.join(save_dir, name))

        with open(os.path.join(save_dir, "xmodel.json"), "w") as fout:  # readable view, never loaded
            json.dump(_json_state(state), fout, indent=2, default=repr)
        # written last, then renamed: xmodel.pkl exists only in a completely saved tree (load requires it)
        with open(os.path.join(save_dir, "xmodel.pkl.tmp"), "wb") as fout:
            pickle.dump(state, fout)
        os.replace(os.path.join(save_dir, "xmodel.pkl.tmp"), os.path.join(save_dir, "xmodel.pkl"))
        self.logger.info("Model saved: path=%s elapsed=%.1fs", save_dir, time.perf_counter() - start)
        return save_dir
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = os.path.join(load_dir, "xmodel.pkl")
        if not os.path.exists(xmodel_path):
            raise FileNotFoundError(f"XModel path {xmodel_path} does not exist")
        
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
        # Label index j is row j of Z: refuse a label list that does not match it
        if model.Z is not None and len(model.initial_labels) != model.Z.shape[0]:
            raise ValueError(f"{len(model.initial_labels)} labels vs {model.Z.shape[0]} Z rows")
        
        model.model = HierarchicalMLModel.load(os.path.join(load_dir, "hml"))
        model.text_encoder = TextEncoder.load(os.path.join(load_dir, "text_encoder"))
        
        return model
    
    @classmethod
    def load_config(cls, path: str | Path) -> object:
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        return XModel(**data)
        
    
    def _fit(self, X_text, Y_text):
        """Encode the grouped texts: returns X (n x d), Y (n x L, sparse), Z = PIFA (L x d)."""
        
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
        Y_binary, classes = LabelEmbeddingFactory.label_binarizer(Y_label_matrix)
        # Label index j is column j of Y; empty label groups have no column
        self.initial_labels = classes.tolist()
        Z = LabelEmbeddingFactory.generate_PIFA(X_emb, Y_binary)
        self.logger.info("Label embeddings completed: labels=%d shape=%s elapsed=%.1fs",
                         len(classes), Z.shape, time.perf_counter() - start)
        
        return X_emb, Y_binary, Z
    
    def train(self, X_text, Y_text):
        """X_text: one list of training texts per label group, Y_text: the group labels. Sets X (n x d),
        Y (n x L), Z (L x d), initial_labels and the tree."""
        start = time.perf_counter()
        self.model = None  # a failed (re)train leaves no tree, never an old tree with the new encoder/labels

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
        mention block to that label's training rows (`mention_block`, `self.X`, `self.Y`). The candidates
        stay the tree's; rows are re-sorted by the fused score before topk."""
        cfg = self.resolve_predict_config(beam_size=beam_size, topk=topk, knn_beta=knn_beta)
        knn_beta = cfg["knn_beta"]
        time_start_encoding = time.perf_counter()
        X_query = self.text_encoder.predict(X_text)
        self.logger.info("Prediction encoding completed: rows=%d shape=%s elapsed=%.1fs",
                         X_query.shape[0], X_query.shape, time.perf_counter() - time_start_encoding)

        # knn changes the order: rank all of the tree's candidates, then cut to topk
        scores = self.model.predict(X_query, beam_size=cfg["beam_size"], topk=0 if knn_beta else cfg["topk"])
        if knn_beta:
            time_start_knn = time.perf_counter()
            Y = self.Y.tocsr()
            assert (Y.getnnz(axis=1) == 1).all(), "knn needs exactly one label per training row"
            block = self.mention_block()
            rows = np.repeat(np.arange(scores.shape[0]), np.diff(scores.indptr))
            knn = label_max_cos(X_query[:, block], self.X[:, block], Y.indices, scores.shape[1],
                                rows=rows, cols=scores.indices)
            vals = scores.data * np.exp(knn_beta * knn.astype(np.float64))
            scores = rank_rows(rows, scores.indices, vals, scores.shape, cfg["topk"])
            self.logger.info("Knn re-scoring completed: beta=%s elapsed=%.1fs", knn_beta,
                             time.perf_counter() - time_start_knn)
        return scores

    def resolve_predict_config(self, **overrides):
        """`predict_config` with the non-None overrides applied."""
        cfg = {**self.predict_config, **{k: v for k, v in overrides.items() if v is not None}}
        check_search(cfg)
        return cfg

    def mention_block(self):
        """Columns of the mention encoder in X (`feature_blocks`); all for "tfidf"."""
        return self.feature_blocks().get("mention", slice(None))

    def feature_blocks(self):
        """{"mention", "char", "context"} column slices of X for "sapbert_char_context"
        ([transformer | char SVD | context SVD], fitted widths); {} for "tfidf"."""
        if self.features != "sapbert_char_context":
            return {}
        widths = self.text_encoder.block_widths
        assert widths is not None, "tree saved before TextEncoder.block_widths: retrain it"
        ends = np.cumsum(widths)
        return {name: slice(e - w, e) for name, w, e in zip(("mention", "char", "context"), widths, ends)}
