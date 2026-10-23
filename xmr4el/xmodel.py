import json
import numbers
import os
import pickle
import time 
import logging
from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix
from uuid import uuid4

from xmr4el import get_logger, set_verbosity
from pathlib import Path
from datetime import datetime
from typing import Optional
from xmr4el.features.label_embeddings import LabelEmbeddingFactory
from xmr4el.data.readers import Preprocessor
from xmr4el.features.encoder import FEATURES, TextEncoder
from xmr4el.hierarchy.clusterers import ClusteringModel
from xmr4el.hierarchy.tree import HierarchicalMLModel, rank_rows
from xmr4el.learning.classifiers import ClassifierModel
from xmr4el.learning.scoring import knn_index, label_max_cos




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


# Training-only state, saved to research.pkl and loaded only by `XModel.load(..., research=True)`
RESEARCH_STATE = ("training_set", "X", "Y", "Z")
PREDICT_DEFAULTS = {"beam_size": 10, "topk": 0, "knn_beta": 0.0, "leaf_candidates": 100}
KNN_BETA_MAX = 100  # fusion multiplies by exp(beta * cos): exp(100) keeps every score finite (overflow near 709)


def check_search(cfg):
    """Raise ValueError unless beam_size and leaf_candidates are ints >= 1, topk an int >= 0 and knn_beta a number in
    [0, KNN_BETA_MAX] (keys left out or None are not checked)."""
    is_int = lambda v: isinstance(v, numbers.Integral) and not isinstance(v, bool)
    bad = []
    if cfg.get("beam_size") is not None and not (is_int(cfg["beam_size"]) and cfg["beam_size"] >= 1):
        bad.append(f"beam_size {cfg['beam_size']!r} (int >= 1)")
    if cfg.get("leaf_candidates") is not None and not (is_int(cfg["leaf_candidates"]) and cfg["leaf_candidates"] >= 1):
        bad.append(f"leaf_candidates {cfg['leaf_candidates']!r} (int >= 1)")
    if cfg.get("topk") is not None and not (is_int(cfg["topk"]) and cfg["topk"] >= 0):
        bad.append(f"topk {cfg['topk']!r} (int >= 0)")
    beta = cfg.get("knn_beta")
    if beta is not None and not (isinstance(beta, numbers.Real) and not isinstance(beta, bool)
                                 and 0 <= beta <= KNN_BETA_MAX):  # NaN fails too
        bad.append(f"knn_beta {beta!r} (number in [0, {KNN_BETA_MAX}])")
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
                 routing_features: bool = True,
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
        # learner settings fail here, not after encoding
        if matcher_config:
            ClassifierModel.init_model(matcher_config)
        if clustering_config:
            ClusteringModel.check_config(clustering_config)
        
        self.min_leaf_size = min_leaf_size
        self.cut_half_cluster = cut_half_cluster
        self.routing_features = routing_features  # HierarchicalMLModel.routing_features
        if not (isinstance(depth, numbers.Integral) and not isinstance(depth, bool) and depth >= 1):
            raise ValueError(f"depth must be an int >= 1, got {depth!r}")
        self.depth = depth
        if features not in FEATURES:
            raise ValueError(f"features must be one of {FEATURES}, got {features!r}")
        self.features = features
        if not (isinstance(min_leaf_size, numbers.Integral) and not isinstance(min_leaf_size, bool) and min_leaf_size >= 0):
            raise ValueError(f"min_leaf_size must be an int >= 0, got {min_leaf_size!r}")
        batch = (transformer_config or {}).get("kwargs", {}).get("batch_size", 0)
        if not (isinstance(batch, numbers.Integral) and not isinstance(batch, bool) and batch >= 0):
            raise ValueError(f"transformer batch_size must be an int >= 0 (0 = default), got {batch!r}")
        # Defaults of `predict`, stored with the tree so a bare evaluate.py reproduces the reported numbers
        unknown = sorted(set(predict_config or {}) - set(PREDICT_DEFAULTS))
        if unknown:
            raise ValueError(f"predict_config: unknown keys {unknown} (known: {sorted(PREDICT_DEFAULTS)})")
        if None in (predict_config or {}).values():
            raise ValueError(f"predict_config {predict_config}: leave a key out instead of null")
        self.predict_config = {**PREDICT_DEFAULTS, **(predict_config or {})}
        check_search(self.predict_config)
        
        self.text_encoder = None
        self.model = None  # HierarchicalMLModel
        self.training_set = None  # grouped training texts as given to train
        # Label index j = column j of Y = row j of Z = initial_labels[j] (binarizer classes_, sorted)
        self.initial_labels = None
        self.X = None  # train features, n x d (row i = i-th text after Preprocessor.prepare_data)
        self.Y = None  # train labels, n x L (sparse, one label per row)
        self.Z = None  # PIFA label embeddings, L x d
        # knn_index of X's mention block: built on the first knn predict, or at save when predict_config knn_beta > 0
        # (saved then, so knn needs no X / Y at load)
        self.knn_index = None
        self.tree_id = None  # set by save
        
    
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
                    
                # Arrays show shape without printing their contents.
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
            "XModel summary:",
            f"  logger: {logger_info}",
            f"  depth={self.depth}, features={self.features}, predict={self.predict_config}",
            f"  cluster: min_leaf_size={self.min_leaf_size}, cut_half_cluster={self.cut_half_cluster}, "
            f"routing_features={self.routing_features}",
            "  configs:",
            f"    vectorizer: {_short(self.vectorizer_config)}",
            f"    transformer: {_short(self.transformer_config)}",
            f"    dimension: {_short(self.dimension_config)}",
            f"    clustering: {_short(self.clustering_config)}",
            f"    matcher: {_short(self.matcher_config)}",
            "  internal state:",
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
        for name, part in parts.items():  # before any file: an unfitted model writes nothing
            if not callable(getattr(part, "save", None)):
                raise TypeError(f"{name} ({type(part).__name__}) has no save(): train or load the model first")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_{timestamp}")
        save_dir, n = base, 1
        while True:  # two saves in one second: suffix _1, _2, ...
            try:
                os.makedirs(save_dir, exist_ok=False)
                break
            except FileExistsError:
                save_dir, n = f"{base}_{n}", n + 1
    
        if self.X is None and getattr(self, "research_path", None):  # loaded without it: carry it over
            self.load_research()
        if self.knn_index is None and self.predict_config["knn_beta"] and self.X is not None:
            self.knn_index = self._knn_index()
        state = self.__dict__.copy()
        state.pop("model"), state.pop("text_encoder"), state.pop("research_path", None)
        state["tree_id"] = uuid4().hex  # this saved artifact (run_experiment records it); self gets it once published
        if not self.predict_config["knn_beta"]:
            state["knn_index"] = None  # saved only for trees whose default search uses knn
        research = {key: state.pop(key) for key in RESEARCH_STATE}
        for name, part in parts.items():
            part.save(os.path.join(save_dir, name))

        with open(os.path.join(save_dir, "xmodel.json"), "w") as fout:  # readable view, never loaded
            json.dump(_json_state({**state, **research}), fout, indent=2, default=repr)
        with open(os.path.join(save_dir, "research.pkl"), "wb") as fout:
            pickle.dump(research, fout)
        # written last, then renamed: xmodel.pkl exists only in a completely saved tree (load requires it)
        with open(os.path.join(save_dir, "xmodel.pkl.tmp"), "wb") as fout:
            pickle.dump(state, fout)
        os.replace(os.path.join(save_dir, "xmodel.pkl.tmp"), os.path.join(save_dir, "xmodel.pkl"))
        self.tree_id = state["tree_id"]
        self.logger.info("Model saved: path=%s elapsed=%.1fs", save_dir, time.perf_counter() - start)
        return save_dir
    
    @classmethod
    def load(cls, load_dir, research=False):
        """Inference state (tree, encoder, labels, search defaults, saved knn index). research=True also loads
        RESEARCH_STATE (training texts, X, Y, Z: diagnostics, baselines, rerankers); otherwise those stay None
        until a knn predict on a tree saved without its knn index loads them."""
        xmodel_path = os.path.join(load_dir, "xmodel.pkl")
        if not os.path.exists(xmodel_path):
            raise FileNotFoundError(f"XModel path {xmodel_path} does not exist")
        
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        model.research_path = os.path.join(load_dir, "research.pkl")
        model.model = HierarchicalMLModel.load(os.path.join(load_dir, "hml"))
        # Label index j is root label j and row j of Z: refuse a label list that does not match them
        n_root = len(model.model.hmodel[0][0].local_to_global_idx)
        if len(model.initial_labels) != n_root:
            raise ValueError(f"{len(model.initial_labels)} labels vs {n_root} root labels")
        if research:
            model.load_research()
        model.text_encoder = TextEncoder.load(os.path.join(load_dir, "text_encoder"))
        
        return model
    
    def load_research(self):
        """RESEARCH_STATE from the loaded tree's research.pkl."""
        with open(self.research_path, "rb") as fin:
            self.__dict__.update(pickle.load(fin))
        if len(self.initial_labels) != self.Z.shape[0]:
            raise ValueError(f"{len(self.initial_labels)} labels vs {self.Z.shape[0]} Z rows")

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
        self.logger.info("Encoding started: rows=%d features=%s", len(X_processed), self.features,
                         extra={"event": {"stage": "encode", "phase": "start", "rows": len(X_processed)}})
        
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
        self.transformer_config = text_encoder.transformer_config  # revision resolved to the commit
        self.logger.info("Encoding completed: shape=%s elapsed=%.1fs", X_emb.shape, time.perf_counter() - start,
                         extra={"event": {"stage": "encode", "phase": "complete"}})

        start = time.perf_counter()
        self.logger.info("Label embeddings started: method=PIFA", extra={"event": {"stage": "labels", "phase": "start"}})
        
        Y_label_matrix = LabelEmbeddingFactory.generate_label_matrix(Y_label_to_indices)
        
        # Process Labels
        Y_binary, classes = LabelEmbeddingFactory.label_binarizer(Y_label_matrix)
        # Label index j is column j of Y; empty label groups have no column
        self.initial_labels = classes.tolist()
        Z = LabelEmbeddingFactory.generate_PIFA(X_emb, Y_binary)
        self.logger.info("Label embeddings completed: labels=%d shape=%s elapsed=%.1fs",
                         len(classes), Z.shape, time.perf_counter() - start,
                         extra={"event": {"stage": "labels", "phase": "complete", "labels": len(classes)}})
        
        return X_emb, Y_binary, Z
    
    def train(self, X_text, Y_text):
        """X_text: one list of training texts per label group, Y_text: the group labels. Sets X (n x d),
        Y (n x L), Z (L x d), initial_labels and the tree."""
        start = time.perf_counter()
        missing = [name for name in ("clustering_config", "matcher_config") if not getattr(self, name)]
        if missing:  # before encoding
            raise ValueError(f"train needs {missing}")
        self.model = None  # a failed (re)train leaves no tree, never an old tree with the new encoder/labels
        self.tree_id = None  # a retrained model is no saved artifact (rerankers check this id)

        self.X, self.Y, self.Z = self._fit(X_text=X_text, Y_text=Y_text)
        self.knn_index = None

        n_labels = self.Z.shape[0]
        local_to_global = np.arange(n_labels, dtype=int)

        hierarchy_start = time.perf_counter()
        self.logger.info("Hierarchy started: depth=%d labels=%d", self.depth, n_labels,
                         extra={"event": {"stage": "hierarchy", "phase": "start", "labels": n_labels}})

        hml = HierarchicalMLModel(
            clustering_config=self.clustering_config,
            matcher_config=self.matcher_config,
            min_leaf_size=self.min_leaf_size,
            cut_half_cluster=self.cut_half_cluster,
            layer=self.depth,
            routing_features=self.routing_features,
        )

        hml.train(
            X_train=self.X,
            Y_train=self.Y,
            Z_train=self.Z,
            local_to_global=local_to_global
        )

        self.model = hml
        self.logger.info("Hierarchy completed: elapsed=%.1fs", time.perf_counter() - hierarchy_start,
                         extra={"event": {"stage": "hierarchy", "phase": "complete"}})
        self.logger.info("Training completed: elapsed=%.1fs", time.perf_counter() - start)
        
        
    def predict(self, X_text, beam_size: Optional[int] = None, topk: Optional[int] = None,
                knn_beta: Optional[float] = None, leaf_candidates: Optional[int] = None):
        """Score CSR (n_queries x n_labels) for raw text queries (`HierarchicalMLModel.predict`).
        Arguments left None take the tree's `predict_config`.

        knn_beta > 0 multiplies each candidate's score by exp(knn_beta * knn), knn = max cosine of the query's
        mention block to that label's training rows (`mention_block`, `self.X`, `self.Y`). The candidates
        stay the tree's; rows are re-sorted by the fused score before topk."""
        search = dict(beam_size=beam_size, topk=topk, knn_beta=knn_beta, leaf_candidates=leaf_candidates)
        self.resolve_predict_config(**search)  # bad settings fail before encoding
        if len(X_text) == 0:
            return csr_matrix((0, len(self.initial_labels)), dtype=np.float64)
        time_start_encoding = time.perf_counter()
        X_query = self.text_encoder.predict(X_text)
        self.logger.info("Prediction encoding completed: rows=%d shape=%s elapsed=%.1fs",
                         X_query.shape[0], X_query.shape, time.perf_counter() - time_start_encoding)
        return self.predict_encoded(X_query, **search)

    def predict_encoded(self, X_query, beam_size: Optional[int] = None, topk: Optional[int] = None,
                        knn_beta: Optional[float] = None, leaf_candidates: Optional[int] = None):
        """`predict` on already encoded queries (`text_encoder.predict` rows): search + knn only."""
        cfg = self.resolve_predict_config(beam_size=beam_size, topk=topk, knn_beta=knn_beta,
                                          leaf_candidates=leaf_candidates)
        knn_beta = cfg["knn_beta"]
        # knn changes the order: rank all of the tree's candidates, then cut to topk
        scores = self.model.predict(X_query, beam_size=cfg["beam_size"], topk=0 if knn_beta else cfg["topk"],
                                    leaf_candidates=cfg["leaf_candidates"])
        if knn_beta:
            time_start_knn = time.perf_counter()
            block = self.mention_block()
            if self.knn_index is None:  # sorting + normalising X's rows costs O(train rows) per call otherwise
                if self.X is None:  # loaded tree saved without its knn index (predict_config knn_beta 0)
                    self.load_research()
                self.knn_index = self._knn_index()
            rows = np.repeat(np.arange(scores.shape[0]), np.diff(scores.indptr))
            knn = label_max_cos(X_query[:, block], None, None, scores.shape[1], rows=rows, cols=scores.indices,
                                index=self.knn_index)
            vals = scores.data * np.exp(knn_beta * knn.astype(np.float64))
            scores = rank_rows(rows, scores.indices, vals, scores.shape, cfg["topk"])
            self.logger.info("Knn re-scoring completed: beta=%s elapsed=%.1fs", knn_beta,
                             time.perf_counter() - time_start_knn)
        return scores

    def _knn_index(self):
        Y = self.Y.tocsr()
        assert (Y.getnnz(axis=1) == 1).all(), "knn needs exactly one label per training row"
        return knn_index(self.X[:, self.mention_block()], Y.indices)

    def resolve_predict_config(self, **overrides):
        """`predict_config` with the non-None overrides applied (keys it lacks take PREDICT_DEFAULTS)."""
        cfg = {**PREDICT_DEFAULTS, **self.predict_config, **{k: v for k, v in overrides.items() if v is not None}}
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
