import json
import os
import joblib
import pickle
import time 
import warnings
import logging

import numpy as np

from xmr4el import get_logger, set_verbosity
from pathlib import Path
from datetime import datetime
from typing import Optional
from scipy.sparse import csr_matrix
from xmr4el.featurization.label_embedding_factory import LabelEmbeddingFactory
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.featurization.text_encoder import TextEncoder
from xmr4el.xmr.base import HierarchicaMLModel
from xmr4el.utils.temp_store import TempVarStore


os.makedirs("/tmp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


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
                 ranker_config: dict = None,
                 cur_config: dict = None,
                 min_leaf_size: int = 20,
                 max_leaf_size: int = None,
                 cut_half_cluster: bool = False,
                 ranker_every_layer: bool = True,
                 train_rankers: bool = True,
                 n_workers: int = 8,
                 depth: int = 1,
                 emb_flag: int = 6,
                 verbose: Optional[int] = None,
                 logger: Optional[logging.Logger] = None
                 ):
        
        # 0 = WARNING, 1 = INFO, 2 = DEBUG
        if logger is not None:
            self.logger = logger
            self.logger.debug("Using user-supplied logger for XModel")
        else:
            if verbose is not None:
                set_verbosity(verbose)
            self.logger = get_logger("models.xmodel")

        # rest of initialization
        self.logger.info("Initializing XModel")
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        # emb_flag 6 (1 = TF-IDF only): context block configs; PubTator loaders build the context from
        # `context_window` words around each mention (None = whole document)
        self.context_vectorizer_config = context_vectorizer_config
        self.context_dimension_config = context_dimension_config
        self.context_window = context_window
        # PubTator loaders expand in-document abbreviations: "append" | "replace" | None (off)
        self.abbrev_expansion = abbrev_expansion
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.ranker_config = ranker_config
        self.cur_config = cur_config
        
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster = cut_half_cluster
        self.ranker_every_layer = ranker_every_layer
        self.train_rankers = train_rankers
        
        self.n_workers = n_workers
        
        self.depth = depth
        self.emb_flag =emb_flag
        
        self._text_encoder = None
        self._hml = None
        self._training_texts = None
        self._original_labels = None
        self._X = None
        self._Y = None
        self._Z = None
    
        self.EPS = 1e-12
        self.TOPK_DBG = 50
        
        self.temp_var = TempVarStore()
    
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
            f"  workers: n_workers={self.n_workers}",
            f"  depth={self.depth}, emb_flag={self.emb_flag}",
            f"  cluster: min_leaf_size={self.min_leaf_size}, max_leaf_size={self.max_leaf_size}, cut_half_cluster={self.cut_half_cluster}",
            f"  ranker_every_layer={self.ranker_every_layer}, train_rankers={getattr(self, 'train_rankers', True)}",
            f"  configs:",
            f"    vectorizer: {_short(self.vectorizer_config)}",
            f"    transformer: {_short(self.transformer_config)}",
            f"    dimension: {_short(self.dimension_config)}",
            f"    clustering: {_short(self.clustering_config)}",
            f"    matcher: {_short(self.matcher_config)}",
            f"    ranker: {_short(self.ranker_config)}",
            f"    cur: {_short(self.cur_config)}",
            f"  internal state:",
            f"    text_encoder: {_short(self._text_encoder)}",
            f"    hml: {_short(self._hml)}",
            f"    training_texts: {_short(self._training_texts)}",
            f"    original_labels: {_short(self._original_labels)}",
            f"    X/Y/Z: {_short(self._X)}, {_short(self._Y)}, {_short(self._Z)}",
            f"  temp_var: {_short(getattr(self, 'temp_var', None))}",
        ]
        return "\n".join(parts)

    def __repr__(self) -> str:
        # concise repr that can be used in containers / REPL
        try:
            return f"XModel(depth={self.depth}, n_workers={self.n_workers}, emb_flag={self.emb_flag})"
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
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = os.path.join(load_dir, "xmodel.pkl")
        assert os.path.exists(xmodel_path), f"XModel path {xmodel_path} does not exist"
        
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
        # Trees saved before the label-order fix stored input order; sorted() is idempotent
        model.initial_labels = sorted(set(model.initial_labels))
        # A legacy tree trained with empty label groups cannot be repaired by sorting
        if model.Z is not None:
            assert len(model.initial_labels) == model.Z.shape[0], (
                f"{len(model.initial_labels)} labels vs {model.Z.shape[0]} Z rows"
            )
        
        model_path = os.path.join(load_dir, "hml")
        hml = HierarchicaMLModel.load(model_path)
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
        
        self.training_set = self.temp_var.save_model_temp(X_text)
        
        self.logger.info("Preparing Data")
        
        X_processed, Y_label_to_indices = Preprocessor.prepare_data_older(X_text, Y_text)
        
        self.logger.info("Started Encoding")
        
        # Encode X_processed
        text_encoder = TextEncoder(
            vectorizer_config=self.vectorizer_config,
            transformer_config=self.transformer_config,
            dimension_config=self.dimension_config, 
            flag=self.emb_flag, # Needs to be a variable, could have a stop to check
            context_vectorizer_config=getattr(self, "context_vectorizer_config", None),
            context_dimension_config=getattr(self, "context_dimension_config", None),
            )
        
        self.text_encoder = text_encoder
        
        X_emb = text_encoder.encode(X_processed)
        
        Y_label_matrix = LabelEmbeddingFactory.generate_label_matrix(Y_label_to_indices)
        
        # Process Labels
        Y_binazer, classes = LabelEmbeddingFactory.label_binarizer(Y_label_matrix)
        # Label index j is column j of Y; empty label groups have no column
        self.initial_labels = classes.tolist()
        Z = LabelEmbeddingFactory.generate_PIFA(X_emb, Y_binazer)
        
        return X_emb, Y_binazer, Z 
    
    def train(self, X_text, Y_text):
        
        self.logger.info("Started Training")
        
        self.X, self.Y, self.Z = self._fit(X_text=X_text, Y_text=Y_text)

        n_labels = self.Z.shape[0]
        local_to_global = np.arange(n_labels, dtype=int)
        global_to_local = {g: i for i, g in enumerate(local_to_global)}

        self.logger.info("Hierarchical Model Pipeline")

        hml = HierarchicaMLModel(
            clustering_config=self.clustering_config,
            matcher_config=self.matcher_config,
            ranker_config=self.ranker_config,
            cur_config=self.cur_config,
            min_leaf_size=self.min_leaf_size,
            max_leaf_size=self.max_leaf_size,
            n_workers=self.n_workers,
            cut_half_cluster=self.cut_half_cluster,
            ranker_every_layer=self.ranker_every_layer,
            layer=self.depth,
            train_rankers=getattr(self, "train_rankers", True),
        )

        hml.train(
            X_train=self.X,
            Y_train=self.Y,
            Z_train=self.Z,
            local_to_global=local_to_global,
            global_to_local=global_to_local
        )

        self.model = hml
        
        self.training_set = self.temp_var.load_model_temp(self.training_set)
        
        self.temp_var.delete_model_temp()
        
    def predict(self, X_text, 
                topk: int = 5, 
                beam_size: int = 5, 
                fusion: str = "geometric", 
                alpha: float = 0.5,
                topk_mode: str = "per_leaf", 
                n_jobs: int =-1,
                path_score: bool = False):
            """Predict label scores for given text inputs.

            Parameters
            ----------
            X_text : list-like
                Raw text queries.
            topk : int, optional
                Number of labels to consider when computing hit counts.
            beam_size : int, optional
                Beam width for hierarchical traversal.
            golden_labels : Sequence[Sequence[str]], optional
                Gold standard label IDs per query (strings).
            return_hits : bool, optional
                If ``True`` and ``golden_labels`` provided, returns hit counts.

            Returns
            -------
            csr_matrix
                Sparse score matrix for all queries.
            list, optional
                Hit counts per query when ``return_hits`` is ``True``.
            """
            
            time_start_encoding = time.time()

            X_query = self.text_encoder.predict(X_text)
            
            time_end_encoding = time.time()
            
            print("Encoding: ", time_end_encoding - time_start_encoding)
            
            # topk_mode "global" is handled by the hierarchy too; its final_path follows its scores
            return self.model.predict(X_query,
                                      topk=topk,
                                      beam_size=beam_size,
                                      fusion=fusion,
                                      alpha=alpha,
                                      n_jobs=n_jobs,
                                      topk_mode=topk_mode,
                                      path_score=path_score)
