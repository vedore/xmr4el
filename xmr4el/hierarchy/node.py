import os
import gc
import pickle
import logging

import numpy as np

from os.path import dirname, isfile, join as pjoin, exists as pexists
from pickle import dump as pkl_dump
from joblib import dump as jdump
from scipy.sparse import csr_matrix, eye as sp_eye
from sklearn.preprocessing import normalize
from xmr4el.hierarchy.clusterers import Clustering
from xmr4el.learning.matcher import Matcher
from xmr4el.learning.ranker import Ranker
from xmr4el.learning.scoring import predict_labels



class MLModel():

    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 ranker_config=None,
                 cur_config=None,
                 min_leaf_size=20,
                 max_leaf_size=None,
                 ranker_every_layer=False,
                 is_last_layer=False,
                 layer=None,
                 n_workers=8,
                 train_rankers=True,
                 ):
        
        self.logger = logging.getLogger(__name__)
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.ranker_config = ranker_config
        self.cur_config = cur_config
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.ranker_every_layer = ranker_every_layer
        self.is_last_layer = is_last_layer
        self.layer = layer
        self.n_workers = n_workers
        self.train_rankers = train_rankers
        
        self._local_to_global_idx = None
        self._global_to_local_idx = None
        
        self._cluster_model = None
        self._matcher_model = None
        self._ranker_model = None
        self._fused_scores = None
        self._alpha = None
        self._label_embeddings = None
        self._ranker_score_fn_cache = None
    
    @property
    def local_to_global_idx(self):
        return self._local_to_global_idx
    
    @local_to_global_idx.setter
    def local_to_global_idx(self, arr: np.ndarray):
        """
        arr[i] should be the global KB label ID for local label index i.
        """
        self._local_to_global_idx = arr
        # build inverse map
        self._global_to_local_idx = {g: i for i, g in enumerate(arr)}
    
    @property
    def global_to_local_idx(self):
        return self._global_to_local_idx
    
    @global_to_local_idx.setter
    def global_to_local_idx(self, value):
        self._global_to_local_idx = value
    
    @property
    def cluster_model(self):
        return self._cluster_model
    
    @cluster_model.setter
    def cluster_model(self, value):
        self._cluster_model = value

    @property
    def matcher_model(self):
        return self._matcher_model
    
    @matcher_model.setter
    def matcher_model(self, value):
        self._matcher_model = value

    @property
    def ranker_model(self):
        return self._ranker_model
    
    @ranker_model.setter
    def ranker_model(self, value):
        self._ranker_model = value
        
    @property
    def fused_scores(self):
        return self._fused_scores
    
    @fused_scores.setter
    def fused_scores(self, value):
        self._fused_scores = value
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        
    @property
    def label_embeddings(self):
        return self._label_embeddings
    
    @label_embeddings.setter
    def label_embeddings(self, value):
        self._label_embeddings = value
    
    @property
    def is_empty(self):
        return True if self.cluster_model is None else False
    

    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        state = self.__dict__.copy()

        # Mapping attribute names to their internal keys
        model_attrs = {
            "cluster_model": "_cluster_model",
            "matcher_model": "_matcher_model",
            "ranker_model": "_ranker_model"
        }

        for model_name, attr_key in model_attrs.items():
            model = getattr(self, model_name)
            if model is not None:
                model_path = pjoin(save_dir, model_name)

                if hasattr(model, 'save') and callable(model.save):
                    model.save(model_path)
                else:
                    jdump(model, f"{model_path}.joblib")

                # Remove model from state before pickling
                state.pop(attr_key, None)

        # Save fused scores separately
        fused_scores = self.fused_scores
        if fused_scores is None and not self.is_last_layer:
            raise ValueError("fused_scores is None. Cannot save.")
        if fused_scores is not None:
            np.save(pjoin(save_dir, "fused_scores.npy"), fused_scores)
        state.pop("_fused_scores", None)
        state.pop("_ranker_score_fn_cache", None)

        # Save label embeddings separately
        label_embeddings = self.label_embeddings
        if label_embeddings is not None:
            np.save(pjoin(save_dir, "label_embeddings.npy"), label_embeddings)
            state.pop("_label_embeddings", None)

        # Save remaining state
        with open(pjoin(save_dir, "mlmodel.pkl"), "wb") as fout:
            pkl_dump(state, fout)
    
    @classmethod
    def load(cls, load_path):
        # Accept either a directory OR a direct file to mlmodel.pkl
        base_dir = load_path
        # If they passed a file (e.g., .../mlmodel.pkl), go up one level
        if isfile(base_dir):
            base_dir = dirname(base_dir)

        model_state_path = pjoin(base_dir, "mlmodel.pkl")
        assert pexists(model_state_path), f"MLModel path {model_state_path} does not exist"

        with open(model_state_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)

        # Load sub-models saved in subfolders: <base_dir>/cluster_model, matcher_model, ranker_model
        model_dirs = {
            "cluster_model": Clustering if hasattr(Clustering, "load") else None,
            "matcher_model": Matcher if hasattr(Matcher, "load") else None,
            "ranker_model": Ranker if hasattr(Ranker, "load") else None,
        }

        for name, cls_ in model_dirs.items():
            subdir = pjoin(base_dir, name)
            if pexists(subdir) and cls_ is not None:
                setattr(model, name, cls_.load(subdir))
            else:
                print(f"Model {name} is not being loaded")

        # Load fused scores / label embeddings
        emb_path = pjoin(base_dir, "fused_scores.npy")
        assert pexists(emb_path) or model.is_last_layer, f"Expecting fused_scores at {emb_path}"
        model.fused_scores = np.load(emb_path, allow_pickle=True) if pexists(emb_path) else None

        label_emb_path = pjoin(base_dir, "label_embeddings.npy")
        model.label_embeddings = np.load(label_emb_path, allow_pickle=True) if pexists(label_emb_path) else None

        return model
        
    def __str__(self):
        return (
            f"Cluster Model: {self.cluster_model or 'None'}\n"
            f"Matcher Model: {self.matcher_model or 'None'}\n"
            f"Ranker Model: {self.ranker_model or 'None'}\n"
        )
    
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
            X_train: X_processed
            Y_train, Y_binazier
            Z, Pifa embeddings
        """
        
        # --- Ensure Z is in fused space ---
        Z_train = normalize(Z_train, norm="l2", axis=1) 
        self.label_embeddings = Z_train
        del Z_train
        
        self.global_to_local_idx = global_to_local
        self.local_to_global_idx = np.array(local_to_global, dtype=int)
        
        del global_to_local
        
        self.logger.info("Training ML: Clustering Phase")
        
        cluster_model = Clustering()
        if not self.is_last_layer:  # a leaf uses identity C below, so it is never clustered
            cluster_model.train(Z=self.label_embeddings,
                                local_to_global_idx=self.local_to_global_idx,
                                min_leaf_size=self.min_leaf_size,
                                max_leaf_size=self.max_leaf_size,
                                clustering_config=self.clustering_config,
                                dtype=np.float32
                                )
            if cluster_model.is_empty:
                return
        
        self.cluster_model = cluster_model
        del cluster_model
        gc.collect()
        
        # Retrieve C
        C = self.cluster_model.c_node

        # Defect #6: a cluster-level matcher makes every label in a cluster tie at predict time
        # (`m = cluster_scores[q_idx, c]` below), so ordering inside a leaf is decided by CSR index
        # order. Identity C at the leaf makes the matcher one-vs-rest over labels -- XR-Linear leaf
        # semantics -- so `M = Y_node @ I = Y_node` and `label_cluster` becomes arange(L), which
        # makes `m` per-label with no change at the scoring site.
        if self.is_last_layer:
            C = sp_eye(self.label_embeddings.shape[0], format="csr", dtype=np.float32)
            self.cluster_model.c_node = C   # predict() re-reads this and asserts K == C.shape[1]

        cluster_labels = np.asarray(C.argmax(axis=1)).flatten()
    
        self.logger.info("Training ML: Matcher Phase")

        # With identity C above, the leaf matcher is one-vs-rest over labels and a leaf label can
        # have a single positive instance. SGDClassifier's `early_stopping` splits off a validation
        # set *stratified* on y, which needs >= 2 members per class and raises otherwise. Disabled
        # for the leaf only, so the non-leaf layers stay byte-identical and the #6 row stays
        # attributable to #6.
        matcher_config = self.matcher_config
        if self.is_last_layer and matcher_config.get("type") == "sklearnsgdclassifier":
            matcher_config = {
                **matcher_config,
                "kwargs": {**matcher_config.get("kwargs", {}), "early_stopping": False},
            }

        # Make the Matcher
        matcher_model = Matcher()  
        matcher_model.train(X_train, 
                            Y_train, 
                            local_to_global_idx=self.local_to_global_idx, 
                            global_to_local_idx=self.global_to_local_idx, 
                            C=C,
                            matcher_config=matcher_config,
                            dtype=np.float32
                            )     
         
        self.matcher_model = matcher_model 
        del matcher_model
        gc.collect()
        
        # Rankers are only used for prediction with -scorer ranker; internal training scores
        # are matcher-only. train_rankers=False skips them; predict then uses cosine.
        train_ranker = self.train_rankers and (self.ranker_every_layer or self.is_last_layer)
        
        def _topb_sparse(P: np.ndarray, b: int) -> csr_matrix:
            # P: (n x K_or_L) dense proba; returns (n x K_or_L) CSR 0/1 mask of top-b per row
            n, K = P.shape
            b = max(1, min(b, K))
            idx_part = np.argpartition(P, K - b, axis=1)[:, -b:]
            rows = np.repeat(np.arange(n, dtype=np.int32), b)
            cols = idx_part.ravel()
            data = np.ones(n * b, dtype=np.int8)
            return csr_matrix((data, (rows, cols)), shape=(n, K))
        
        if train_ranker:
        
            M_TFN = self.matcher_model.m_node
            M_MAN = None
        
            if self.is_last_layer:
                P = self.matcher_model.predict_proba(X_train)
                # With identity C above, M_TFN is Y_node, so this top-b mask is the ranker's whole
                # negative pool (ranker.py) instead of a cluster's worth of instances.
                # b=5 leaves too few negatives for neg_mult * n_pos; 20 keeps the pool fed.
                M_MAN = _topb_sparse(P, b=20)
            
            self.logger.info("Training ML: Ranker Phase")
            
            # print("Ranker")
            ranker_model = Ranker()
            ranker_model.train(X_train, 
                                Y_train, 
                                self.label_embeddings, 
                                M_TFN, 
                                M_MAN, 
                                cluster_labels,
                                local_to_global_idx=self.local_to_global_idx,
                                layer=self.layer,
                                n_label_workers=self.n_workers,
                                ranker_config=self.ranker_config,
                                cur_config=self.cur_config
                                )
            
            self.ranker_model = ranker_model
            del ranker_model
        else:
            self.ranker_model = None
            
        gc.collect()
        
        self.fused_scores = None
        if not self.is_last_layer:
            print("Fusing Scores")
            cluster_scores = self.matcher_model.predict_proba(X_train)
            self.fused_scores = csr_matrix(np.maximum(cluster_scores, 0.0))
        
    def predict(self, X_query, beam_size=5, topk=None, return_matrix=False,
                fusion="lp_fusion", eps=1e-6, alpha=0.5, p=3, scorer=None):
        return predict_labels(self, X_query, beam_size, topk, return_matrix, fusion, eps, alpha, p, scorer)
