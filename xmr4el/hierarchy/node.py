import os
import pickle
import logging

import numpy as np

from os.path import dirname, isfile, join as pjoin, exists as pexists
from pickle import dump as pkl_dump
from scipy.sparse import csr_matrix, eye as sp_eye
from sklearn.preprocessing import normalize
from xmr4el.hierarchy.clusterers import Clustering
from xmr4el.learning.matcher import Matcher



def _load_npy(path):
    """np.load; a sparse matrix saved by np.save comes back as a 0-d object array: unwrap it."""
    a = np.load(path, allow_pickle=True)
    return a.item() if a.dtype == object and a.shape == () else a


class MLModel():

    def __init__(self, 
                 clustering_config=None, 
                 matcher_config=None, 
                 min_leaf_size=20,
                 is_last_layer=False,
                 layer=None,
                 ):
        
        self.logger = logging.getLogger(__name__)
        
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.min_leaf_size = min_leaf_size
        self.is_last_layer = is_last_layer
        self.layer = layer
        
        self._local_to_global_idx = None
        self._global_to_local_idx = None
        
        self._cluster_model = None
        self._matcher_model = None
        self._fused_scores = None
        self._label_embeddings = None
    
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
    def fused_scores(self):
        return self._fused_scores
    
    @fused_scores.setter
    def fused_scores(self, value):
        self._fused_scores = value
        
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
        # Mapping attribute names to their internal keys
        model_attrs = {
            "cluster_model": "_cluster_model",
            "matcher_model": "_matcher_model",
        }
        for model_name in model_attrs:
            model = getattr(self, model_name)
            if model is not None and not callable(getattr(model, "save", None)):
                raise TypeError(f"{model_name} ({type(model).__name__}) has no save()")
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        state = self.__dict__.copy()
        for model_name, attr_key in model_attrs.items():
            model = getattr(self, model_name)
            if model is not None:
                model.save(pjoin(save_dir, model_name))
                state.pop(attr_key, None)  # Remove model from state before pickling

        # Save fused scores separately
        fused_scores = self.fused_scores
        if fused_scores is None and not self.is_last_layer:
            raise ValueError("fused_scores is None. Cannot save.")
        if fused_scores is not None:
            np.save(pjoin(save_dir, "fused_scores.npy"), fused_scores)
        state.pop("_fused_scores", None)

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
        if not pexists(model_state_path):
            raise FileNotFoundError(f"MLModel path {model_state_path} does not exist")

        with open(model_state_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)

        # Load sub-models saved in subfolders: <base_dir>/cluster_model, matcher_model
        for name, cls_ in (("cluster_model", Clustering), ("matcher_model", Matcher)):
            subdir = pjoin(base_dir, name)
            if pexists(subdir):
                setattr(model, name, cls_.load(subdir))

        # Load fused scores / label embeddings
        emb_path = pjoin(base_dir, "fused_scores.npy")
        if not (pexists(emb_path) or model.is_last_layer):
            raise ValueError(f"Expecting fused_scores at {emb_path}")
        model.fused_scores = _load_npy(emb_path) if pexists(emb_path) else None

        label_emb_path = pjoin(base_dir, "label_embeddings.npy")
        model.label_embeddings = _load_npy(label_emb_path) if pexists(label_emb_path) else None

        return model
        
    def __str__(self):
        return (
            f"Cluster Model: {self.cluster_model or 'None'}\n"
            f"Matcher Model: {self.matcher_model or 'None'}\n"
        )
    
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """
            X_train: X_processed
            Y_train, Y_binary
            Z, Pifa embeddings
        """
        
        # Clustering input; a leaf only needs its label count (identity C)
        Z_train = normalize(Z_train, norm="l2", axis=1)
        n_labels = Z_train.shape[0]
        if not self.is_last_layer:
            self.label_embeddings = Z_train
        del Z_train
        
        self.global_to_local_idx = global_to_local
        self.local_to_global_idx = np.array(local_to_global, dtype=int)
        
        del global_to_local
        
        cluster_model = Clustering()
        if not self.is_last_layer:  # a leaf uses identity C below, so it is never clustered
            self.logger.debug("Clustering started: layer_index=%s labels=%d", self.layer, self.label_embeddings.shape[0])
            cluster_model.train(Z=self.label_embeddings,
                                min_leaf_size=self.min_leaf_size,
                                clustering_config=self.clustering_config,
                                dtype=np.float32
                                )
            if cluster_model.is_empty:
                return
        
        self.cluster_model = cluster_model
        del cluster_model
        
        # Retrieve C
        C = self.cluster_model.c_node

        # Identity C at the leaf makes the matcher one-vs-rest over labels (XR-Linear leaf, M = Y_node), so
        # column j of the leaf's predict_proba scores local label j (HierarchicalMLModel.predict). A
        # cluster-level leaf matcher tied every label of a cluster (defect #6).
        if self.is_last_layer:
            C = sp_eye(n_labels, format="csr", dtype=np.float32)
            self.cluster_model.c_node = C

        self.logger.debug("Matcher started: layer_index=%s rows=%d targets=%d", self.layer, X_train.shape[0], C.shape[1])

        # Make the Matcher
        matcher_model = Matcher()  
        matcher_model.train(X_train, 
                            Y_train, 
                            local_to_global_idx=self.local_to_global_idx, 
                            global_to_local_idx=self.global_to_local_idx, 
                            C=C,
                            matcher_config=self.matcher_config,
                            dtype=np.float32
                            )     
         
        self.matcher_model = matcher_model 
        del matcher_model
        
        self.fused_scores = None
        if not self.is_last_layer:
            self.logger.debug("Preparing matcher routing scores: layer_index=%s", self.layer)
            cluster_scores = self.matcher_model.predict_proba(X_train)
            self.fused_scores = csr_matrix(np.maximum(cluster_scores, 0.0))
