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
        
        self._local_to_global_idx = None  # see the setter
        self.global_to_local_idx = None
        
        self.cluster_model = None  # Clustering; a leaf holds identity C
        self.matcher_model = None
        self.fused_scores = None  # internal node: matcher cluster scores on its train rows (n x K), consumed by prepare_layer; not saved
        self.label_embeddings = None  # internal node: normalised Z (L x d); not saved
    
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
        self.global_to_local_idx = {g: i for i, g in enumerate(arr)}
    
    @property
    def is_empty(self):
        return self.cluster_model is None
    

    
    def save(self, save_dir):
        model_attrs = ("cluster_model", "matcher_model")
        for model_name in model_attrs:
            model = getattr(self, model_name)
            if model is not None and not callable(getattr(model, "save", None)):
                raise TypeError(f"{model_name} ({type(model).__name__}) has no save()")
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        state = self.__dict__.copy()
        for model_name in model_attrs:
            model = state.pop(model_name)  # saved in its own dir, not pickled
            if model is not None:
                model.save(pjoin(save_dir, model_name))

        # training-only arrays (prepare_layer / clustering consume them before the save); prediction needs neither
        state.pop("fused_scores", None)
        state.pop("label_embeddings", None)

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

        return model
        
    def __str__(self):
        return (
            f"Cluster Model: {self.cluster_model or 'None'}\n"
            f"Matcher Model: {self.matcher_model or 'None'}\n"
        )
    
    def train(self, X_train, Y_train, Z_train, local_to_global, global_to_local):
        """Train one node: X_train (n x d), Y_train (n x L, the node's labels), Z_train (L x d, PIFA rows of
        those labels); local_to_global[j] = global label index of local label j. An internal node clusters Z
        into C (L x K) and trains its matcher on Y @ C; a leaf uses identity C (L x L)."""
        
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
