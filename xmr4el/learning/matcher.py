import os
import json
import pickle
import numpy as np
import scipy.sparse as sp
from typing import Any, Dict, List, Tuple, Optional
from xmr4el.learning.classifiers import ClassifierModel


class MatcherTrainer():
    
    # Y = Y_binazer
    @staticmethod
    def train(
        X: np.ndarray,
        Y: np.ndarray,
        local_to_global_idx: List[int],
        global_to_local_idx: Dict[int, int],
        C: np.ndarray,
        config: Dict[str, Any],
        dtype: Any = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ClassifierModel]:
        """Train the matcher classifier.

        Parameters
        ----------
        X:
            Feature matrix for documents.
        Y:
            Label matrix for documents.
        local_to_global_idx:
            Mapping of local label indices to global indices.
        global_to_local_idx:
            Mapping of global label indices to local indices.
        C:
            Matrix used to construct the matching graph.
        config:
            Configuration dictionary for the classifier.
        dtype:
            Data type for internal numpy arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, ClassifierModel]
            Filtered feature matrix, filtered label matrix, matching matrix and the trained classifier model.
        """
    
        label_indices_local = [global_to_local_idx[g] for g in local_to_global_idx]
        Y_sub = Y[:, label_indices_local]
        
        keep_mask = np.asarray(Y_sub.sum(axis=1)).flatten() > 0
        
        X_node = X[keep_mask]
        Y_node = Y_sub[keep_mask]
        
        M_raw = Y_node @ C
        M = (M_raw > 0).astype(int)
        
        model = ClassifierModel.train(X_node, M, config, dtype, onevsrest=True)
        
        return X_node, Y_node, M, model
    



class Matcher:
    """Matcher pipeline that encapsulates model training and inference."""

    def __init__(
        self,
    ) -> None:
        """Initialize the matcher pipeline.

        Parameters
        ----------
        matcher_config:
            Optional configuration dictionary for the matcher.
        dtype:
            Desired data type for internal numpy arrays.
        """

        self._model: Optional[ClassifierModel] = None
        
    @property
    def model(self) -> Optional[ClassifierModel]:
        """Return the trained classification model."""
        return self._model
    
    @model.setter
    def model(self, value: ClassifierModel) -> None:
        """Set the trained classification model."""
        self._model = value
        
    def save(self, save_dir: str) -> None:
        """Persist the matcher object to disk.

        Parameters
        ----------
        save_dir:
            Directory where the model and state will be saved.
        """

        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        model = self.model

        if model is not None:
            model_path = os.path.join(save_dir, "matcher")

            if hasattr(model, "save") and callable(model.save):
                model.save(model_path)
            else:
                joblib.dump(model, f"{model_path}.joblib")

            state.pop("_model", None)

        with open(os.path.join(save_dir, "matcher.pkl"), "wb") as fout:
            pickle.dump(state, fout)
            
    @classmethod
    def load(cls, load_dir: str) -> "Matcher":
        """Load a matcher object from disk.

        Parameters
        ----------
        load_dir:
            Directory containing the saved matcher state.

        Returns
        -------
        Matcher
            Reconstructed matcher instance.
        """
        
        matcher_path = os.path.join(load_dir, "matcher.pkl")
        assert os.path.exists(matcher_path), f"Matcher path {matcher_path} does not exist"

        with open(matcher_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "matcher")
        matcher = ClassifierModel.load(model_path)
        setattr(model, "_model", matcher)
        
        return model
    
    
    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        local_to_global_idx: List[int],
        global_to_local_idx: Dict[int, int],
        C: np.ndarray,
        matcher_config: Optional[Dict[str, Any]] = None,
        dtype: Any = np.float32,
    ) -> None:
        """Train the matcher model.

        Parameters
        ----------
        X:
            Feature matrix for documents.
        Y:
            Label matrix for documents.
        local_to_global_idx:
            Mapping of local label indices to global indices.
        global_to_local_idx:
            Mapping of global label indices to local indices.
        C:
            Matrix used to construct the matching graph.
        """

        _, _, _, model = MatcherTrainer.train(
            X=X,
            Y=Y,
            local_to_global_idx=local_to_global_idx,
            global_to_local_idx=global_to_local_idx,
            C=C,
            config=matcher_config,
            dtype=dtype,
        )
        
        self.model = model
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given feature matrix."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict label probabilities for the given feature matrix."""
        return self.model.predict_proba(X)
    
    def classes(self) -> np.ndarray:
        """Return the classes predicted by the classifier."""
        return self.model.classes()
