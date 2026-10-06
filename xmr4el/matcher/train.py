import numpy as np

from typing import Any, Dict, List, Tuple
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


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
    

def _selfcheck():
    """Identity C at the leaf (defect #6) turns the matcher into one-vs-rest over labels, where a
    label can have a single positive. SGD's `early_stopping` stratifies its validation split and
    raises on that; MLModel.train disables it for the last layer only. Asserts both halves."""
    import json
    import scipy.sparse as sp

    cfg = json.load(open(".models/xmr4el_flag6_sapbert_config.json"))["matcher_config"]  # SGD matcher
    rng = np.random.default_rng(0)
    n, d, L = 40, 12, 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    Y = sp.lil_matrix((n, L), dtype=np.float32)
    for i in range(n - 1):
        Y[i, i % (L - 1)] = 1.0
    Y[n - 1, L - 1] = 1.0          # label L-1 gets exactly one positive
    Y = Y.tocsr()
    assert np.asarray(Y.sum(axis=0)).ravel()[L - 1] == 1

    kw = {"local_to_global_idx": list(range(L)),
          "global_to_local_idx": {g: g for g in range(L)},
          "C": sp.eye(L, format="csr", dtype=np.float32)}

    def _run(early_stopping):
        c = {**cfg, "kwargs": {**cfg["kwargs"], "early_stopping": early_stopping}}
        return MatcherTrainer.train(X=X, Y=Y, config=c, **kw)

    try:
        _run(True)
        raise AssertionError("expected a stratify ValueError with early_stopping=True")
    except ValueError as exc:
        assert "least populated" in str(exc), exc

    _, _, M, model = _run(False)
    assert M.shape == (n, L), M.shape
    assert model.predict_proba(X).shape == (n, L)
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
