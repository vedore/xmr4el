import numpy as np
from typing import Dict, List, Sequence, Tuple
from numpy import zeros, asarray, sum as npsum
from numpy.linalg import norm
from sklearn.preprocessing import normalize, MultiLabelBinarizer


class LabelEmbeddingFactory():
    """Utility factory to build label embeddings."""
        
    @staticmethod
    def generate_label_matrix(label_to_indices: Dict[int, List[int]]) -> List[List[int]]:
        """Expand a mapping from labels to corpus indices into a label matrix (row i = corpus row i)."""
        label_to_matrix: List[List[int]] = [None] * sum(len(ids) for ids in label_to_indices.values())
        for key, ids in label_to_indices.items():
            for i in ids:
                label_to_matrix[i] = [key]
        return label_to_matrix
        
    @staticmethod
    def label_binarizer(labels: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Binarize labels using ``MultiLabelBinarizer``."""
        mlb = MultiLabelBinarizer(sparse_output=True)
        Y = mlb.fit_transform(labels)
        return Y, mlb.classes_
    
    @staticmethod
    def generate_PIFA(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate PIFA label embeddings from features and label matrix."""
        Z: List[np.ndarray] = []

        D = X.shape[1]

        for label_idx in range(Y.shape[1]):
            row_indices = Y[:, label_idx].nonzero()[0]

            if len(row_indices) == 0:
                Z.append(zeros(D))
            else:
                positive_x = X[row_indices]
                v_ell = npsum(positive_x, axis=0)
                v_ell = asarray(v_ell).ravel()
                z_ell = v_ell / (norm(v_ell) + 1e-10)
                Z.append(z_ell)

        Z = np.vstack(Z)
        Z = normalize(Z, norm="l2", axis=1)
        return Z
    
