import numpy as np
from typing import Dict, List, Sequence, Tuple
from scipy.sparse import csr_matrix
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
    def label_binarizer(labels: Sequence[Sequence[int]]) -> Tuple[csr_matrix, np.ndarray]:
        """Y (n x L, sparse) and the sorted classes (column j = classes[j]), via ``MultiLabelBinarizer``."""
        mlb = MultiLabelBinarizer(sparse_output=True)
        Y = mlb.fit_transform(labels)
        return Y, mlb.classes_
    
    @staticmethod
    def generate_PIFA(X, Y) -> np.ndarray:
        """PIFA: Z = row-normalised Y.T @ X (L x d, dense, X's dtype); X (n x d), Y (n x L, sparse 0/1)."""
        Z = Y.T @ X
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        return normalize(Z, norm="l2", axis=1).astype(X.dtype, copy=False)
