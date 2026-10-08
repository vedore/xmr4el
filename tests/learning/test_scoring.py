import numpy as np
from scipy.sparse import csr_matrix

from xmr4el.learning.scoring import label_max_cos


def test_label_max_cos():
    rng = np.random.default_rng(0)
    Q, B = rng.normal(size=(5, 4)), rng.normal(size=(7, 4))
    b_labels = np.array([2, 0, 2, 0, 3, 2, 0])  # label 1 has no rows
    cos = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ (B / np.linalg.norm(B, axis=1, keepdims=True)).T
    want = np.full((5, 4), -1.0)
    for j in (0, 2, 3):
        want[:, j] = cos[:, b_labels == j].max(axis=1)
    assert np.allclose(label_max_cos(Q, B, b_labels, 4, chunk=2), want, atol=1e-6)
    assert np.allclose(label_max_cos(csr_matrix(Q), csr_matrix(B), b_labels, 4), want, atol=1e-6)
