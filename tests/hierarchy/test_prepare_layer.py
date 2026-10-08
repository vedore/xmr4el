"""prepare_layer hands each child its labels' parent Z rows (normalized) and routing-augmented X."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from xmr4el.hierarchy.tree import HierarchicalMLModel


def test_prepare_layer():

    rs = np.random.RandomState(0)
    n, D, L, K = 3000, 20, 6, 2
    X = csr_matrix(normalize(rs.rand(n, D)))
    Y = csr_matrix((np.ones(n), (np.arange(n), rs.randint(0, L, n))), shape=(n, L))
    Z = normalize(rs.rand(L, D))
    C = csr_matrix((np.ones(L), (np.arange(L), np.arange(L) % K)), shape=(L, K))
    fused = rs.rand(n, K)
    children = HierarchicalMLModel.prepare_layer(None, X, Y, Z, C, fused, np.arange(L))
    for X_aug, _, Z_aug, l2g, _, _ in children:
        assert X_aug.shape[1] == D + 3
        assert np.allclose(Z_aug, Z[l2g]), "child Z must be the parent's rows of its labels"
