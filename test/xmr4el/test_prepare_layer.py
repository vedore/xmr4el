"""prepare_layer must keep the parent's label embedding in the child Z (no node-size-scaled pad)."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from xmr4el.xmr.base import HierarchicaMLModel

rs = np.random.RandomState(0)
n, D, L, K = 3000, 20, 6, 2
X = csr_matrix(normalize(rs.rand(n, D)))
Y = csr_matrix((np.ones(n), (np.arange(n), rs.randint(0, L, n))), shape=(n, L))
Z = normalize(rs.rand(L, D))
C = csr_matrix((np.ones(L), (np.arange(L), np.arange(L) % K)), shape=(L, K))
fused = rs.rand(n, K)
children = HierarchicaMLModel.prepare_layer(None, X, Y, Z, C, fused, np.arange(L))
for X_aug, _, Z_aug, l2g, _, _ in children:
    assert X_aug.shape[1] == Z_aug.shape[1] == D + 3
    cos = (Z_aug[:, :D] * Z[l2g]).sum(1) / np.linalg.norm(Z_aug, axis=1)
    assert np.allclose(cos, 1.0), f"child Z lost the label embedding: cos {cos}"
print("ok")
