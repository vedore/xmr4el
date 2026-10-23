"""SVD projection on the used columns equals sklearn's TruncatedSVD.transform, for one row and a batch."""
import numpy as np
import scipy.sparse as sp

from xmr4el.features.reduction import DimensionModel


def test_svd_transform_matches_sklearn():
    X = sp.random(300, 2000, density=0.01, format="csr", random_state=0)
    dim = DimensionModel.fit(X, {"type": "sklearntruncatedsvd", "kwargs": {"n_components": 20}})
    for Q in (X[:1], X[5:6], X):
        assert np.array_equal(dim.transform(Q), dim.model.transform(Q))
