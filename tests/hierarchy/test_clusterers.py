import numpy as np
from xmr4el.hierarchy.clusterers import ClusteringTrainer


def test_small_cluster_reassignment():
    """No label may be dropped: every C_node row must have exactly one cluster."""
    rs = np.random.RandomState(0)
    # 3 tight blobs of 20 + a stray pair far away: unbalanced kmeans gives the pair its own
    # cluster, which is below min_leaf_size and would otherwise be dropped.
    Z = np.vstack([rs.normal(c, 0.01, size=(20, 4)) for c in (0.0, 5.0, 10.0)]
                  + [rs.normal(40.0, 0.01, size=(2, 4))]).astype(np.float32)
    cfg = {"type": "sklearnkmeans", "kwargs": {"n_clusters": 4, "random_state": 0}}
    C, _ = ClusteringTrainer.train(Z, cfg, min_leaf_size=5)
    assert C is not None, "clustering returned nothing"
    assert C.shape == (Z.shape[0], 3), f"expected 3 valid clusters, got {C.shape}"
    per_row = np.asarray(C.sum(axis=1)).ravel()
    assert (per_row == 1).all(), f"{int((per_row == 0).sum())} labels dropped from C_node"
    # the strays must land in the blob they are actually nearest to
    assert C[60].indices[0] == C[61].indices[0] == C[40].indices[0]
    print("clustering selfcheck ok")
