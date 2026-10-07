import numpy as np
from xmr4el.hierarchy.clusterers import ClusteringModel, ClusteringTrainer


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


def test_balanced_kmeans():
    """Balanced sizes on non-divisible n, blobs recovered, seeded, zero row kept."""
    rs = np.random.RandomState(0)
    for n, k in ((1003, 6), (11745, 16)):
        Z = rs.normal(size=(n, 8)).astype(np.float32)
        Z[5] = 0
        labels = ClusteringModel.train(Z, {"type": "balancedkmeans", "kwargs": {"n_clusters": k}}).labels()
        sizes = np.bincount(labels)
        assert len(labels) == n and len(sizes) == k and sizes.max() - sizes.min() <= 1, sizes
        again = ClusteringModel.train(Z, {"type": "balancedkmeans", "kwargs": {"n_clusters": k}}).labels()
        assert (labels == again).all()
    # 4 blobs in distinct directions (cosine), 25 each
    Z = np.vstack([np.eye(8)[i] + rs.normal(0, 0.01, size=(25, 8)) for i in range(4)]).astype(np.float32)
    labels = ClusteringModel.train(Z, {"type": "balancedkmeans", "kwargs": {"n_clusters": 4}}).labels()
    blocks = labels.reshape(4, 25)
    assert (blocks == blocks[:, :1]).all() and len(set(blocks[:, 0])) == 4, blocks
