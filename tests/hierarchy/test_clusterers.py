import numpy as np
from xmr4el.hierarchy.clusterers import ClusteringModel, ClusteringTrainer, _balanced_assign


def test_small_cluster_reassignment():
    """No label may be dropped: labels of undersized clusters move to the nearest valid centroid."""
    rs = np.random.RandomState(0)
    e = np.eye(8)
    # Balanced 4-way split of 62 labels gives clusters of 16/15/16/15 = these blobs. With min_leaf_size 16 the
    # 15-blobs are reassigned; each leans towards one 16-blob.
    dirs = [e[0], e[1] + 0.6 * e[0], e[2], e[3] + 0.6 * e[2]]
    Z = np.vstack([d + rs.normal(0, 0.01, size=(m, 8)) for d, m in zip(dirs, (16, 15, 16, 15))]).astype(np.float32)
    cfg = {"type": "balancedkmeans", "kwargs": {"n_clusters": 4}}
    C, _ = ClusteringTrainer.train(Z, cfg, min_leaf_size=16)
    assert C is not None and C.shape == (62, 2), C
    assert (np.asarray(C.sum(axis=1)).ravel() == 1).all(), "labels dropped from C_node"
    col = C.indices
    assert (col[:31] == col[0]).all() and (col[31:] == col[31]).all() and col[0] != col[31], col


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


def test_balanced_assign_matches_greedy():
    """The vectorized fill equals the greedy scan over all entries by S - p descending (steps=0: p = 0)."""
    rs = np.random.RandomState(0)
    for n, k in ((500, 7), (2000, 64)):
        S = rs.normal(size=(n, k)).astype(np.float32)
        S[:, 0] += 1  # contested column
        cap = np.full(k, n // k)
        cap[: n - cap.sum()] += 1
        want, left = np.full(n, -1), cap.copy()
        for f in np.argsort(-S, axis=None, kind="stable"):
            i, j = divmod(int(f), k)
            if want[i] < 0 and left[j]:
                want[i], left[j] = j, left[j] - 1
        assert (_balanced_assign(S, cap, steps=0) == want).all()
