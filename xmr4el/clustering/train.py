import logging

import numpy as np

from typing import Any, Dict, Optional, Tuple, Counter
from numpy import ones
from scipy.sparse import csr_matrix
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


logger = logging.getLogger(__name__)


class ClusteringTrainer:
    """Utility class providing the clustering training routine."""
    
    @staticmethod
    def train(
        Z: np.ndarray,
        config: Dict[str, Any],
        min_leaf_size: int = 20,
        max_leaf_size: Optional[int] = None,
        dtype: Any = np.float32,
    ) -> Tuple[Optional[csr_matrix], Optional[ClusteringModel]]:
        """Train clustering without recursive partitioning."""
        
        n_points = Z.shape[0]
        n_clusters = config["kwargs"]["n_clusters"]

        if n_points <= min_leaf_size:
            # print(f"Too few points ({n_points}), stopping clustering.")
            return None, None

        # Train clustering model
        clustering_model = ClusteringModel.train(Z, config, dtype)
        cluster_labels = clustering_model.labels()
        cluster_counts = Counter(cluster_labels)
        # print(f"Cluster sizes: {cluster_counts}")

        # Filter out invalid clusters
        valid_clusters = [cid for cid, cnt in cluster_counts.items() if cnt >= min_leaf_size]
        if len(valid_clusters) <= 1:
            # print(f"Only {len(valid_clusters)} valid clusters after pruning, stopping clustering.")
            return None, None

        assert len(cluster_labels) == n_points, (
            f"clustering returned {len(cluster_labels)} labels for {n_points} points; "
            "C_node rows would be misaligned with Z"
        )

        # Points in undersized clusters are reassigned to the nearest valid centroid.
        # Dropping them would leave an all-zero C_node row, i.e. a label that can never be
        # proposed (xmr/base.py:474) and never reaches the next layer (xmr/base.py:817).
        cluster_to_col = {cid: i for i, cid in enumerate(valid_clusters)}
        cols = np.array([cluster_to_col.get(int(c), -1) for c in cluster_labels])

        orphans = cols < 0
        n_orphans = int(orphans.sum())
        if n_orphans:
            Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
            centroids = np.vstack([Zn[cluster_labels == cid].mean(axis=0) for cid in valid_clusters])
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
            cols[orphans] = np.argmax(Zn[orphans] @ centroids.T, axis=1)

        logger.info(
            f"Clusters: {len(cluster_counts)} raw -> {len(valid_clusters)} valid "
            f"(min_leaf_size={min_leaf_size}); reassigned {n_orphans}/{n_points} points "
            f"from undersized clusters. Sizes: {sorted(Counter(cols.tolist()).values(), reverse=True)}"
        )

        C_node = csr_matrix(
            (ones(n_points, dtype=dtype), (np.arange(n_points), cols)),
            shape=(n_points, len(valid_clusters)),
        )
        return C_node, clustering_model


def _selfcheck():
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


if __name__ == "__main__":
    _selfcheck()