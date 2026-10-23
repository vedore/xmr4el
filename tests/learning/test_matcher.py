"""Local target order is independent of global label IDs; rows without positives stay excluded."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from xmr4el.hierarchy.node import MLModel
from xmr4el.learning.classifiers import ClassifierModel


@pytest.mark.parametrize("sparse", [False, True])
def test_node_targets_and_persistence(tmp_path, sparse):
    X = np.tile(np.eye(3, dtype=np.float32), (10, 1))
    Y = csr_matrix(X)
    config = {"type": "jointlogisticregression", "kwargs": {}}
    expected = ClassifierModel.train(X, Y, config).predict_proba(X)
    features = np.vstack([X, np.full((1, 3), np.nan, dtype=np.float32)])
    if sparse:
        features = csr_matrix(features)
    targets = csr_matrix(np.vstack([X, np.zeros((1, 3))]))
    node = MLModel(is_last_layer=True, matcher_config=config)
    node.train(features, targets, np.eye(3, dtype=np.float32), np.array([9, 3, 7]))
    node.save(tmp_path)
    restored = MLModel.load(tmp_path)
    assert restored.local_to_global_idx.tolist() == [9, 3, 7]
    assert np.allclose(restored.matcher_model.predict_proba(features[:-1]), expected)
    assert restored.local_to_global_idx[expected.argmax(1)].tolist() == [9, 3, 7] * 10
