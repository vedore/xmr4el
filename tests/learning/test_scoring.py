import logging
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import csr_matrix, eye

from xmr4el.hierarchy.node import MLModel
from xmr4el.learning.ranker import ranker_input


def test_scoring():
    X = np.array([[1., 0.], [0., 1.], [0., 0.]])
    Z = np.eye(2)
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    node = MLModel()
    node.local_to_global_idx = np.array([4, 7])
    node.global_to_local_idx = {4: 0, 7: 1}
    node.cluster_model = SimpleNamespace(c_node=eye(2, format="csr"))
    node.matcher_model = Mock(predict_proba=lambda rows: probabilities)
    node.label_embeddings = Z
    expected = (X @ Z.T / (np.linalg.norm(X, axis=1)[:, None] + 1e-12) + 1) / 2
    for fusion in ("lp_fusion", "geometric"):
        for alpha in (0, 0.5, 1):
            # Cosine needs no dense copy of the sparse query.
            with patch.object(csr_matrix, "toarray", side_effect=AssertionError("densified cosine input")):
                sparse_scores, labels = node.predict(csr_matrix(X), beam_size=2, fusion=fusion, alpha=alpha, scorer="cosine")
            dense_scores, _ = node.predict(X, beam_size=2, fusion=fusion, alpha=alpha, scorer="cosine")
            m, r = np.clip(probabilities, 1e-6, 1), np.clip(expected, 1e-6, 1)
            want = ((m**3 * (1-alpha) + r**3 * alpha)**(1/3)
                    if fusion == "lp_fusion" else m**(1-alpha) * r**alpha)
            assert np.allclose(sparse_scores, want) and np.allclose(dense_scores, want)
            assert all(row.tolist() == [4, 7] for row in labels)
    np.testing.assert_array_equal(ranker_input(csr_matrix(X), Z[0]).toarray(), ranker_input(X, Z[0]))
    ranker = Mock(config={})
    ranker.predict_proba.side_effect = lambda rows: np.tile([0.1, 0.9], (len(rows), 1))
    node.ranker_model = SimpleNamespace(model_dict={4: ranker, 7: ranker})
    node.cosine_scorer = True  # explicit arguments override the legacy saved switch
    scores, _ = node.predict(csr_matrix(X), beam_size=2, alpha=1, scorer="ranker")
    assert np.allclose(scores, 0.9)
    before = ranker.predict_proba.call_count
    node.predict(csr_matrix(X), beam_size=2, scorer="cosine")
    assert ranker.predict_proba.call_count == before
    ranker.predict_proba.side_effect = RuntimeError("broken ranker")
    with TestCase().assertLogs(node.logger, level=logging.WARNING) as logs:
        scores, _ = node.predict(csr_matrix(X), beam_size=2, alpha=1, scorer="ranker")
    assert node.ranker_failed == {4, 7} and "broken ranker" in logs.output[0]
    assert np.allclose(scores, np.clip(expected, 1e-6, 1))
    try:
        node.predict(X, scorer="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid scorer accepted")
