"""Concrete adapters reject unsupported backends and preserve configured behavior on disk."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from xmr4el.features.reduction import DimensionModel
from xmr4el.features.vectorizers import Vectorizer
from xmr4el.hierarchy.clusterers import ClusteringModel
from xmr4el.learning.classifiers import ClassifierModel


@pytest.mark.parametrize("adapter, fit, data, config", [
    (Vectorizer, Vectorizer.fit, ["alpha", "beta", "gamma"],
     {"type": "tfidf", "kwargs": {"analyzer": "char", "ngram_range": [1, 2]}}),
    (DimensionModel, DimensionModel.fit, csr_matrix(np.eye(3, dtype=np.float32)),
     {"type": "sklearntruncatedsvd", "kwargs": {"n_components": 2}}),
    (DimensionModel, DimensionModel.fit, csr_matrix(np.eye(3, dtype=np.float32)),
     {"type": "sklearntruncatedsvd", "kwargs": {"n_components": 10}}),
])
def test_transform_round_trip(tmp_path, adapter, fit, data, config):
    original = deepcopy(config)
    model = fit(data, config)
    before = model.transform(data)
    model.save(tmp_path)
    after = adapter.load(tmp_path).transform(data)
    assert config == original
    assert issparse(before) == issparse(after)
    if issparse(before):
        before, after = before.toarray(), after.toarray()
    assert np.array_equal(before, after)


@pytest.mark.parametrize("adapter", [Vectorizer, DimensionModel, ClusteringModel, ClassifierModel])
@pytest.mark.parametrize("config", [{}, {"type": "unknown", "kwargs": {}}])
def test_unsupported_backends(adapter, config):
    with pytest.raises(ValueError, match="type"):
        adapter.check_config(config)
