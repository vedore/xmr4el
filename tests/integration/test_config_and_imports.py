from copy import deepcopy
import os
import subprocess
import sys

import numpy as np

from xmr4el.data.readers import Preprocessor
from xmr4el.features.vectorizers import Vectorizer
from xmr4el.features.reduction import DimensionModel
from xmr4el.hierarchy.clusterers import ClusteringModel


def test_config_and_imports():
    for texts, labels in (([["x"]], []), ([], ["L"])):
        try:
            Preprocessor.prepare_data(texts, labels)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched inputs accepted")
    cfg = {"type": "tfidf", "kwargs": {"analyzer": "char", "ngram_range": [1, 2]}}
    original = deepcopy(cfg)
    X = Vectorizer.fit(["alpha", "beta", "gamma", "delta"], cfg).transform(["alpha", "beta", "gamma", "delta"])
    assert cfg == original
    for fit, data, cfg in (
        (DimensionModel.fit, X, {"type": "sklearntruncatedsvd", "kwargs": {"n_components": 2}}),
        (ClusteringModel.train, np.eye(4), {"type": "balancedkmeans", "kwargs": {"n_clusters": 2}}),
    ):
        original = deepcopy(cfg)
        fit(data, cfg)
        assert cfg == original
    code = """
import os, sys, warnings
before = {k: os.environ.get(k) for k in ('JOBLIB_TEMP_FOLDER', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
from xmr4el.xmodel import XModel
assert before == {k: os.environ.get(k) for k in before}
assert not any(getattr(f[1], 'pattern', '') == '.*does not have valid feature names.*' for f in warnings.filters)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
