import numpy as np
from xmr4el.eval import _selfcheck, ranking_metrics


def test_eval_selfcheck():
    _selfcheck()
    metrics = ranking_metrics([1, 2, 0], ks=(1, 5))
    assert metrics["acc@1"] == 1 / 3 and np.isclose(metrics["MRR"], 0.5)
    assert metrics["R@1"] == 1 / 3 and metrics["R@5"] == 2 / 3
