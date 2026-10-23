import numpy as np
from scipy.sparse import csr_matrix
from xmr4el.eval import _selfcheck, evaluate_tree, ranking_metrics


def test_eval_selfcheck():
    _selfcheck()
    metrics = ranking_metrics([1, 2, 0], ks=(1, 5))
    assert metrics["acc@1"] == 1 / 3 and np.isclose(metrics["MRR"], 0.5)
    assert metrics["R@1"] == 1 / 3 and metrics["R@5"] == 2 / 3


class StubTree:
    """Just what evaluate_tree reads; predict returns no candidates."""
    context_window = 5
    abbrev_expansion = None
    features = "tfidf"

    def __init__(self, labels):
        self.initial_labels = labels

    def resolve_predict_config(self, **overrides):
        return overrides

    def predict(self, texts, **search):
        return csr_matrix((len(texts), len(self.initial_labels)))


def test_population_hash(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("1|t|alpha beta gamma\n1|a|x\n"
                    "1\t0\t5\talpha\tType\tL1\n1\t6\t10\tbeta\tType\tL2\n1\t11\t16\tgamma\tType\tL1\n\n",
                    encoding="utf-8")
    m, _ = evaluate_tree(StubTree(["L1", "L2"]), "t", str(path))
    again, _ = evaluate_tree(StubTree(["L1", "L2"]), "t", str(path))
    only_l1, _ = evaluate_tree(StubTree(["L1"]), "t", str(path))
    only_l2, _ = evaluate_tree(StubTree(["L2", "L3"]), "t", str(path))
    assert m["population_sha256"] == again["population_sha256"]
    assert len({m["population_sha256"], only_l1["population_sha256"], only_l2["population_sha256"]}) == 3
    assert m["preprocessing"] == {"context_window": 5, "abbrev_expansion": None}
    assert (m["rows"], m["acc@1"], m["R@cand"]) == (3, 0.0, 0.0)
