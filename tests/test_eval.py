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


def test_strings_hash_follows_the_effective_dictionary(tmp_path):
    """Same test rows and reference file, other vocabulary: the dictionary changes, so does strings_sha256 (R01)."""
    test, train = tmp_path / "test.txt", tmp_path / "train.txt"
    test.write_text("1|t|alpha\n1|a|x\n1\t0\t5\talpha\tType\tA\n\n", encoding="utf-8")
    train.write_text("2|t|alpha alpha\n2|a|x\n2\t0\t5\talpha\tType\tA\n2\t6\t11\talpha\tType\tB\n\n", encoding="utf-8")
    both, _ = evaluate_tree(StubTree(["A", "B"]), "t", str(test), train_path=str(train))
    only_a, _ = evaluate_tree(StubTree(["A", "C"]), "t", str(test), train_path=str(train))
    assert both["population_sha256"] == only_a["population_sha256"]
    assert both["strings"]["seen, >1 label"]["n"] == 1 and only_a["strings"]["seen, 1 label"]["n"] == 1
    assert both["strings_sha256"] != only_a["strings_sha256"]
    assert both["eval_protocol"] == only_a["eval_protocol"]


def test_zero_sample_metrics(tmp_path):
    """No row in vocabulary: acc@1_all is a defined 0; an empty string group has null rates, never 0 (R11)."""
    test, train = tmp_path / "test.txt", tmp_path / "train.txt"
    test.write_text("1|t|alpha\n1|a|x\n1\t0\t5\talpha\tType\tA\n\n", encoding="utf-8")
    train.write_text("2|t|alpha\n2|a|x\n2\t0\t5\talpha\tType\tA\n\n", encoding="utf-8")
    none, state = evaluate_tree(StubTree(["Z"]), "t", str(test))
    assert state is None and none["rows"] == 0 and none["acc@1_all"] == 0.0 and "acc@1" not in none
    m, _ = evaluate_tree(StubTree(["A"]), "t", str(test), train_path=str(train))
    assert m["strings"]["seen, >1 label"] == {"n": 0, "tree": None, "dict": None, "either": None, "routed": None}
    assert m["strings"]["seen, 1 label"]["dict"] == 1.0
