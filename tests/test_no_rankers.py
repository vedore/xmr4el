"""train_rankers=False skips every ranker and leaves cosine-scored predictions unchanged.
Small synthetic hierarchy, trained twice (with / without rankers); runs offline in seconds."""
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from xmr4el.encoder import LabelEmbeddingFactory
from xmr4el.node import MLModel
from xmr4el.tree import HierarchicalMLModel


def train(X, Y, Z, cfg, train_rankers, min_leaf_size=2, n_clusters=3, layer=2, cut_half=False):
    hml = HierarchicalMLModel(
        clustering_config={"type": "balancedkmeans", "kwargs": {"n_clusters": n_clusters, "iter_limit": 50}},
        matcher_config=cfg["matcher_config"], ranker_config=cfg["ranker_config"],
        cur_config=cfg["cur_config"], min_leaf_size=min_leaf_size, max_leaf_size=20,
        cut_half_cluster=cut_half, ranker_every_layer=True, n_workers=1, layer=layer,
        train_rankers=train_rankers,
    )
    L = Z.shape[0]
    caller_config = hml.clustering_config
    original = deepcopy(caller_config)
    with patch.object(hml, "prepare_layer", wraps=hml.prepare_layer) as prepare, \
         patch.object(MLModel, "fused_predict", side_effect=AssertionError("unused leaf fusion")):
        hml.train(X_train=X, Y_train=Y, Z_train=Z, local_to_global=np.arange(L),
                  global_to_local={i: i for i in range(L)})
    assert caller_config == original == hml.clustering_config, "training mutated clustering settings"
    assert prepare.call_count == sum(len(nodes) for nodes in hml.hmodel[:-1]), "prepared children of leaves"
    assert all(m.fused_scores is None for m in hml.hmodel[-1])
    assert hml.child_index_map[-1] == [{} for m in hml.hmodel[-1]]
    for leaf in hml.hmodel[-1]:
        leaf.cosine_scorer = True
    return hml


def test_no_rankers():
    cfg = json.load(open(Path(__file__).resolve().parents[1] / "configs/xmr4el_base_config.json"))
    rng = np.random.default_rng(0)
    L, per, d = 24, 10, 16
    centers = rng.normal(size=(L, d))
    y = np.repeat(np.arange(L), per)
    X = csr_matrix(normalize(centers[y] + 0.3 * rng.normal(size=(len(y), d))))
    Y = csr_matrix((np.ones(len(y)), (np.arange(len(y)), y)), shape=(len(y), L))
    Z = LabelEmbeddingFactory.generate_PIFA(X, Y)
    Xq = csr_matrix(normalize(centers + 0.3 * rng.normal(size=(L, d))))

    with_r, without_r = train(X, Y, Z, cfg, True), train(X, Y, Z, cfg, False)
    assert any(m.ranker_model is not None for layer in with_r.hmodel for m in layer)
    assert all(m.ranker_model is None for layer in without_r.hmodel for m in layer), "rankers trained"

    kw = dict(beam_size=2, topk=0, alpha=1.0)
    (_, a), (_, b) = with_r.predict(Xq, **kw), without_r.predict(Xq, **kw)  # (routes, scores_csr)
    a, b = a.toarray(), b.toarray()
    assert a.shape == b.shape and np.allclose(a, b), "cosine predictions must not depend on rankers"
    assert (a.argmax(axis=1) == np.arange(L)).mean() > 0.5, "synthetic labels should be easy"

    # per_leaf topk must keep each leaf's best labels: the top-1 under topk=1 equals the top-1 under topk=0
    (_, t1) = without_r.predict(Xq, beam_size=2, topk=1, alpha=1.0)
    t1 = t1.toarray()
    assert (t1 > 0).sum(axis=1).max() <= 2, "topk=1 keeps at most one label per visited leaf"
    assert (t1.argmax(axis=1) == b.argmax(axis=1)).all(), "per-leaf topk cut the wrong labels"

    # path_score: each leaf's score is multiplied by its routing path probability, nothing else changes
    kw0 = dict(beam_size=2, topk=0, alpha=0.0)
    (routes, off), (_, on) = without_r.predict(Xq, **kw0), without_r.predict(Xq, path_score=True, **kw0)
    expect = off.toarray()
    for r in routes:
        for p in r["paths"]:
            if p["trail"]:
                expect[r["query_index"], p["leaf_global_labels"]] *= np.exp(p["trail"][-1]["path_logscore"])
    assert np.allclose(on.toarray(), expect), "path_score must scale each leaf by exp(path_logscore)"
    assert not np.allclose(on.toarray(), off.toarray()), "beam 2 visits leaves with different path scores"

    # score matrix width is the label count, not the highest retrieved label + 1
    assert without_r.predict(Xq[:1], beam_size=1, topk=1)[1].shape == (1, L)

    # leaves too small to cluster (8 labels, min_leaf_size 8) still train with identity C
    small = train(X, Y, Z, cfg, False, min_leaf_size=8)
    assert len(small.hmodel) == 2 and small.predict(Xq, beam_size=2, topk=0)[1].shape == (L, L)

    # targets follow row order when a label repeats across groups
    assert LabelEmbeddingFactory.generate_label_matrix({"A": [0, 2], "B": [1]}) == [["A"], ["B"], ["A"]]

    # an internal node that cannot split fails loudly instead of returning a truncated tree
    try:
        train(X, Y, Z, cfg, False, min_leaf_size=30)
        raise AssertionError("unsplittable root must raise")
    except ValueError:
        pass

    # cut_half_cluster halves n_clusters below the root: 4 root clusters of 6 labels, then 2 per node
    deep = train(X, Y, Z, cfg, False, n_clusters=4, layer=3, cut_half=True)
    assert [m.cluster_model.c_node.shape[1] for m in deep.hmodel[1]] == [2] * 4
    with TemporaryDirectory() as d:
        deep.save(d)
        restored = HierarchicalMLModel.load(d)
    assert restored.clustering_config == deep.clustering_config
    assert restored.clustering_config["kwargs"]["n_clusters"] == 4
    assert np.allclose(deep.predict(Xq, **kw)[1].toarray(), restored.predict(Xq, **kw)[1].toarray())

    # global topk: final_path ranks the same labels as the returned scores
    routes, g = without_r.predict(Xq, beam_size=2, topk=1, topk_mode="global")
    for r in routes:
        row = g.getrow(r["query_index"])
        assert list(r["final_path"]["leaf_global_labels"]) == list(row.indices[np.argsort(-row.data)])
    print("ok")

