"""train_rankers=False skips every ranker and leaves predictions unchanged.
Small synthetic hierarchy, trained twice (with / without rankers); runs offline in seconds."""
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from xmr4el.features.label_embeddings import LabelEmbeddingFactory
from xmr4el.hierarchy.tree import HierarchicalMLModel, augment_features


def train(X, Y, Z, cfg, train_rankers, min_leaf_size=2, n_clusters=3, layer=2, cut_half=False):
    hml = HierarchicalMLModel(
        clustering_config={"type": "balancedkmeans", "kwargs": {"n_clusters": n_clusters, "iter_limit": 20}},
        matcher_config=cfg["matcher_config"], ranker_config=cfg["ranker_config"],
        cur_config=cfg["cur_config"], min_leaf_size=min_leaf_size, max_leaf_size=20,
        cut_half_cluster=cut_half, ranker_every_layer=True, n_workers=1, layer=layer,
        train_rankers=train_rankers,
    )
    L = Z.shape[0]
    caller_config = hml.clustering_config
    original = deepcopy(caller_config)
    with patch.object(hml, "prepare_layer", wraps=hml.prepare_layer) as prepare:
        hml.train(X_train=X, Y_train=Y, Z_train=Z, local_to_global=np.arange(L),
                  global_to_local={i: i for i in range(L)})
    assert caller_config == original == hml.clustering_config, "training mutated clustering settings"
    assert prepare.call_count == sum(len(nodes) for nodes in hml.hmodel[:-1]), "prepared children of leaves"
    assert all(m.fused_scores is None for m in hml.hmodel[-1])
    assert hml.child_index_map[-1] == [{} for m in hml.hmodel[-1]]
    return hml


def test_no_rankers():
    cfg = json.load(open(Path(__file__).resolve().parents[2] / "configs/xmr4el_base_config.json"))
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

    kw = dict(beam_size=2, topk=0)
    a, b = with_r.predict(Xq, **kw).toarray(), without_r.predict(Xq, **kw).toarray()
    assert a.shape == b.shape and np.allclose(a, b), "predictions must not depend on rankers"
    assert (b.argmax(axis=1) == np.arange(L)).mean() > 0.5, "synthetic labels should be easy"

    # topk keeps each row's best labels
    t1 = without_r.predict(Xq, beam_size=2, topk=1).toarray()
    assert ((t1 > 0).sum(axis=1) == 1).all() and (t1.argmax(axis=1) == b.argmax(axis=1)).all()

    # score = root matcher prob of the label's cluster x leaf matcher prob (beam 3 visits all 3 leaves)
    P = np.asarray(without_r.hmodel[0][0].matcher_model.predict_proba(Xq))
    expect = np.zeros((L, L))
    for c, child in without_r.child_index_map[0][0].items():
        leaf = without_r.hmodel[1][child]
        Xa = augment_features(Xq, P[:, c], P.sum(axis=1), P.max(axis=1))
        expect[:, leaf.local_to_global_idx] = P[:, [c]] * leaf.matcher_model.predict_proba(Xa)
    assert np.allclose(without_r.predict(Xq, beam_size=3).toarray(), expect)

    # score matrix width is the label count, not the highest retrieved label + 1
    assert without_r.predict(Xq[:1], beam_size=1, topk=1).shape == (1, L)

    # leaves too small to cluster (8 labels, min_leaf_size 8) still train with identity C
    small = train(X, Y, Z, cfg, False, min_leaf_size=8)
    assert len(small.hmodel) == 2 and small.predict(Xq, beam_size=2).shape == (L, L)

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
    assert np.allclose(deep.predict(Xq, **kw).toarray(), restored.predict(Xq, **kw).toarray())

    print("ok")

