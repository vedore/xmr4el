"""train_rankers=False skips every ranker and leaves cosine-scored predictions unchanged.
Small synthetic hierarchy, trained twice (with / without rankers); runs offline in seconds."""
import json

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from xmr4el.featurization.label_embedding_factory import LabelEmbeddingFactory
from xmr4el.xmr.base import HierarchicaMLModel


def train(X, Y, Z, cfg, train_rankers):
    hml = HierarchicaMLModel(
        clustering_config={"type": "balancedkmeans", "kwargs": {"n_clusters": 3, "iter_limit": 50}},
        matcher_config=cfg["matcher_config"], ranker_config=cfg["ranker_config"],
        cur_config=cfg["cur_config"], min_leaf_size=2, max_leaf_size=20,
        cut_half_cluster=False, ranker_every_layer=True, n_workers=1, layer=2,
        train_rankers=train_rankers,
    )
    L = Z.shape[0]
    hml.train(X_train=X, Y_train=Y, Z_train=Z, local_to_global=np.arange(L),
              global_to_local={i: i for i in range(L)})
    for leaf in hml.hmodel[-1]:
        leaf.cosine_scorer = True
    return hml


def main():
    cfg = json.load(open(".models/xmr4el_base_config.json"))
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
    print("ok")


if __name__ == "__main__":
    main()
