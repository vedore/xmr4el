import numpy as np
from scipy.sparse import csr_matrix
from xmr4el.learning.ranker import RankerTrainer


def test_ranker_curriculum():
    """Asserts the two things the curriculum degrades silently on: models are carried
    across epochs (not retrained from scratch) and epochs draw different negatives."""
    rs = np.random.RandomState(0)
    n, d, L = 200, 8, 3
    X = csr_matrix(rs.rand(n, d).astype(np.float32))
    Y = csr_matrix((rs.rand(n, L) > 0.8).astype(np.int8))
    Z = rs.rand(L, d).astype(np.float32)
    M_TFN = np.ones((n, 1), dtype=int)
    cfg = {"type": "sklearnsgdclassifier",
           "kwargs": {"loss": "hinge", "learning_rate": "adaptive", "eta0": 0.01, "random_state": 42}}
    cur = {"E_warm": 1, "ratios_warm": (0.7, 0.3), "ratios_hard": (0.6, 0.3, 0.1),
           "neg_mult": 5, "seed": 42}

    def run(n_epochs):
        return RankerTrainer.train(X=X, Y=Y, Z=Z, M_TFN=M_TFN, M_MAN=None,
                                   cluster_labels=np.zeros(L, dtype=int), config=cfg,
                                   cur_config=cur, local_to_global_idx=np.arange(L),
                                   n_label_workers=1, n_epochs=n_epochs)

    one, three = run(1), run(3)
    assert one and three, "no rankers trained"
    gid = next(iter(one))
    # SGD's t_ counts samples seen; 3 epochs must accumulate more than 1 if warm-started
    assert three[gid].model.model.t_ > one[gid].model.model.t_, "epochs retrain from scratch, not warm-started"

    picks = [RankerTrainer._select_negatives_curriculum(
                epoch=e, positive_positions=np.arange(5), neg_positions_all=np.arange(5, 200),
                cos_scores_neg=rs.rand(195), ip_scores_neg=rs.rand(195),
                E_warm=cur["E_warm"], ratios_warm=cur["ratios_warm"],
                ratios_hard=cur["ratios_hard"], neg_mult=cur["neg_mult"],
                rng=np.random.RandomState(cur["seed"] + e), cluster_size=200)
             for e in (1, 2)]
    assert not np.array_equal(picks[0], picks[1]), "epochs draw identical negatives"
    print("ranker selfcheck ok")
