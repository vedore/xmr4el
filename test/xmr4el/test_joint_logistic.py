"""JointOvRLogistic solves liblinear's one-vs-rest L2 logistic objective for all labels at once: same probabilities as
sklearn's liblinear OneVsRest on a 0/1 indicator target, same shapes through the ClassifierModel wrapper. Offline."""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel, JointOvRLogistic


def main():
    rng = np.random.default_rng(0)
    L, d, per = 7, 20, 25
    y = np.repeat(np.arange(L), per)
    X = normalize(rng.normal(size=(L, d))[y] + 0.8 * rng.normal(size=(len(y), d)))
    Y = np.eye(L)[y]

    for features in (X, X.tolist(), csr_matrix(X), csc_matrix(X.astype(np.float32))):
        dense = JointOvRLogistic._dense(features)
        assert dense.dtype == np.float32 and dense.shape == (len(y), d + 1)
        assert np.array_equal(dense[:, :-1], X.astype(np.float32)) and (dense[:, -1] == 1).all()
    for cw in ("unsupported", {0: 1, 1: 2}, 1):
        try:
            JointOvRLogistic(class_weight=cw).fit(X, Y)
        except ValueError as e:
            assert "class_weight must be None or 'balanced'" in str(e)
        else:
            raise AssertionError("unsupported class_weight accepted")

    for cw in ("balanced", None):
        ref = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=1.0, class_weight=cw, tol=1e-8)).fit(X, Y)
        joint = JointOvRLogistic(C=1.0, class_weight=cw, tol=1e-8, max_iter=2000).fit(csr_matrix(X), csr_matrix(Y))
        P, Q = ref.predict_proba(X), joint.predict_proba(X)
        assert ref.multilabel_ and P.shape == Q.shape == (len(y), L)
        assert np.abs(P - Q).max() < 1e-3, (cw, np.abs(P - Q).max())
        assert (P.argmax(1) == Q.argmax(1)).all()

    w = ClassifierModel.train(csr_matrix(X), csr_matrix(Y), {"type": "jointlogisticregression", "kwargs": {}})
    assert w.predict_proba(X).shape == (len(y), L) and list(w.classes()) == list(range(L))
    m = JointOvRLogistic().fit(X, y)  # 1-D class labels
    assert (m.predict(X) == y).mean() > 0.8 and list(m.classes_) == list(range(L))
    print("ok")


if __name__ == "__main__":
    main()
