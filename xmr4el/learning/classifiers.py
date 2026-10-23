import os
import json
import logging
import numbers
import pickle
import warnings
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack as sp_hstack, issparse
from scipy.special import expit
from threadpoolctl import threadpool_limits
from xmr4el import torch_device


logger = logging.getLogger(__name__)


class ClassifierModel:
    """Configured joint OvR logistic estimator and its persisted weights."""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    @staticmethod
    def check_config(config):
        if config.get("type") != "jointlogisticregression":
            raise ValueError(f"invalid classifier type {config.get('type')!r}; expected 'jointlogisticregression'")

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_config.json"), "w", encoding="utf-8") as fout:
            json.dump(self.config, fout)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, "classifier_config.json"), encoding="utf-8") as fin:
            config = json.load(fin)
        cls.check_config(config)
        with open(os.path.join(load_dir, "classifier_model.pkl"), "rb") as fin:
            return cls(config, pickle.load(fin))

    @classmethod
    def init_model(cls, config):
        cls.check_config(config)
        kwargs = {"C": 1.0, "class_weight": "balanced", "tol": 1e-3, "max_iter": 500, **config.get("kwargs", {})}
        return cls({"type": "jointlogisticregression", "kwargs": kwargs}, JointOvRLogistic(**kwargs))

    @classmethod
    def train(cls, X_train, y_train, config):
        wrapper = cls.init_model(config)
        with threadpool_limits(limits=1):
            wrapper.model.fit(X_train, y_train)
        return wrapper

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def classes(self):
        return self.model.classes_


# ---------------------------
# Joint one-vs-rest L2 logistic regression
# ---------------------------
# CUDA runs every fit (CPU only if it runs out of memory): on the G1 server CPU L-BFGS took 19-119 s per 1404-label
# leaf vs 2-7 s on CUDA. MPS runs fits with n * L >= this: per L-BFGS iteration 89929x128 0.20 -> 0.12 s, 20000x128
# break-even, 2064x92 leaves 2x slower (transfer and launch overhead).
GPU_MIN_ENTRIES = 5_000_000
# Labels per fit block: n * labels <= this (512 MB per float32 (n, labels) matrix). The largest G1 fit, a chemical
# root (~475k rows x 128 clusters, ~61M), is one block.
LABEL_BLOCK_ENTRIES = 2 ** 27
# Sparse X with density below this stays sparse in fit (torch CSR, CPU/CUDA) and predict (scipy CSR), so no dense (n, d)
# copy is built; denser X is expanded, where dense BLAS is faster. Both configs' X (SapBERT + SVD) is fully dense, so
# they keep the dense path. W stays dense, (d+1) x labels: a no-SVD TF-IDF d bounds memory through W instead of X.
SPARSE_MAX_DENSITY = 0.1


class JointOvRLogistic:
    """Every one-vs-rest L2 logistic problem of a node solved at once by torch L-BFGS (float32; on CUDA if available,
    on MPS if n * L >= GPU_MIN_ENTRIES; else CPU; CPU again after a CUDA/MPS out-of-memory).

    Per label the objective is liblinear's (`LogisticRegression(solver="liblinear")`): 0.5 ||w||^2 + C sum_i c_i
    log(1 + exp(-y_i w.x_i)), the bias a regularised constant-1 feature, c_i from `class_weight` "balanced" or 1.
    The labels share X, so all of them are one (d+1, L) problem; on a 167-label leaf this matched liblinear's top-1 on
    every dev row at 1/8 of the time. Nodes above LABEL_BLOCK_ENTRIES are fit in label blocks. `predict_proba`
    returns per-label sigmoids, (n, L), like a multilabel OneVsRestClassifier (what the matcher trains on its 0/1
    indicator matrix).
    """

    def __init__(self, C=1.0, class_weight="balanced", tol=1e-3, max_iter=500):
        bad = [f"{name} {v!r} (finite number > 0)" for name, v in (("C", C), ("tol", tol))
               if not (isinstance(v, numbers.Real) and not isinstance(v, bool) and 0 < v < float("inf"))]
        if not (isinstance(max_iter, numbers.Integral) and not isinstance(max_iter, bool) and max_iter >= 1):
            bad.append(f"max_iter {max_iter!r} (int >= 1)")
        if class_weight not in (None, "balanced"):
            bad.append(f"class_weight {class_weight!r} (None or 'balanced')")
        if bad:
            raise ValueError("JointOvRLogistic: " + "; ".join(bad))
        self.C, self.class_weight, self.tol, self.max_iter = C, class_weight, tol, max_iter

    @staticmethod
    def _dense(X):
        X = X if hasattr(X, "toarray") else np.asarray(X)
        # Column-major storage lets scipy fill the feature slice without another dense array.
        Xb = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float32, order="F")
        if hasattr(X, "toarray"):
            X.astype(np.float32, copy=False).toarray(out=Xb[:, :-1])
        else:
            Xb[:, :-1] = X
        Xb[:, -1] = 1
        return Xb

    @staticmethod
    def _features(X):
        """[X | 1] in float32: CSR if X is sparse with density < SPARSE_MAX_DENSITY, else `_dense`."""
        if issparse(X) and X.nnz < SPARSE_MAX_DENSITY * X.shape[0] * X.shape[1]:
            ones = csr_matrix(np.ones((X.shape[0], 1), dtype=np.float32))
            return sp_hstack([X.astype(np.float32, copy=False), ones], format="csr")
        return JointOvRLogistic._dense(X)

    @staticmethod
    def _csr_tensor(A, dev):
        idx = np.result_type(A.indptr, A.indices)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta")
            return torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(idx, copy=False)),
                                           torch.from_numpy(A.indices.astype(idx, copy=False)),
                                           torch.from_numpy(A.data), A.shape, check_invariants=False).to(dev)

    def fit(self, X, y):
        y = y.tocsc() if hasattr(y, "tocsc") else np.asarray(y)
        if y.ndim == 1:  # class labels -> indicator over sorted classes
            self.classes_, idx = np.unique(y, return_inverse=True)
        else:
            self.classes_ = np.arange(y.shape[1])
        Xb, L = self._features(X), len(self.classes_)
        n = Xb.shape[0]
        # float64 sum: finite iff every entry is (no full-size mask)
        if not n or y.shape[0] != n or not np.isfinite(Xb.sum(dtype=np.float64)):
            raise ValueError(f"JointOvRLogistic.fit needs >= 1 row, finite X and one y row per X row "
                             f"(X {Xb.shape[0]} rows, y {y.shape[0]})")
        # The per-label problems are independent: label blocks keep every (n, labels) matrix at <= LABEL_BLOCK_ENTRIES
        # entries. A node with fewer entries is one block (the joint fit); more blocks reach the same per-label
        # optimum by a different L-BFGS path (each block has its own line search and stopping test).
        block = max(1, LABEL_BLOCK_ENTRIES // n)
        W, self.n_iter_ = [], 0
        for a in range(0, L, block):
            b = min(a + block, L)
            if y.ndim == 1:
                pos = idx[:, None] == np.arange(a, b)
            else:
                yb = y[:, a:b]
                pos = (yb.toarray() if hasattr(yb, "toarray") else yb) > 0
            S = np.where(pos, 1.0, -1.0).astype(np.float32)
            if self.class_weight == "balanced":
                npos = pos.sum(axis=0)
                cw = np.where(pos, n / (2.0 * np.maximum(npos, 1)), n / (2.0 * np.maximum(n - npos, 1)))
            else:
                cw = np.ones_like(S)
            Wb, it = self._fit_block(Xb, S, (cw * self.C).astype(np.float32))
            if it >= self.max_iter:
                logger.warning("JointOvRLogistic stopped at max_iter=%d before max |grad| <= tol=%g (n=%d, labels "
                               "%d-%d of %d)", self.max_iter, self.tol, n, a, b, L)
            W.append(Wb)
            self.n_iter_ = max(self.n_iter_, it)
        self.W_ = W[0] if len(W) == 1 else np.hstack(W)
        return self

    def _fit_block(self, Xb, S, cw):
        # torch L-BFGS on CPU: scipy's L-BFGS-B step and numpy's single-threaded logaddexp/expit were ~95% of a
        # 6000x2304x734 leaf fit; torch runs both multithreaded (4x faster, 0.999 top-1 agreement). Stops when the
        # max |gradient| <= tol, scipy's gtol rule. The objective is divided by n (sklearn's lbfgs scaling, same
        # minimizer): unscaled, the gradient never reached tol and every fit ran to the float32 loss stall.
        n, L = S.shape
        dev = torch_device()
        if dev.type == "mps" and (n * L < GPU_MIN_ENTRIES or issparse(Xb)):  # MPS has no sparse CSR matmul
            dev = torch.device("cpu")
        try:
            return self._lbfgs(Xb, S, cw, dev)
        except RuntimeError as oom:  # torch.cuda.OutOfMemoryError subclasses RuntimeError; MPS raises a plain one
            if not isinstance(oom, torch.cuda.OutOfMemoryError) and "MPS backend out of memory" not in str(oom):
                raise
            logger.warning("JointOvRLogistic: %s out of memory (n=%d, L=%d), fitting on CPU", dev.type, n, L)
            if dev.type == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
            return self._lbfgs(Xb, S, cw, torch.device("cpu"))

    def _lbfgs(self, Xb, S, cw, dev):
        n, L = S.shape
        St, cwt = (torch.from_numpy(a).to(dev) for a in (S, cw))
        if issparse(Xb):  # X.T kept as its own CSR: torch has no CSR.T @ dense on every device
            Xt, XTt = self._csr_tensor(Xb, dev), self._csr_tensor(Xb.T.tocsr(), dev)
        else:
            Xt = torch.from_numpy(Xb).to(dev)
            XTt = Xt.T
        W = torch.zeros(Xb.shape[1], L, dtype=torch.float32, device=dev)
        opt = torch.optim.LBFGS([W], lr=1, max_iter=self.max_iter, tolerance_grad=self.tol, tolerance_change=1e-9,
                                history_size=10, line_search_fn="strong_wolfe")

        def closure():
            M = St * (Xt @ W)
            W.grad = (W + XTt @ (-cwt * St * torch.sigmoid(-M))) / n
            return (0.5 * (W * W).sum() + (cwt * torch.nn.functional.softplus(-M)).sum()) / n

        with torch.no_grad():
            opt.step(closure)
        return W.cpu().numpy(), opt.state[W]["n_iter"]

    def decision_function(self, X):
        return np.asarray(self._features(X) @ self.W_)

    def predict_proba(self, X):
        return expit(self.decision_function(X))

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]
