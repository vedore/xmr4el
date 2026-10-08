import os
import json
import pickle
import numpy as np
from abc import ABCMeta
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from xmr4el import torch_device


classifier_dict = {}



class ClassifierMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of ClassifierModel.
    Automatically registers each subclass in classifier_dict.
    """
    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "ClassifierModel":
            classifier_dict[name.lower()] = new_cls
        return new_cls


class ClassifierModel(metaclass=ClassifierMeta):
    """Wrapper for all classifier models"""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def save(self, classifier_folder):
        """Save classifier model to disk."""
        os.makedirs(classifier_folder, exist_ok=True)
        with open(
            os.path.join(classifier_folder, "classifier_config.json"),
            "w",
            encoding="utf-8",
        ) as fout:
            fout.write(json.dumps(self.config))
        self.model.save(classifier_folder)

    @classmethod
    def load(cls, classifier_folder):
        """Load a saved classifier model from disk."""
        with open(os.path.join(classifier_folder, "classifier_config.json"), "r", encoding="utf-8") as fin:
            config = json.loads(fin.read())

        classifier_type = config.get("type", None)
        assert classifier_type is not None, f"{classifier_folder} is not a valid classifier folder"
        assert classifier_type in classifier_dict, f"invalid classifier type {config['type']}"
        model = classifier_dict[classifier_type].load(classifier_folder, config)
        return cls(config, model)

    @classmethod
    def init_model(cls, config, onevsrest=False):
        """
        Initialize (but do not fit) a classifier, returning a ClassifierModel wrapper.
        """
        classifier_type = config.get("type", None)
        assert classifier_type is not None, f"config {config} should contain a key 'type' for the classifier type"

        kwargs = config.get("kwargs", {})
        # delegate to the subclass's init_model (required for all subclasses)
        model = classifier_dict[classifier_type].init_model(kwargs, onevsrest=onevsrest)
        # keep the possibly-updated kwargs from the subclass instance
        out_cfg = {"type": classifier_type, "kwargs": model.config}
        return cls(out_cfg, model)

    @classmethod
    def train(cls, X_train, y_train, config, dtype=np.float32, onevsrest=False):
        """
        Train using the already-initialized model from init_model().
        Works regardless of whether you call ClassifierModel.train(...) or a subclass's train(...).
        """
        classifier_type = config.get("type", None)
        assert classifier_type is not None, f"config {config} should contain a key 'type' for the classifier type"

        # Initialize via subclass init_model, then fit.
        wrapper = cls.init_model(config, onevsrest=onevsrest)
        # dtype is unused here, but retained for API compatibility
        with threadpool_limits(limits=1):
            wrapper.model.model.fit(X_train, y_train)
        # sync any updated params back
        wrapper.config["kwargs"] = wrapper.model.config
        return wrapper

    # Delegations to underlying model object
    def predict(self, predict_input):
        return self.model.predict(predict_input)

    def decision_function(self, predict_input):
        return self.model.decision_function(predict_input)

    def predict_proba(self, predict_input):
        return self.model.predict_proba(predict_input)

    def classes(self):
        return self.model.classes()

    

# ---------------------------
# Joint one-vs-rest L2 logistic regression
# ---------------------------
# Fits with n * L >= this run on `torch_device()`. MPS per L-BFGS iteration: 89929x128 0.20 -> 0.12 s, 20000x128
# break-even, 2064x92 leaves 2x slower (transfer and launch overhead).
# ponytail: threshold measured on MPS only; re-measure on a CUDA box, where the break-even is likely lower.
GPU_MIN_ENTRIES = 5_000_000


class JointOvRLogistic:
    """Every one-vs-rest L2 logistic problem of a node solved at once by torch L-BFGS (float32; CPU, or
    `torch_device()` if n * L >= GPU_MIN_ENTRIES).

    Per label the objective is liblinear's (`LogisticRegression(solver="liblinear")`): 0.5 ||w||^2 + C sum_i c_i
    log(1 + exp(-y_i w.x_i)), the bias a regularised constant-1 feature, c_i from `class_weight` "balanced" or 1.
    The labels share X, so all of them are one (d+1, L) problem; on a 167-label leaf this matched liblinear's top-1 on
    every dev row at 1/8 of the time. `predict_proba` returns per-label sigmoids, (n, L), like a multilabel
    OneVsRestClassifier (what the matcher trains on its 0/1 indicator matrix).
    """

    def __init__(self, C=1.0, class_weight="balanced", tol=1e-3, max_iter=500):
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

    def fit(self, X, y):
        if self.class_weight not in (None, "balanced"):
            raise ValueError("JointOvRLogistic class_weight must be None or 'balanced'")
        y = y.toarray() if hasattr(y, "toarray") else np.asarray(y)
        if y.ndim == 1:  # class labels -> indicator over sorted classes
            self.classes_, idx = np.unique(y, return_inverse=True)
            y = np.eye(len(self.classes_), dtype=np.float32)[idx]
        else:
            self.classes_ = np.arange(y.shape[1])
        Xb, n, L = self._dense(X), y.shape[0], y.shape[1]
        S = np.where(y > 0, 1.0, -1.0).astype(np.float32)
        if self.class_weight == "balanced":
            npos = (y > 0).sum(axis=0)
            cw = np.where(y > 0, n / (2.0 * np.maximum(npos, 1)), n / (2.0 * np.maximum(n - npos, 1)))
        else:
            cw = np.ones_like(S)
        cw = (cw * self.C).astype(np.float32)

        # torch L-BFGS on CPU: scipy's L-BFGS-B step and numpy's single-threaded logaddexp/expit were ~95% of a
        # 6000x2304x734 leaf fit; torch runs both multithreaded (4x faster, 0.999 top-1 agreement). Stops when the
        # max |gradient| <= tol, scipy's gtol rule. The objective is divided by n (sklearn's lbfgs scaling, same
        # minimizer): unscaled, the gradient never reached tol and every fit ran to the float32 loss stall.
        dev = torch_device() if n * L >= GPU_MIN_ENTRIES else torch.device("cpu")
        Xt, St, cwt = (torch.from_numpy(a).to(dev) for a in (Xb, S, cw))
        W = torch.zeros(Xb.shape[1], L, device=dev)
        opt = torch.optim.LBFGS([W], lr=1, max_iter=self.max_iter, tolerance_grad=self.tol, tolerance_change=1e-9,
                                history_size=10, line_search_fn="strong_wolfe")

        def closure():
            M = St * (Xt @ W)
            W.grad = (W + Xt.T @ (-cwt * St * torch.sigmoid(-M))) / n
            return (0.5 * (W * W).sum() + (cwt * torch.nn.functional.softplus(-M)).sum()) / n

        with torch.no_grad():
            opt.step(closure)
        self.W_, self.n_iter_ = W.cpu().numpy(), opt.state[W]["n_iter"]
        return self

    def decision_function(self, X):
        return self._dense(X) @ self.W_

    def predict_proba(self, X):
        return expit(self.decision_function(X))

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


class JointLogisticRegression(ClassifierModel):
    """Config type "jointlogisticregression": `JointOvRLogistic` (kwargs C, class_weight, tol, max_iter). Always
    one-vs-rest, so `onevsrest` is ignored."""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        with open(os.path.join(load_dir, "classifier_model.pkl"), "rb") as fin:
            return cls(config, pickle.load(fin))

    @classmethod
    def init_model(cls, config, onevsrest=False):
        cfg = {"C": 1.0, "class_weight": "balanced", "tol": 1e-3, "max_iter": 500, **(config or {})}
        return cls(cfg, JointOvRLogistic(**cfg))

    @classmethod
    def train(cls, X_train, y_train, config=None, dtype=np.float32, onevsrest=False):
        wrapper = cls.init_model(config or {}, onevsrest=onevsrest)
        wrapper.model.fit(X_train, y_train)
        return wrapper

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def classes(self):
        return self.model.classes_
