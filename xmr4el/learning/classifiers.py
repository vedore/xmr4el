import os
import json
import pickle
import multiprocessing
import numpy as np
import joblib
from abc import ABCMeta
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from typing import Any, Dict, List, Tuple, Optional


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
        config_path = os.path.join(classifier_folder, "classifier_config.json")

        if not os.path.exists(config_path):
            config = {"type": "sklearnlogisticregression", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())

        classifier_type = config.get("type", None)
        assert classifier_type is not None, f"{classifier_folder} is not a valid classifier folder"
        assert classifier_type in classifier_dict, f"invalid classifier type {config['type']}"
        model = classifier_dict[classifier_type].load(classifier_folder, config)
        return cls(config, model)

    @classmethod
    def init_model(cls, config=None, onevsrest=False):
        """
        Initialize (but do not fit) a classifier, returning a ClassifierModel wrapper.
        """
        config = config if config is not None else {"type": "sklearnlogisticregression", "kwargs": {}}
        classifier_type = config.get("type", None)
        assert classifier_type is not None, f"config {config} should contain a key 'type' for the classifier type"

        kwargs = config.get("kwargs", {})
        # delegate to the subclass's init_model (required for all subclasses)
        model = classifier_dict[classifier_type].init_model(kwargs, onevsrest=onevsrest)
        # keep the possibly-updated kwargs from the subclass instance
        out_cfg = {"type": classifier_type, "kwargs": model.config}
        return cls(out_cfg, model)

    @classmethod
    def train(cls, X_train, y_train=None, config=None, dtype=np.float32, onevsrest=False):
        """
        Train using the already-initialized model from init_model().
        Works regardless of whether you call ClassifierModel.train(...) or a subclass's train(...).
        """
        config = config if config is not None else {"type": "sklearnlogisticregression", "kwargs": {}}
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

    def partial_fit(self, X, Y, classes, dtype):
        self.model.partial_fit(X, Y, classes, dtype)

    # Delegations to underlying model object
    def predict(self, predict_input):
        return self.model.predict(predict_input)

    def decision_function(self, predict_input):
        return self.model.decision_function(predict_input)

    def predict_proba(self, predict_input):
        return self.model.predict_proba(predict_input)

    def classes(self):
        return self.model.classes()

    def coef(self):
        return self.model.coef()

    def intercept(self):
        return self.model.intercept()

    def is_linear_model(self):
        return self.model.is_linear_model()
    
    def supports_partial_fit(self) -> bool:
        """Whether this model can be updated incrementally via partial_fit."""
        return self.model.supports_partial_fit

    

# ---------------------------
# Sklearn Logistic Regression
# ---------------------------
class SklearnLogisticRegression(ClassifierModel):
    """Sklearn Logistic Regression"""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        classifier_path = os.path.join(load_dir, "classifier_model.pkl")
        assert os.path.exists(classifier_path), f"Classifier path {classifier_path} does not exist"
        with open(classifier_path, "rb") as fin:
            model_data = pickle.load(fin)
        return cls(config, model_data)

    @classmethod
    def init_model(cls, config, onevsrest=False):
        defaults = {  # sklearn >= 1.8: L2 is the default via l1_ratio; penalty and n_jobs are deprecated
            "dual": False,
            "tol": 0.0001,
            "C": 1.0,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": None,
            "random_state": None,
            "solver": "lbfgs",
            "max_iter": 100,
            "verbose": 0,
            "warm_start": False,
        }
        cfg = {**defaults, **(config or {})}
        est = LogisticRegression(**cfg)
        if onevsrest:
            est = OneVsRestClassifier(est, n_jobs=multiprocessing.cpu_count())
        return cls(cfg, est)

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
        return self.model.classes_ if hasattr(self.model, "classes_") else self.model.classes

    def coef(self):
        return self.model.coef_ if hasattr(self.model, "coef_") else None

    def intercept(self):
        return self.model.intercept_ if hasattr(self.model, "intercept_") else None

    def is_linear_model(self):
        return True

    def supports_partial_fit(self) -> bool:
        return False

# ---------------------------
# Sklearn SGDClassifier
# ---------------------------
class SklearnSGDClassifier(ClassifierModel):
    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        classifier_path = os.path.join(load_dir, "classifier_model.pkl")
        assert os.path.exists(classifier_path), f"Classifier path {classifier_path} does not exist"
        with open(classifier_path, "rb") as fin:
            model_data = pickle.load(fin)
        return cls(config, model_data)

    @classmethod
    def init_model(cls, config, onevsrest=False):
        defaults = {
            "loss": 'hinge',
            "penalty": 'l2',
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 0.001,
            "shuffle": True,
            "verbose": 0,
            "epsilon": 0.1,
            "n_jobs": None,
            "random_state": None,
            "learning_rate": 'optimal',
            "power_t": 0.5,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "class_weight": None,
            "warm_start": False,
            "average": False
        }
        cfg = {**defaults, **(config or {})}
        est = SGDClassifier(**cfg)
        if onevsrest:
            est = OneVsRestClassifier(est, n_jobs=cfg.get("n_jobs", None))
        return cls(cfg, est)

    @classmethod
    def train(cls, X_train, y_train, config=None, dtype=np.float32, onevsrest=False):
        wrapper = cls.init_model(config or {}, onevsrest=onevsrest)
        wrapper.model.fit(X_train, y_train)
        return wrapper
    
    def partial_fit(self, X, Y, classes, dtype):
        self.model.partial_fit(X, Y, classes)

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def classes(self):
        return self.model.classes_

    def coef(self):
        return self.model.coef_

    def intercept(self):
        return self.model.intercept_

    def is_linear_model(self):
        return True
    
    def supports_partial_fit(self) -> bool:
        return True 


# ---------------------------
# Joint one-vs-rest L2 logistic regression
# ---------------------------
class JointOvRLogistic:
    """Every one-vs-rest L2 logistic problem of a node solved at once by L-BFGS on BLAS matrix products.

    Per label the objective is liblinear's (`LogisticRegression(solver="liblinear")`): 0.5 ||w||^2 + C sum_i c_i
    log(1 + exp(-y_i w.x_i)), the bias a regularised constant-1 feature, c_i from `class_weight` "balanced" or 1.
    The labels share X, so all of them are one (d+1, L) problem; on a 167-label leaf this matched liblinear's top-1 on
    every dev row at 1/8 of the time. `predict_proba` returns per-label sigmoids, (n, L), like a multilabel
    OneVsRestClassifier (what the matcher trains on its 0/1 indicator matrix).
    """

    def __init__(self, C=1.0, class_weight="balanced", tol=1e-4, max_iter=500):
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

        def f(w):
            W = w.reshape(-1, L).astype(np.float32)
            M = S * (Xb @ W)
            loss = 0.5 * float((W * W).sum()) + float((cw * np.logaddexp(0, -M)).sum())
            return loss, (W + Xb.T @ (-cw * S * expit(-M))).ravel().astype(np.float64)

        # This module pins OpenBLAS/MKL to 1 thread for the process-parallel sklearn wrappers (Linux numpy); the joint
        # solver is one BLAS-bound problem, so it lifts the cap for its own fit.
        with threadpool_limits(limits=os.cpu_count(), user_api="blas"):
            r = minimize(f, np.zeros(Xb.shape[1] * L), jac=True, method="L-BFGS-B",
                         options={"maxiter": self.max_iter, "gtol": self.tol})
        self.W_, self.n_iter_ = r.x.reshape(-1, L).astype(np.float32), r.nit
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
        cfg = {"C": 1.0, "class_weight": "balanced", "tol": 1e-4, "max_iter": 500, **(config or {})}
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

    def is_linear_model(self):
        return True

    def supports_partial_fit(self) -> bool:
        return False
