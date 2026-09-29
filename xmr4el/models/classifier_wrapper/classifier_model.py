import importlib
import os
import json
import pickle
import pkgutil
import sys
import torch
import multiprocessing

import numpy as np

from abc import ABCMeta
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SVC
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier

classifier_dict = {}

# if torch.cuda.is_available():
#     from cuml.linear_model import LogisticRegression as CUMLLogisticRegression
    

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


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

    @classmethod
    def load_subclasses(cls, package_name):
        """Dynamically imports all modules in the package to register subclasses."""
        package = sys.modules[package_name]
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package_name}.{modname}")


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

    @staticmethod
    def load_config_from_args(args):
        """
        Parse config from argparse.Namespace (path or inline JSON).
        """
        if args.classifier_config_path is not None:
            with open(args.classifier_config_path, "r", encoding="utf-8") as fin:
                classifier_config_json = fin.read()
        else:
            classifier_config_json = args.classifier_config_json

        try:
            classifier_config = json.loads(classifier_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                f"Failed to load classifier config json from {classifier_config_json} ({jex})"
            )
        return classifier_config
    

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
        defaults = {
            "penalty": "l2",
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
            "n_jobs": -1,
            "l1_ratio": None,
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
# Sklearn Random Forest
# ---------------------------

class SklearnRandomForestClassifier(ClassifierModel):
    """Sklearn Random Forest"""

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
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None,
            "monotonic_cst": None,
        }
        cfg = {**defaults, **(config or {})}
        est = RandomForestClassifier(**cfg)
        # onevsrest is usually unnecessary for RF multiclass, but we keep the signature
        return cls(cfg, est)

    @classmethod
    def train(cls, X_train, y_train, config=None, dtype=np.float32, onevsrest=False):
        wrapper = cls.init_model(config or {}, onevsrest=onevsrest)
        wrapper.model.fit(X_train, y_train)
        return wrapper

    def predict(self, predict_input):
        return self.model.predict(predict_input)

    def is_linear_model(self):
        return False
    
    def supports_partial_fit(self) -> bool:
        return False


# ---------------------------
# Sklearn SVC
# ---------------------------

class SklearnSupportVectorClassification(ClassifierModel):
    """Sklearn SVC"""

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
            "C": 1.0,
            "kernel": 'rbf',
            "degree": 3,
            "gamma": 'scale',
            "coef0": 0.0,
            "shrinking": True,
            "probability": True,
            "tol": 0.001,
            "cache_size": 200,
            "class_weight": None,
            "verbose": False,
            "max_iter": -1,
            "decision_function_shape": 'ovr',
            "break_ties": False,
            "random_state": None
        }
        cfg = {**defaults, **(config or {})}
        est = SVC(**cfg)
        # SVC already does OvR internally for multiclass; external OneVsRest is usually redundant.
        return cls(cfg, est)

    @classmethod
    def train(cls, X_train, y_train, config=None, dtype=np.float32, onevsrest=False):
        wrapper = cls.init_model(config or {}, onevsrest=onevsrest)
        wrapper.model.fit(X_train, y_train)
        return wrapper

    def predict(self, predict_input):
        return self.model.predict(predict_input)

    def decision_function(self, X):
        # Available for SVC even with probability=True
        return self.model.decision_function(X)

    def is_linear_model(self):
        return False
    
    def supports_partial_fit(self) -> bool:
        return False


# ---------------------------
# cuML Logistic Regression (GPU)
# ---------------------------

class CumlLogisticRegression(ClassifierModel):
    """cuML Logistic Regression"""

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
            "penalty": "l2",
            "tol": 0.0001,
            "C": 1.0,
            "fit_intercept": True,
            "class_weight": None,
            "max_iter": 1000,
            "linesearch_max_iter": 50,
            "verbose": False,
            "l1_ratio": None,
            "solver": "qn",
            "handle": None,
            "output_type": None,
        }
        cfg = {**defaults, **(config or {})}
        est = CUMLLogisticRegression(**cfg)
        return cls(cfg, est)

    @classmethod
    def train(cls, X_train, y_train, config=None):
        wrapper = cls.init_model(config or {}, onevsrest=False)
        wrapper.model.fit(X_train, y_train)
        return wrapper

    def predict(self, predict_input):
        return self.model.predict(predict_input)
    
    def supports_partial_fit(self) -> bool:
        return False


# ---------------------------
# LightGBM Classifier
# ---------------------------

class LightGBMClassifier(ClassifierModel):
    """LightGBM Classifier"""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config, file_name=None):
        classifier_path = os.path.join(load_dir, "classifier_model.pkl")
        assert os.path.exists(classifier_path), f"Classifier path {classifier_path} does not exist"
        with open(classifier_path, "rb") as fin:
            model_data = pickle.load(fin)
        return cls(config, model_data)

    @classmethod
    def init_model(cls, config, onevsrest=False):
        defaults = {
            "boosting_type": "gbdt",      # 'gbdt', 'dart', 'rf'
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample_for_bin": 200000,
            "objective": "multiclass",    # e.g., None, 'binary', 'multiclass', 'regression'
            "class_weight": None,
            "min_split_gain": 0.0,
            "min_child_weight": 1e-3,
            "min_child_samples": 20,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": None,
            "n_jobs": -1,
            "importance_type": "split",
            "verbosity": -1,
        }
        cfg = {**defaults, **(config or {})}
        est = LGBMClassifier(**cfg)
        if onevsrest:
            est = OneVsRestClassifier(est, n_jobs=cfg.get("n_jobs", -1))
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

    def classes(self):
        return self.model.classes_

    def is_linear_model(self):
        return False
    
    def supports_partial_fit(self) -> bool:
        return False
