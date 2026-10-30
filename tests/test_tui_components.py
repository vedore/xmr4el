"""The Model Builder registry (xmr4el_tui/components.py, stdlib, never imports the backend) against the backend: same
defaults, same accept / reject decisions, and every configuration the builder can produce loads with
XModel.load_config and fits the TF-IDF / SVD settings it offers. No training, no checkpoint download."""
import copy
import inspect
import json

import numpy as np
import pytest

from xmr4el.features.reduction import DimensionModel
from xmr4el.features.transformers import MODEL_NAMES
from xmr4el.features.vectorizers import Vectorizer
from xmr4el.hierarchy.clusterers import BalancedKMeans
from xmr4el.learning.classifiers import ClassifierModel, JointOvRLogistic
from xmr4el.xmodel import PREDICT_DEFAULTS, XModel
from xmr4el_tui import components
from xmr4el_tui.components import KMEANS_FIELDS, LOGREG_FIELDS, SEARCH_FIELDS, SVD_FIELDS, TFIDF_FIELDS

BASE, _, _ = components.template(".")
CORPUS = ["heart attack", "attack of the heart", "myocardial infarction", "kidney failure", "renal failure"]


def defaults(fields):
    return {f.path[-1]: f.default for f in fields}


def test_defaults_mirror_the_backend():
    tfidf = Vectorizer.fit(CORPUS).config["kwargs"]
    assert {k: list(v) if isinstance(v, tuple) else v for k, v in tfidf.items() if k != "dtype"} == defaults(
        TFIDF_FIELDS)
    assert DimensionModel.fit(np.eye(4, dtype=np.float32)).config["kwargs"] == defaults(SVD_FIELDS)
    assert ClassifierModel.init_model({"type": "jointlogisticregression"}).config["kwargs"] == defaults(LOGREG_FIELDS)
    assert BalancedKMeans.DEFAULTS == defaults(KMEANS_FIELDS) and PREDICT_DEFAULTS == defaults(SEARCH_FIELDS)
    sig = inspect.signature(XModel).parameters
    for s in components.SLOTS:
        for f in s.extras:
            assert sig[f.path[0]].default == f.default, f.path
    assert MODEL_NAMES == components.MODEL_NAMES and BASE["features"] == components.HYBRID


def test_closed_components_take_exactly_their_fields():
    """`closed` kwargs are the backend's whole accepted set: the matcher's constructor arguments and the clustering
    defaults. A key outside it blocks readiness; every key inside constructs the backend."""
    sig = [n for n in inspect.signature(JointOvRLogistic).parameters]
    closed = {c.type: [f.path[-1] for f in c.fields] for s in components.SLOTS for c in s.components if c.closed}
    assert closed == {"jointlogisticregression": sig, "balancedkmeans": list(BalancedKMeans.DEFAULTS)}
    JointOvRLogistic(**defaults(LOGREG_FIELDS))
    for sid, key in (("matcher", "max_iterr"), ("hierarchy", "n_cluster")):
        cfg = copy.deepcopy(BASE)
        cfg[components.slot(sid).key]["kwargs"][key] = 10
        found = components.problems(cfg)[sid]
        assert [(f, key in m) for f, m in found] == [(None, True)], found
        cfg[components.slot(sid).key]["kwargs"].pop(key)
        assert not components.problems(cfg)
    with pytest.raises(TypeError):
        ClassifierModel.init_model({"type": "jointlogisticregression", "kwargs": {"max_iterr": 10}})


@pytest.mark.parametrize("sid,name,bad", [
    ("hierarchy", "n_clusters", 0), ("hierarchy", "iter_limit", 0), ("hierarchy", "refine_iter", -1),
    ("hierarchy", "seed", -1), ("hierarchy", "n_clusters", 2.0), ("hierarchy", "depth", 0),
    ("hierarchy", "min_leaf_size", -1), ("matcher", "C", 0.0), ("matcher", "tol", -1.0), ("matcher", "max_iter", 0),
    ("matcher", "class_weight", "auto"), ("search", "beam_size", 0), ("search", "topk", -1),
    ("search", "knn_beta", 100.5), ("search", "leaf_candidates", 0), ("search", "beam_size", True),
    ("transformer", "batch_size", -1)])
def test_values_the_builder_rejects_the_backend_rejects(sid, name, bad):
    slot = components.slot(sid)
    f = next(f for f in components.installed(BASE, slot).fields + slot.extras if f.path[-1] == name)
    good = copy.deepcopy(BASE)
    XModel(**good)
    assert components.check(f, components.value(good, slot, f)) is None
    cfg = copy.deepcopy(BASE)
    components.assign(cfg, {}, slot, f, bad)
    assert components.check(f, bad) is not None
    with pytest.raises(ValueError):
        XModel(**cfg)


def builder_configs():
    """Base config with every component and every enumeration value the builder offers."""
    out = []
    for s in components.SLOTS:
        for comp in s.components:
            cfg = copy.deepcopy(BASE)
            if components.installed(cfg, s) is not comp:
                cfg[s.key] = copy.deepcopy(comp.base)
                if comp.id == "custom":
                    cfg[s.key]["kwargs"]["model_name"] = "org/model"
            out.append(cfg)
            for f in comp.fields + s.extras:
                for _, v in f.choices:
                    c = copy.deepcopy(cfg)
                    components.assign(c, {}, s, f, v)
                    out.append(c)
    return out


def test_every_builder_config_round_trips_through_load_config(tmp_path):
    for i, cfg in enumerate(builder_configs()):
        assert not components.problems(cfg), cfg
        path = tmp_path / f"{i}.json"
        path.write_text(components.dumps(cfg))
        model = XModel.load_config(path)
        for key, value in cfg.items():
            if key == "verbose":  # logging level, not kept on the model
                continue
            got = getattr(model, key)
            assert got == ({**PREDICT_DEFAULTS, **value} if key == "predict_config" else value), key
        assert json.loads(path.read_text()) == cfg


def test_offered_tfidf_and_svd_settings_fit():
    for cfg in builder_configs():
        for key in ("vectorizer_config", "context_vectorizer_config"):
            vec = Vectorizer.fit(CORPUS, cfg[key])
            for key2 in ("dimension_config", "context_dimension_config"):
                svd = copy.deepcopy(cfg[key2])
                svd["kwargs"]["n_components"] = 2  # small corpus: keep the SVD (not skipped)
                DimensionModel.fit(vec.transform(CORPUS), svd)
