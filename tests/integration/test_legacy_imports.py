"""Pickle GLOBAL references from the previous flat package still resolve."""
import pickle

from xmr4el.features.encoder import TextEncoder
from xmr4el.features.reduction import DimensionModel, SklearnTruncatedSVD
from xmr4el.features.vectorizers import Vectorizer, Tfidf
from xmr4el.hierarchy.node import MLModel
from xmr4el.hierarchy.tree import HierarchicalMLModel
from xmr4el.hierarchy.clusterers import Clustering, BalancedKMeans
from xmr4el.learning.classifiers import ClassifierModel, JointOvRLogistic
from xmr4el.learning.matcher import Matcher
from xmr4el.learning.ranker import Ranker


def test_legacy_imports():
    for module, classes in (
        ("encoder", (TextEncoder,)),
        ("vectorizers", (Vectorizer, Tfidf, DimensionModel, SklearnTruncatedSVD)),
        ("node", (MLModel,)), ("tree", (HierarchicalMLModel,)),
        ("clusterers", (Clustering, BalancedKMeans)),
        ("classifiers", (ClassifierModel, JointOvRLogistic, Matcher)),
        ("ranker", (Ranker,)),
    ):
        for cls in classes:
            assert pickle.loads(f"cxmr4el.{module}\n{cls.__name__}\n.".encode()) is cls
    # The legacy store is unpickled but no longer creates directories or owns training data.
    store = pickle.loads(b"cxmr4el.temp_store\nTempVarStore\n.")()
    assert vars(store) == {}
