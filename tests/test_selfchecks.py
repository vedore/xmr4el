"""Module self-checks (also runnable as `python -m xmr4el.<module>`)."""
from xmr4el import classifiers, clusterers, ranker
from xmr4el import eval as xmr_eval


def test_matcher_selfcheck():
    classifiers._matcher_selfcheck()


def test_clustering_selfcheck():
    clusterers._clustering_selfcheck()


def test_ranker_selfcheck():
    ranker._selfcheck()


def test_eval_selfcheck():
    xmr_eval._selfcheck()
