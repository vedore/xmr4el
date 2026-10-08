"""Offline pipeline round-trip and independent training-text ownership."""
from copy import deepcopy
import logging
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from xmr4el.learning.scoring import label_max_cos
from xmr4el.xmodel import XModel


def test_pipeline_persistence(caplog, capsys):
    caplog.set_level(logging.INFO, logger="xmr4el")
    config = dict(
        vectorizer_config={"type": "tfidf", "kwargs": {"analyzer": "char", "ngram_range": [2, 4]}},
        dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 6, "random_state": 42}},
        clustering_config={"type": "balancedkmeans", "kwargs": {"n_clusters": 2}},
        matcher_config={"type": "jointlogisticregression", "kwargs": {"max_iter": 100}},
        features="tfidf", depth=2, min_leaf_size=2,
    )
    texts = [[f"concept{j} synonym{i} group{j // 4}" for i in range(8)] for j in range(8)]
    labels = [f"L{j}" for j in range(8)][::-1]
    first, second = XModel(**deepcopy(config)), XModel(**deepcopy(config))
    first.train(texts, labels)
    second.train(texts, labels)
    texts[0].append("caller mutation")
    assert first.training_set == second.training_set and first.training_set != texts
    queries = [group[0] for group in texts]
    expected = first.predict(queries, beam_size=2).toarray()
    # knn re-scoring keeps the tree's candidates and multiplies each by exp(beta * max cosine to the label's rows)
    rescored = first.predict(queries, beam_size=2, knn_beta=2).toarray()
    knn = label_max_cos(first.text_encoder.predict(queries), first.X, first.Y.tocsr().indices, len(labels))
    assert ((rescored > 0) == (expected > 0)).all() and np.allclose(rescored, expected * np.exp(2 * knn))
    # predict arguments left None come from predict_config (stored with the tree)
    assert XModel(predict_config={"knn_beta": 2}).resolve_predict_config(beam_size=3, topk=None) == \
        {"beam_size": 3, "topk": 0, "knn_beta": 2}
    with TemporaryDirectory() as tmp:
        first.save(tmp)
        saved = next(Path(tmp).iterdir())
        with open(saved / "xmodel.pkl", "rb") as f:
            state = pickle.load(f)
        assert "_text_encoder" not in state and "_hml" not in state and "temp_var" not in state
        with open(saved / "text_encoder/text_encoder.pkl", "rb") as f:
            state = pickle.load(f)
        assert "_vectorizer_model" not in state and "_dimension_model" not in state
        restored = XModel.load(saved)
        assert restored.initial_labels == sorted(labels)
        assert restored.predict_config == first.predict_config == {"beam_size": 10, "topk": 0, "knn_beta": 0.0}
        assert restored.training_set == first.training_set
        assert np.allclose(restored.predict(queries, beam_size=2).toarray(), expected)
    assert capsys.readouterr().out == "", "library code must use logging, not stdout"
    messages = [record.getMessage() for record in caplog.records]
    assert any("Node started: layer=2 node=2/2" in message for message in messages)
    for stage in ("Encoding completed", "Label embeddings completed", "Hierarchy completed",
                  "Model saved: path=", "Routing completed", "Ranking completed"):
        assert any(stage in message for message in messages), stage
    assert not any("not being loaded" in message or "Matcher started" in message for message in messages)
