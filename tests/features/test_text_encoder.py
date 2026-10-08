"""features "tfidf" and "sapbert_char_context" (transformer | char TF-IDF on the mention | context TF-IDF): three equal
blocks, encode (train) and predict (query) agree, save/load round-trips. Transformer stubbed; runs offline."""
import tempfile

import numpy as np
from unittest.mock import patch

from xmr4el.features.encoder import TextEncoder
from xmr4el.data.readers import Preprocessor
from xmr4el.features.transformers import CLS_POOLED, MODEL_NAMES, Transformer


def fake_transformer(texts, _config):
    # Large raw norms, like the real sentence embeddings; per-block normalisation must neutralise them
    return np.array([[10.0 * len(t), 30.0, 1.0] for t in texts], dtype=np.float32)


def test_text_encoder():
    char = {"type": "tfidf", "kwargs": {"analyzer": "char_wb", "ngram_range": [2, 4],
                                        "sublinear_tf": True, "max_df": 1.0}}
    svd3 = {"type": "sklearntruncatedsvd", "kwargs": {"n_components": 3, "random_state": 0}}

    enc1 = TextEncoder(vectorizer_config=char, dimension_config=svd3, features="tfidf")
    plain = ["aspirin tablets", "aspirin", "tumour growth", "tumours"]
    X1 = enc1.encode(plain).toarray()
    with tempfile.TemporaryDirectory() as d:
        enc1.save(d)
        X1q = TextEncoder.load(d).predict(plain).toarray()
    assert X1.shape == (4, 3) and np.allclose((X1 ** 2).sum(axis=1), 1)
    assert np.allclose(X1, X1q, atol=1e-6), "tfidf: saved/loaded predict must reproduce encode"

    enc6 = TextEncoder(
        vectorizer_config=char, dimension_config=svd3, transformer_config={"type": "stub", "kwargs": {}},
        features="sapbert_char_context",
        context_vectorizer_config={"type": "tfidf", "kwargs": {"ngram_range": [1, 1], "max_df": 1.0}},
        context_dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 2, "random_state": 0}},
    )
    texts6 = ["aspirin [SEP] doc one", "aspirin [SEP] another doc", "tumour [SEP] doc one",
              "aspirine [SEP] fever dose", "tumours [SEP] "]  # last: empty window
    with patch.object(TextEncoder, "_encode_text_using_transformer", staticmethod(fake_transformer)):
        X6 = enc6.encode(texts6).toarray()
        with tempfile.TemporaryDirectory() as d:
            enc6.save(d)
            X6q = TextEncoder.load(d).predict(texts6).toarray()
    assert X6.shape == (5, 8), X6.shape
    assert np.allclose((X6[:4, :3] ** 2).sum(axis=1), 1 / 3), "three equal blocks"
    assert np.allclose(X6[0, :6], X6[1, :6]), "mention blocks see only the mention"
    assert not np.allclose(X6[0, 3:6], X6[2, 3:6]), "char block must depend on the mention"
    assert np.allclose(X6[4, 6:], 0) and np.isclose((X6[4, :3] ** 2).sum(), 1 / 2), "empty context = zero block"
    assert not np.allclose(X6[0, 6:], X6[1, 6:]), "context block must depend on the context"
    assert np.allclose(X6, X6q, atol=1e-6), "saved/loaded sapbert_char_context encoder must reproduce encode"
    enc = TextEncoder(features="flag6")
    with tempfile.TemporaryDirectory() as d:
        enc.save(d)
        for action in (lambda: enc.encode(texts6), lambda: enc.predict(texts6), lambda: TextEncoder.load(d)):
            try:
                action()
            except ValueError as e:
                assert "features must be one of" in str(e)
            else:
                raise AssertionError("unknown features accepted")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1|t|Aspirin works\n1|a|It lowers fever.\n1\t17\t23\tlowers\tT\tC2\n")
    assert Preprocessor.load_pubtator_file(f.name, window=2)["corpus"] == ["lowers [SEP] works It fever."]
    assert Preprocessor.load_pubtator_file(f.name)["corpus"] == ["lowers [SEP] Aspirin works It lowers fever."]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1|t|Surgical site infection (SSI)\n1|a|SSI fell.\n1\t31\t34\tSSI\tT\tC1\n"
                "2|t|SSI rose\n2|a|x\n2\t0\t3\tSSI\tT\tC1\n")  # doc 2 does not define SSI
    load = lambda **kw: [t.split(" [SEP]")[0] for t in Preprocessor.load_pubtator_file(f.name, **kw)["corpus"]]
    assert load() == ["SSI", "SSI"]
    assert load(abbrev="append", window=1) == ["SSI Surgical site infection", "SSI"]
    assert load(abbrev="replace") == ["Surgical site infection", "SSI"]

    assert MODEL_NAMES["sapbert"] in CLS_POOLED, "config type sapbert must load SapBERT with [CLS] pooling"
    assert MODEL_NAMES["sentencetbiobert"] not in CLS_POOLED
    calls = []
    def fake_predict(cls, name, texts, **kw):  # embedding = position of the text in the batch sent to the model
        calls.append((name, kw, list(texts)))
        return np.arange(len(texts), dtype=np.float32)[:, None]

    with patch.object(Transformer, "_predict", classmethod(fake_predict)):
        Transformer.transform(["x"], {"type": "sapbert", "kwargs": {"batch_size": 7}})
        Transformer.transform(["x"], {"type": "any", "kwargs": {"model_name": "org/model", "pooling": "mean"}})
        _, E = Transformer.transform(["b", "a", "b", "c", "a"], {"type": "sapbert", "kwargs": {}})
    sent = calls[2][2]
    assert sorted(sent) == ["a", "b", "c"], "each distinct text is embedded once"
    assert [sent[int(i)] for i in E.ravel()] == ["b", "a", "b", "c", "a"], "rows map back to input order"
    assert calls[0][0] == MODEL_NAMES["sapbert"] and calls[0][1]["batch_size"] == 7
    assert calls[1][0] == "org/model" and calls[1][1]["pooling"] == "mean", "any checkpoint via kwargs.model_name"
    print("ok")

