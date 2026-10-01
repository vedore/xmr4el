"""emb_flag 5/6: both mention blocks see only the mention, each block carries half the row norm, and
encode (train) and predict (query) agree. Transformer stubbed; runs offline."""
import numpy as np
from unittest.mock import patch

from xmr4el.featurization.text_encoder import TextEncoder


def fake_transformer(texts, _config):
    # Large raw norms, like the real sentence embeddings (flag 4: 0.998 of the row norm)
    return np.array([[10.0 * len(t), 30.0, 1.0] for t in texts], dtype=np.float32)


def main():
    texts = ["aspirin [SEP] doc one", "aspirin [SEP] another doc", "tumour [SEP] doc one",
             "aspirine [SEP] x", "tumours [SEP] y"]
    enc = TextEncoder(
        vectorizer_config={"type": "tfidf", "kwargs": {"analyzer": "char_wb", "ngram_range": [2, 4],
                                                       "sublinear_tf": True, "max_df": 1.0}},
        dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 3, "random_state": 0}},
        transformer_config={"type": "stub", "kwargs": {}},
        flag=5,
    )
    with patch.object(TextEncoder, "_encode_text_using_transformer", staticmethod(fake_transformer)), \
         patch.object(TextEncoder, "_predict_text_using_transformer",
                      staticmethod(lambda X_test, transformer_config: fake_transformer(X_test, None))):
        X = enc.encode(texts).toarray()
        Xq = enc.predict(texts).toarray()

    assert X.shape == (5, 6), X.shape
    assert np.allclose((X[:, :3] ** 2).sum(axis=1), 0.5), "each block must hold half the row norm"
    assert np.allclose(X[0], X[1]), "same mention, different context -> identical row"
    assert not np.allclose(X[0, 3:], X[2, 3:]), "char block must depend on the mention"
    assert np.allclose(X, Xq, atol=1e-6), "predict must reproduce encode"

    # flag 6: third block from the context (text after [SEP]); three equal blocks; survives save/load
    import tempfile
    enc6 = TextEncoder(
        vectorizer_config=enc.vectorizer_config, dimension_config=enc.dimension_config,
        transformer_config=enc.transformer_config, flag=6,
        context_vectorizer_config={"type": "tfidf", "kwargs": {"ngram_range": [1, 1], "max_df": 1.0}},
        context_dimension_config={"type": "sklearntruncatedsvd", "kwargs": {"n_components": 2, "random_state": 0}},
    )
    with patch.object(TextEncoder, "_encode_text_using_transformer", staticmethod(fake_transformer)), \
         patch.object(TextEncoder, "_predict_text_using_transformer",
                      staticmethod(lambda X_test, transformer_config: fake_transformer(X_test, None))):
        texts6 = texts[:3] + ["aspirine [SEP] fever dose", "tumours [SEP] "]  # last: empty window
        X6 = enc6.encode(texts6).toarray()
        with tempfile.TemporaryDirectory() as d:
            enc6.save(d)
            X6q = TextEncoder.load(d).predict(texts6).toarray()
    assert X6.shape == (5, 8), X6.shape
    assert np.allclose((X6[:4, :3] ** 2).sum(axis=1), 1 / 3), "three equal blocks"
    assert np.allclose(X6[4, 6:], 0) and np.isclose((X6[4, :3] ** 2).sum(), 1 / 2), "empty context = zero block"
    assert not np.allclose(X6[0, 6:], X6[1, 6:]), "context block must depend on the context"
    assert np.allclose(X6, X6q, atol=1e-6), "saved/loaded flag-6 encoder must reproduce encode"

    from xmr4el.featurization.preprocessor import Preprocessor
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1|t|Aspirin works\n1|a|It lowers fever.\n1\t17\t23\tlowers\tT\tC2\n")
    assert Preprocessor.load_pubtator_file(f.name, window=2)["corpus"] == ["lowers [SEP] works It fever."]
    assert Preprocessor.load_pubtator_file(f.name)["corpus"] == ["lowers [SEP] Aspirin works It lowers fever."]

    from xmr4el.models.featurization_wrapper.transformers import CLS_POOLED, transformer_dict
    sap = transformer_dict["sapbert"]({})
    assert sap.model_name in CLS_POOLED, "config type sapbert must load SapBERT with [CLS] pooling"
    assert transformer_dict["sentencetbiobert"]({}).model_name not in CLS_POOLED
    print("ok")


if __name__ == "__main__":
    main()
