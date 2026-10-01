"""emb_flag 5: both blocks see only the mention, each block carries half the row norm, and
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

    from xmr4el.models.featurization_wrapper.transformers import CLS_POOLED, transformer_dict
    sap = transformer_dict["sapbert"]({})
    assert sap.model_name in CLS_POOLED, "config type sapbert must load SapBERT with [CLS] pooling"
    assert transformer_dict["sentencetbiobert"]({}).model_name not in CLS_POOLED
    print("ok")


if __name__ == "__main__":
    main()
