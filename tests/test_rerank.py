"""Cross-encoder reranker: label texts follow the Y columns, a tiny BERT learns a toy task, save/load round trip."""
import numpy as np
import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from xmr4el.data.readers import Preprocessor
from xmr4el.features.label_embeddings import LabelEmbeddingFactory
from scipy.sparse import csr_matrix
from xmr4el.rerank import CrossEncoderReranker, label_texts, rerank_order, top_k

WORDS = ["alpha", "beta", "gamma", "delta", "ctx", "one", "two"]


def tiny_reranker():
    tok = BertTokenizer(vocab={w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ";", *WORDS])})
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64, num_labels=1,
                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return CrossEncoderReranker(BertForSequenceClassification(cfg), tok, {"tree": "t1", "k": 4}, max_length=32)


def test_label_texts_follow_y_columns():
    """Random columns j: label text j holds only label classes[j]'s strings, most frequent first, ties first-seen."""
    rng = np.random.default_rng(0)
    labels = [f"L{i:02d}" for i in rng.permutation(30)]  # unsorted groups: column order != input order
    groups = [[f"{lab} s{rng.integers(3)} [SEP] c" for _ in range(rng.integers(1, 6))] for lab in labels]
    groups[0] = [f"{labels[0]} b [SEP] x", f"{labels[0]} a [SEP] y", f"{labels[0]} A [SEP] z"]
    _, label_to_idx = Preprocessor.prepare_data(groups, labels)
    Y, classes = LabelEmbeddingFactory.label_binarizer(LabelEmbeddingFactory.generate_label_matrix(label_to_idx))
    names = label_texts(groups, Y)
    assert len(names) == len(classes)
    for j in rng.choice(len(classes), 10, replace=False):
        assert {s.split()[0] for s in names[j].split("; ")} == {classes[j].lower()}, (j, names[j])
    assert names[list(classes).index(labels[0])] == f"{labels[0].lower()} a; {labels[0].lower()} b"


def test_top_k_and_rerank_order():
    s = csr_matrix(np.array([[0.1, 0.5, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    idx, val = top_k(s, 2)
    assert idx[0].tolist() == [1, 2] and val[0].tolist() == [0.5, 0.3] and idx[1].size == 0
    tree, logits = np.array([0.5, 0.3, 0.2]), np.array([0.0, 1.0, 1.0])
    assert rerank_order(tree, logits, 0).tolist() == [0, 1, 2]
    assert rerank_order(tree, logits, float("inf")).tolist() == [1, 2, 0]  # logit tie -> tree score
    assert rerank_order(tree, logits, 1).tolist() == [1, 2, 0]


def test_fit_score_save_load(tmp_path):
    """Gold = the label whose text is the query's mention word (needs query-label interaction, ~800 steps);
    K = 4, negatives = the other words."""
    labels = WORDS[:4]
    queries = [f"{w} [SEP] ctx {c}" for w in labels for c in ("one", "two")] * 4
    gold = [labels.index(q.split()[0]) for q in queries]
    candidates = [[g, *[j for j in range(4) if j != g]] for g in gold]
    rr = tiny_reranker()
    rr.fit(queries, candidates, labels, epochs=100, rows_per_batch=4, lr=1e-3)
    scores = rr.score([q for q in queries for _ in labels], labels * len(queries)).reshape(len(queries), 4)
    assert (scores.argmax(1) == gold).all(), scores

    rr.save(str(tmp_path / "rr"))
    back = CrossEncoderReranker.load(str(tmp_path / "rr"), tree="t1")
    assert back.meta == {"tree": "t1", "k": 4} and back.max_length == 32
    np.testing.assert_allclose(back.score(queries[:3], labels[:3]), rr.score(queries[:3], labels[:3]), atol=1e-5)
    with pytest.raises(ValueError, match="trained on tree t1"):
        CrossEncoderReranker.load(str(tmp_path / "rr"), tree="t2")
