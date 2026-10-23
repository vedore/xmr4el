"""The real encoder (HF tokenizer + model, CLS / mean pooling) on a tiny BERT built offline: pooling, truncation, row
order across batches and after an OOM, and the embedding cache."""
from unittest.mock import patch
import logging
import numpy as np
import pytest
import torch
from torch.cuda import OutOfMemoryError
from transformers import BertConfig, BertModel, BertTokenizerFast

from xmr4el.features import transformers
from xmr4el.features.transformers import Transformer, sentence_model

WORDS = ["aspirin", "fever", "acute", "renal", "failure", "cancer", "of", "the", "liver", "dose"]
TEXTS = [" ".join(WORDS[(3 * i) % 10:(3 * i) % 10 + 1 + i % 4]) for i in range(23)]  # 1-4 words: padding varies


@pytest.fixture
def tiny_bert(tmp_path, monkeypatch):
    """Local BERT directory (hidden 8, 32 positions) with a word-level vocabulary; encoding on CPU."""
    path = tmp_path / "tiny_bert"
    path.mkdir()
    (path / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=15, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=16, max_position_embeddings=32)).save_pretrained(path)
    monkeypatch.setattr(transformers, "torch_device", lambda: torch.device("cpu"))
    sentence_model.cache_clear()
    yield str(path)
    sentence_model.cache_clear()


def one_by_one(path, texts, pooling, max_len=32):
    """Each text alone (no padding) through the HF model: the reference for padded batches."""
    tok, model = BertTokenizerFast.from_pretrained(path), BertModel.from_pretrained(path).eval()
    rows = []
    with torch.no_grad():
        for t in texts:
            H = model(**tok([t], truncation=True, max_length=max_len, return_tensors="pt")).last_hidden_state[0]
            rows.append((H[0] if pooling == "cls" else H.mean(0)).numpy())
    return np.array(rows)


def test_pooling_and_truncation(tiny_bert):
    for pooling in ("cls", "mean"):
        out = sentence_model(tiny_bert, "cpu", pooling)(TEXTS)
        assert out.dtype == np.float32 and out.shape == (len(TEXTS), 8)
        assert np.allclose(out, one_by_one(tiny_bert, TEXTS, pooling), atol=1e-5), pooling  # padding is masked
    long = [" ".join(WORDS * 2)]  # 20 words + [CLS] [SEP]
    assert np.allclose(sentence_model(tiny_bert, "cpu", "mean", 5)(long), one_by_one(tiny_bert, long, "mean", 5),
                       atol=1e-5)
    assert not np.allclose(sentence_model(tiny_bert, "cpu", "mean")(long), one_by_one(tiny_bert, long, "mean", 5))
    with pytest.raises(ValueError, match="pooling"):
        sentence_model(tiny_bert, "cpu", "max")


def test_embedding_row_order(tiny_bert):
    """Embeddings keep corpus row order across 12 batches and after an OOM; a batch size reduced by an OOM is kept
    (retrying the size that failed would burn the retry budget)."""
    ref = one_by_one(tiny_bert, TEXTS, "mean")
    real = sentence_model(tiny_bert, torch.device("cpu"))
    for oom_at in (None, TEXTS[4]):  # TEXTS[4]: the third batch OOMs once, then the batch size shrinks
        sizes = []

        def encode(batch):
            nonlocal oom_at
            sizes.append(len(batch))
            if batch[0] == oom_at and len(batch) > 1:
                oom_at = None
                raise OutOfMemoryError("test")
            return real(batch)

        with patch.object(transformers, "sentence_model", return_value=encode):
            emb = Transformer._predict(tiny_bert, TEXTS, batch_size=2)
        assert np.allclose(emb, ref, atol=1e-5) and oom_at is None
    assert sizes[:4] == [2, 2, 2, 1] and set(sizes[4:]) == {1}

    sizes = []

    def big(batch):
        sizes.append(len(batch))
        if len(batch) > 2:
            raise OutOfMemoryError("test")
        return real(batch)

    with patch.object(transformers, "sentence_model", return_value=big):
        emb = Transformer._predict(tiny_bert, TEXTS, batch_size=8, max_oom_retries=3)
    assert np.allclose(emb, ref, atol=1e-5) and sizes[:3] == [8, 4, 2] and set(sizes[3:]) <= {1, 2}


def test_mps_oom(tiny_bert, caplog, capsys):
    caplog.set_level(logging.INFO, logger="xmr4el.features.transformers")
    prefix = "xmr4el.features.transformers"
    oom = RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)")
    real, texts, sizes, fail = sentence_model(tiny_bert, torch.device("cpu")), TEXTS[:4], [], [oom]

    def encode(batch):
        sizes.append(len(batch))
        if len(batch) > 1 or fail[0] is not oom:
            raise fail[0]
        return real(batch)

    with patch(f"{prefix}.sentence_model", return_value=encode), \
         patch(f"{prefix}.torch_device", return_value=torch.device("mps")), \
         patch(f"{prefix}.torch.mps.empty_cache") as clear, patch(f"{prefix}.empty_cache") as cuda_clear:
        result = Transformer._predict(tiny_bert, texts, batch_size=4)
        assert np.allclose(result, one_by_one(tiny_bert, texts, "mean"), atol=1e-5) and sizes == [4, 2, 1, 1, 1, 1]
        assert clear.call_count == 2 and not cuda_clear.called
        for error, retries in ((oom, 0), (RuntimeError("unrelated failure"), 3)):
            fail[0] = error
            try:
                Transformer._predict(tiny_bert, texts, batch_size=4, max_oom_retries=retries)
            except RuntimeError as e:
                assert e is error if retries else e.__cause__ is oom
            else:
                raise AssertionError("failed batch did not raise")
        assert clear.call_count == 3, "unrelated RuntimeError must not retry or clear cache"
    assert capsys.readouterr().out == ""
    assert any(record.levelno == logging.WARNING and "retry_batch_size=2" in record.getMessage()
               for record in caplog.records)
    assert not any("Transformer batch" in record.getMessage() for record in caplog.records)


def test_transform_cache(tiny_bert):
    """transform embeds each distinct text once, in input order; a second transform of the same distinct texts
    loads the saved embeddings instead of re-encoding; another max_seq_length is another cache entry."""
    calls = []
    real = Transformer._predict.__func__

    def counted(cls, name, texts, **kw):
        calls.append(list(texts))
        return real(cls, name, texts, **kw)

    cfg = {"type": "local", "kwargs": {"model_name": tiny_bert, "pooling": "cls"}}
    with patch.object(Transformer, "_predict", classmethod(counted)):
        _, first = Transformer.transform(["fever", "aspirin", "fever"], cfg)
        _, again = Transformer.transform(["aspirin", "fever"], cfg)  # same distinct texts
        Transformer.transform(["aspirin", "dose"], cfg)
        Transformer.transform(["aspirin", "fever"], {**cfg, "kwargs": {**cfg["kwargs"], "max_seq_length": 10}})
    assert calls == [["aspirin", "fever"], ["aspirin", "dose"], ["aspirin", "fever"]], calls
    ref = one_by_one(tiny_bert, ["aspirin", "fever"], "cls")
    assert np.allclose(first, ref[[1, 0, 1]], atol=1e-5) and np.array_equal(again, first[[1, 0]])


def test_transform_progress_events(tiny_bert, progress_events, monkeypatch):
    """events=True: ticks count distinct texts and end at the total (a batch per tick here), a cache hit emits
    cache_hit; events=False (prediction) emits nothing."""
    monkeypatch.setattr(transformers, "TICK_SECONDS", 0)
    cfg = {"type": "local", "kwargs": {"model_name": tiny_bert, "pooling": "cls", "batch_size": 2}}
    Transformer.transform(["fever", "aspirin", "fever", "dose"], cfg)
    assert progress_events == []
    Transformer.transform(["acute", "renal", "acute", "liver", "dose"], cfg, events=True)
    assert [(e["phase"], e["done"], e["total"]) for e in progress_events] == [("tick", 2, 4), ("tick", 4, 4)]
    progress_events.clear()
    Transformer.transform(["renal", "acute", "dose", "liver"], cfg, events=True)
    assert progress_events == [{"stage": "encode", "part": "transformer", "phase": "cache_hit", "distinct": 4}]
