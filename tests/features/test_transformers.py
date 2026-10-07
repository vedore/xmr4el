"""Embedding row order and OOM recovery."""
from unittest.mock import Mock, patch
import logging
import numpy as np
from xmr4el.features.transformers import Transformer
from torch.cuda import OutOfMemoryError


def test_embedding_row_order():
    """Transformer embeddings must keep corpus row order past 10 on-disk batches and after an OOM."""

    class Fake:
        oom_at = None

        def to(self, device):
            return self

        def encode(self, batch, **kwargs):
            if batch[0] == self.oom_at and len(batch) > 1:
                self.oom_at = None
                raise OutOfMemoryError("fake")
            return [[float(t)] for t in batch]

    texts = [str(i) for i in range(23)]  # 12 batches of 2
    for oom_at in (None, "4"):  # "4": third batch OOMs once, then the batch size shrinks
        fake = Fake()
        fake.oom_at = oom_at
        with patch("xmr4el.features.transformers.SentenceTransformer", lambda name: fake):
            emb = Transformer._predict("fake", texts, batch_size=2)
        assert emb[:, 0].tolist() == list(range(23)), (oom_at, emb[:, 0].tolist())
        assert fake.oom_at is None, "OOM path was not exercised"

    # A batch size reduced by OOM is kept: retrying the size that failed would burn the retry budget
    class Big(Fake):
        def encode(self, batch, **kwargs):
            if len(batch) > 2:
                raise OutOfMemoryError("fake")
            return [[float(t)] for t in batch]

    with patch("xmr4el.features.transformers.SentenceTransformer", lambda name: Big()):
        emb = Transformer._predict("fake", texts, batch_size=8, max_oom_retries=3)
    assert emb[:, 0].tolist() == list(range(23))
    print("embedding row order ok")


def test_mps_oom(caplog, capsys):
    caplog.set_level(logging.INFO, logger="xmr4el.features.transformers")
    prefix = "xmr4el.features.transformers"
    oom = RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)")
    texts = ["0", "1", "2", "3"]
    batches = []

    def encode(batch, **kwargs):
        batches.append(len(batch))
        if len(batch) > 1:
            raise oom
        return [[float(t)] for t in batch]

    model = Mock(encode=Mock(side_effect=encode))
    with patch(f"{prefix}.sentence_model", return_value=model), \
         patch(f"{prefix}.torch.cuda.is_available", return_value=False), \
         patch(f"{prefix}.torch.backends.mps.is_available", return_value=True), \
         patch(f"{prefix}.torch.mps.empty_cache") as clear, patch(f"{prefix}.empty_cache") as cuda_clear:
        result = Transformer._predict("fake", texts, batch_size=4)
        assert result[:, 0].tolist() == list(range(4)) and batches == [4, 2, 1, 1, 1, 1]
        assert clear.call_count == 2 and not cuda_clear.called
        for error, retries in ((oom, 0), (RuntimeError("unrelated failure"), 3)):
            model.encode.side_effect = error
            try:
                Transformer._predict("fake", texts, batch_size=4, max_oom_retries=retries)
            except RuntimeError as e:
                assert e is error if retries else e.__cause__ is oom
            else:
                raise AssertionError("failed batch did not raise")
        assert clear.call_count == 3, "unrelated RuntimeError must not retry or clear cache"
    assert capsys.readouterr().out == ""
    assert any(record.levelno == logging.WARNING and "retry_batch_size=2" in record.getMessage()
               for record in caplog.records)
    assert not any("Transformer batch" in record.getMessage() for record in caplog.records)
    print("MPS OOM checks ok")
