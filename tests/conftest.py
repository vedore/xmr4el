import logging

import pytest

from xmr4el.features import transformers


@pytest.fixture(autouse=True)
def _transformer_cache(tmp_path, monkeypatch):
    """Tests use fake encoders: keep their embeddings out of outputs/cache."""
    monkeypatch.setattr(transformers, "CACHE_DIR", str(tmp_path / "transformer_cache"))


@pytest.fixture
def progress_events():
    """Events emitted on xmr4el.progress (which does not propagate) during the test, in order."""
    events = []
    handler = logging.Handler()
    handler.emit = lambda record: events.append(record.event)
    logger = logging.getLogger("xmr4el.progress")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield events
    logger.removeHandler(handler)
    logger.setLevel(level)
