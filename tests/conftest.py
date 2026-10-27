import logging

import pytest


@pytest.fixture(autouse=True)
def _transformer_cache(request, tmp_path, monkeypatch):
    """Tests use fake encoders: keep their embeddings out of outputs/cache. TUI tests never import the backend
    (the string target is imported only here, so collecting tests/tui loads no torch)."""
    if "tui" not in request.path.parts:
        monkeypatch.setattr("xmr4el.features.transformers.CACHE_DIR", str(tmp_path / "transformer_cache"))


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
