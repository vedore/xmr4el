import pytest

from xmr4el.features import transformers


@pytest.fixture(autouse=True)
def _transformer_cache(tmp_path, monkeypatch):
    """Tests use fake encoders: keep their embeddings out of outputs/cache."""
    monkeypatch.setattr(transformers, "CACHE_DIR", str(tmp_path / "transformer_cache"))
