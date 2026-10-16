"""Installed-wheel smoke check (CI, not collected by pytest): run from outside the repo with a read-only install.
The package's default transformer cache sits next to the install; a cache miss must still return embeddings."""
from unittest.mock import patch

import numpy as np

import xmr4el
from xmr4el.features.transformers import CACHE_DIR, Transformer

assert "site-packages" in xmr4el.__file__, xmr4el.__file__
emb = np.arange(4, dtype=np.float32).reshape(2, 2)
with patch.object(Transformer, "_predict", return_value=emb):
    _, out = Transformer.transform(["b", "a", "b"], {"type": "sapbert", "kwargs": {}})
assert out.tolist() == [[2, 3], [0, 1], [2, 3]], out
print("installed smoke OK; cache dir", CACHE_DIR)
