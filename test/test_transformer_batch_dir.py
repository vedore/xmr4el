from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from xmr4el.models.featurization_wrapper import transformers


def test_transformer_batch_paths():
    transformer = transformers.Transformer
    create_batch_dir = transformer._create_batch_dir
    expected_embeddings = np.array([[1, 2], [3, 4]], dtype=np.float32)

    with TemporaryDirectory() as tmp, chdir(tmp), patch.object(
        transformers, "SentenceTransformer"
    ) as model_factory, patch.object(transformers.torch.cuda, "is_available", return_value=False):
        model_factory.return_value.to.return_value.encode.return_value = expected_embeddings
        for kwargs in ({}, {"batch_dir": "nested/batches"}, {"batch_dir": str(Path(tmp) / "absolute")}):
            expected_dir = Path(kwargs.get("batch_dir", "batch_dir")).resolve()

            def checked_create(batch_dir):
                # Assert before writing so a regression cannot touch the filesystem root.
                assert Path(batch_dir).resolve() == expected_dir
                create_batch_dir(batch_dir)

            with patch.object(transformer, "_create_batch_dir", side_effect=checked_create):
                embeddings = transformer._predict("mock-model", ["first", "second"], **kwargs)

            np.testing.assert_array_equal(embeddings, expected_embeddings)
            assert not expected_dir.exists()


if __name__ == "__main__":
    test_transformer_batch_paths()
