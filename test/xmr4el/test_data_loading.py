"""Check both local input formats and the training CLI without training a model."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import pickle
import runpy
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from xmr4el.data import Preprocessor
from xmr4el.transformers import Transformer
from xmr4el.xmodel import XModel
from torch.cuda import OutOfMemoryError


def test_local_inputs():
    script = Path(__file__).with_name("test_train_pipeline.py")
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        (root / "datasets").mkdir()
        tsv = root / "data/train.tsv"
        labels = root / "data/labels.txt"
        pubtator = root / "datasets/corpus.txt"
        tsv.write_text("1\tbeta\n0\talpha\n0\talias\n", encoding="utf-8")
        labels.write_text("L1\nL2\n", encoding="utf-8")
        pubtator.write_text(
            "1|t|Title\n1|a|Context\n"
            "1\t0\t5\talpha\tType\tL1\n"
            "1\t6\t10\tbeta\tType\tL2\n\n"
            "2|t|Other title\n2|a|Other context\n"
            "2\t0\t5\talias\tType\tL1\n\n",
            encoding="utf-8",
        )
        tsv_expected = [["alpha", "alias"], ["beta"]]
        pub_expected = [
            ["alpha [SEP] Title Context", "alias [SEP] Other title Other context"],
            ["beta [SEP] Title Context"],
        ]
        assert Preprocessor.load_data_labels_from_file(tsv, labels) == {
            "corpus": tsv_expected, "labels": ["L1", "L2"]
        }
        grouped, ids = Preprocessor.organize_pubtator_output(
            Preprocessor.load_pubtator_file(pubtator)
        )
        assert grouped == pub_expected and ids == ["L1", "L2"]
        flat, mapping = Preprocessor.prepare_data(grouped, ids)
        assert flat == pub_expected[0] + pub_expected[1]
        assert mapping == {"L1": [0, 1], "L2": [2]}

        for path, extra, flag, expected in (
            (tsv, ["-labels_path", str(labels)], 1, tsv_expected),
            (pubtator, [], 6, pub_expected),
        ):
            model = Mock(emb_flag=flag, context_window=None, abbrev_expansion=None)
            argv = [str(script), "-train_path", str(path), "-ds_len", "1", *extra]
            with patch("sys.argv", argv), patch.object(
                XModel, "load_config", return_value=model
            ):
                runpy.run_path(str(script), run_name="__main__")
            model.train.assert_called_once_with(expected[:1], ["L1"])
            model.save.assert_called_once()

        model = Mock(emb_flag=6, context_window=None)
        argv = [str(script), "-train_path", str(tsv), "-labels_path", str(labels)]
        with patch("sys.argv", argv), patch.object(
            XModel, "load_config", return_value=model
        ), redirect_stderr(StringIO()):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("plain TSV must reject the split-input encoder")
        model.train.assert_not_called()
    print("local input checks ok")


def test_label_mapping():
    """initial_labels[j] must name column j of Y and row j of Z, through train, save and load."""
    groups = {"L3": ["c"], "L1": ["a1", "a2"], "L0": [], "L2": ["b"]}  # unsorted, one empty group
    owner = {t: label for label, texts in groups.items() for t in texts}
    encoder = Mock()
    encoder.return_value.encode.side_effect = lambda texts: np.eye(len(texts))  # row i = text i
    with patch("xmr4el.xmodel.TextEncoder", encoder), patch(
        "xmr4el.xmodel.HierarchicaMLModel"
    ):
        xm = XModel()
        xm.train(list(groups.values()), list(groups))
    flat, _ = Preprocessor.prepare_data(list(groups.values()), list(groups))
    Y, Z = xm.Y.tocsc(), np.asarray(xm.Z)
    assert xm.initial_labels == ["L1", "L2", "L3"], xm.initial_labels
    assert Y.shape[1] == Z.shape[0] == len(xm.initial_labels)
    for j, label in enumerate(xm.initial_labels):
        rows = Y[:, j].nonzero()[0]
        assert {owner[flat[i]] for i in rows} == {label}, (j, label)
        assert set(Z[j].nonzero()[0]) == set(rows), (j, label)

    xm.text_encoder = None  # the Mock cannot be pickled; TextEncoder.load is stubbed below
    with TemporaryDirectory() as tmp, patch("xmr4el.xmodel.TextEncoder"), patch(
        "xmr4el.xmodel.HierarchicaMLModel"
    ):
        xm.save(tmp)
        (saved,) = Path(tmp).iterdir()
        assert XModel.load(saved).initial_labels == ["L1", "L2", "L3"]

        # Legacy trees stored first-seen input order; loading must sort it
        with open(saved / "xmodel.pkl", "rb") as f:
            state = pickle.load(f)
        state["_original_labels"] = ["L3", "L1", "L2"]
        with open(saved / "xmodel.pkl", "wb") as f:
            pickle.dump(state, f)
        assert XModel.load(saved).initial_labels == ["L1", "L2", "L3"]

        # A legacy empty group has no Y column; sorting cannot repair it, so load must refuse
        state["_original_labels"] = ["L3", "L1", "L0", "L2"]
        with open(saved / "xmodel.pkl", "wb") as f:
            pickle.dump(state, f)
        try:
            XModel.load(saved)
        except AssertionError:
            pass
        else:
            raise AssertionError("load accepted more labels than Z rows")
    print("label mapping checks ok")


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
        with patch("xmr4el.transformers.SentenceTransformer", lambda name: fake):
            emb = Transformer._predict("fake", texts, batch_size=2)
        assert emb[:, 0].tolist() == list(range(23)), (oom_at, emb[:, 0].tolist())
        assert fake.oom_at is None, "OOM path was not exercised"

    # A batch size reduced by OOM is kept: retrying the size that failed would burn the retry budget
    class Big(Fake):
        def encode(self, batch, **kwargs):
            if len(batch) > 2:
                raise OutOfMemoryError("fake")
            return [[float(t)] for t in batch]

    with patch("xmr4el.transformers.SentenceTransformer", lambda name: Big()):
        emb = Transformer._predict("fake", texts, batch_size=8, max_oom_retries=3)
    assert emb[:, 0].tolist() == list(range(23))
    print("embedding row order ok")


def test_mps_oom():
    prefix = "xmr4el.transformers"
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
    print("MPS OOM checks ok")


if __name__ == "__main__":
    test_local_inputs()
    test_label_mapping()
    test_embedding_row_order()
    test_mps_oom()
