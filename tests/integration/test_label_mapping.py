"""Label order through training and persistence."""
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import numpy as np
import pytest
from xmr4el.data.readers import Preprocessor
from xmr4el.xmodel import XModel


def test_label_mapping():
    """initial_labels[j] must name column j of Y and row j of Z, through train, save and load."""
    groups = {"L3": ["c"], "L1": ["a1", "a2"], "L0": [], "L2": ["b"]}  # unsorted, one empty group
    owner = {t: label for label, texts in groups.items() for t in texts}
    encoder = Mock()
    encoder.return_value.encode.side_effect = lambda texts: np.eye(len(texts))  # row i = text i
    encoder.return_value.transformer_config = None
    with patch("xmr4el.xmodel.TextEncoder", encoder), patch(
        "xmr4el.xmodel.HierarchicalMLModel"
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
        "xmr4el.xmodel.HierarchicalMLModel"
    ) as hml:
        root = Mock(local_to_global_idx=np.arange(3))
        hml.load.return_value.hmodel = [[root]]
        xm.save(tmp)
        (saved,) = Path(tmp).iterdir()
        assert XModel.load(saved).initial_labels == ["L1", "L2", "L3"]
        assert XModel.load(saved, research=True).Z.shape[0] == 3

        # More labels than root labels / Z rows (e.g. an empty group given a name) must be refused at load
        root.local_to_global_idx = np.arange(4)
        with pytest.raises(ValueError, match="3 labels vs 4 root labels"):
            XModel.load(saved)
        with open(saved / "xmodel.pkl", "rb") as f:
            state = pickle.load(f)
        state["initial_labels"] = ["L0", "L1", "L2", "L3"]
        with open(saved / "xmodel.pkl", "wb") as f:
            pickle.dump(state, f)
        with pytest.raises(ValueError, match="4 labels vs 3 Z rows"):
            XModel.load(saved, research=True)
    print("label mapping checks ok")
