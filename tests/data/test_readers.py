"""Local file loading and CLI inputs."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from xmr4el.data.readers import Preprocessor
from xmr4el.xmodel import XModel


def test_local_inputs():
    script = Path(__file__).resolve().parents[2] / "scripts/train.py"
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


def test_bc5cdr_ids():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "cdr.txt"
        path.write_text(
            "1|t|Ocular and auditory toxicity\n1|a|x\n"
            "1\t0\t28\tOcular and auditory toxicity\tDisease\tD1|D2\tOcular toxicity|auditory toxicity\n"
            "1\t0\t6\tOcular\tDisease\t-1\n"
            "1\t0\t28\taudiovisual toxicity\tDisease\tD1|D2\n"
            "1\t0\t28\tA and B\tDisease\tD4|-1\tA|B\n"
            "1\t0\t6\tOcular\tDisease\tD3\t\n"
            "1\tCID\tD1\tD3\n",
            encoding="utf-8",
        )
        d = Preprocessor.load_pubtator_file(str(path))
        assert [t.split(" [SEP]")[0] for t in d["corpus"]] == ["Ocular toxicity", "auditory toxicity", "A", "Ocular"]
        assert d["labels"] == ["D1", "D2", "D4", "D3"] and d["spans"] == [(0, 28), (0, 28), (0, 28), (0, 6)]
