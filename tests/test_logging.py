"""Logging levels survive initialization and respect application-owned handlers."""
import logging
from pathlib import Path
import runpy
from unittest.mock import Mock, patch

from xmr4el import get_logger, set_verbosity
from xmr4el.data.readers import Preprocessor
from xmr4el.xmodel import XModel


def test_logging_levels(caplog, capsys):
    package = get_logger()
    original_level = package.level
    original_handlers = package.handlers[:]
    try:
        with caplog.at_level(logging.DEBUG):
            for verbosity, expected in ((0, logging.WARNING), (1, logging.INFO), (2, logging.DEBUG)):
                caplog.clear()
                set_verbosity(verbosity)
                XModel()
                assert package.level == expected
                assert package.handlers == original_handlers
                logger = get_logger("test")
                logger.debug("diagnostic")
                logger.info("progress")
                logger.warning("fallback")
                messages = [record.getMessage() for record in caplog.records]
                assert messages.count("fallback") == 1
                assert messages.count("progress") == (verbosity >= 1)
                assert messages.count("diagnostic") == (verbosity >= 2)
        assert capsys.readouterr().out == ""
    finally:
        package.setLevel(original_level)


def test_label_truncation_warning(tmp_path, caplog):
    texts, labels = tmp_path / "texts.tsv", tmp_path / "labels.txt"
    texts.write_text("0\tmention\n")
    labels.write_text("L1\nL2\nL3\n")
    with caplog.at_level(logging.WARNING, logger="xmr4el"):
        result = Preprocessor.load_data_labels_from_file(texts, labels)
    assert result["labels"] == ["L1"]
    assert "Truncating labels: supplied=3 text_groups=1" in caplog.text


def test_train_log_levels(caplog, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/train.py"

    def load_config(path):
        # CLI flags must override the verbosity stored in a model config.
        set_verbosity(2)
        return Mock(emb_flag=1)

    with caplog.at_level(logging.DEBUG, logger="xmr4el"):
        for flag, expected in (([], logging.INFO), (["-quiet"], logging.WARNING), (["-verbose"], logging.DEBUG)):
            caplog.clear()
            with patch("sys.argv", [str(script), "-train_path", "synthetic.tsv",
                                   "-labels_path", "labels.txt", *flag]), \
                 patch.object(XModel, "load_config", side_effect=load_config), \
                 patch.object(Preprocessor, "load_data_labels_from_file",
                              return_value={"corpus": [["mention"]], "labels": ["L1"]}):
                runpy.run_path(str(script), run_name="__main__")
            assert get_logger().level == expected
            assert ("Run completed" in caplog.text) == (expected <= logging.INFO)
    assert capsys.readouterr().out == ""
