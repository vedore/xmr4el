import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def test_split_cli():
    script = Path(__file__).resolve().parents[2] / "scripts/split_pubtator.py"
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus = root / "corpus.txt"
        corpus.write_text("1|t|Title\n1|a|Context\n1\t0\t5\talpha\tType\tL1\n\n", encoding="utf-8")
        subprocess.run([sys.executable, str(script), "--input", str(corpus), "--outdir", str(root / "out"),
                        "--train_ratio", "1", "--dev_ratio", "0", "--test_ratio", "0", "--emit_jsonl"],
                       check=True, cwd=root, capture_output=True)
        assert (root / "out/corpus_pubtator_train.txt").read_text() == corpus.read_text()
        assert (root / "out/corpus_pubtator_dev.txt").read_text() == ""
        assert (root / "out/corpus_pubtator_test.txt").read_text() == ""
        row = json.loads((root / "out/train.jsonl").read_text())
        assert row["cui"] == "L1" and row["context"] == "Title Context"
