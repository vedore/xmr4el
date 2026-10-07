"""CTD vocabulary -> PubTator pseudo-documents, read back by the standard loader."""
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from xmr4el.data.readers import Preprocessor

convert = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/dict_to_pubtator.py"))["convert"]


def test_ctd_to_pubtator():
    rows = [
        "# DiseaseName\tDiseaseID\n",
        "Fever\tMESH:D005334\t\t\t\t\t\tPyrexia|fever|Hyperthermia\t\n",
        "Some syndrome\tOMIM:123456\t\t\t\t\t\tAlias\t\n",
        "Rash\tMESH:D005076\n",
    ]
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "dict.txt"
        path.write_text("".join(convert(rows, "Disease")), encoding="utf-8")
        d = Preprocessor.load_pubtator_file(str(path))
    assert d["corpus"] == ["Fever [SEP] Fever", "Pyrexia [SEP] Pyrexia", "Hyperthermia [SEP] Hyperthermia", "Rash [SEP] Rash"]
    assert d["labels"] == ["D005334"] * 3 + ["D005076"]
