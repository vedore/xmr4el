"""CTD vocabulary -> PubTator pseudo-documents, read back by the standard loader."""
import runpy

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from xmr4el.data.readers import Preprocessor

convert = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/dict_to_pubtator.py"))["convert"]


def test_ctd_to_pubtator():
    rows = [
        "# DiseaseName\tDiseaseID\tAltDiseaseIDs\tDefinition\tParentIDs\tTreeNumbers\tParentTreeNumbers\tSynonyms\tSlimMappings\n",
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


def test_ctd_chemicals_synonym_columns_and_omim():
    """Chemicals: synonyms are MESHSynonyms + CTDCuratedSynonyms, never Definition (column 8); -omim keeps OMIM ids."""
    header = ("# ChemicalName\tChemicalID\tCasRN\tPubChemCID\tPubChemSID\tDTXSID\tInChIKey\tDefinition\tParentIDs\t"
              "TreeNumbers\tParentTreeNumbers\tMESHSynonyms\tCTDCuratedSynonyms\n")
    row = "Aspirin\tMESH:D001241\t\t\t\t\t\tA salicylate drug.\t\t\t\tAcetylsalicylic Acid|aspirin\tASA\n"
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "dict.txt"
        path.write_text("".join(convert(["# Fields:\n", header, "#\n", row], "Chemical")), encoding="utf-8")
        d = Preprocessor.load_pubtator_file(str(path))
    assert [c.split(" [SEP]")[0] for c in d["corpus"]] == ["Aspirin", "Acetylsalicylic Acid", "ASA"]
    disease = ["# DiseaseName\tDiseaseID\tSynonyms\n", "Some syndrome\tOMIM:123456\tAlias\n"]
    assert list(convert(disease, "Disease")) == []
    assert [b.split("\t")[-1].strip() for b in convert(disease, "Disease", omim=True)] == ["OMIM:123456"] * 2
    with pytest.raises(ValueError, match="header"):
        list(convert([row], "Chemical"))
