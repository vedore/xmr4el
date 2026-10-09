"""scripts/to_pubtator.py: type filter + id normalisation (NCBI / BioRED forms) and BioC -> PubTator offsets, read back
by the standard loader."""
from collections import Counter
from pathlib import Path
import runpy

from xmr4el.data.readers import Preprocessor

mod = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/to_pubtator.py"))


def test_norm_id():
    norm = mod["norm_id"]
    assert norm(" D012878") == "D012878" and norm("MESH:D1") == "D1" and norm("OMIM:123") == "OMIM:123"
    assert norm("D1,D2") == norm("D1+D2") == norm("MESH:D1, MESH:D2") == norm("D1|D2") == "D1|D2"
    assert norm("-") == norm("") == norm("-1") == "-1"


def test_filter(tmp_path):
    title = "Breast or ovarian cancer and aspirin in MODY"
    lines = [f"1|t|{title}\n", "1|a|\n",
             "1\t0\t24\tBreast or ovarian cancer\tDiseaseOrPhenotypicFeature\tD001943,D010051\n",
             "1\t10\t24\tovarian cancer\tDiseaseOrPhenotypicFeature\t D010051\n",
             "1\t29\t36\taspirin\tChemicalEntity\tD001241\n",
             "1\t40\t44\tMODY\tDiseaseOrPhenotypicFeature\t-\n",
             "1\tAssociation\tD001241\tD010051\tNovel\n", "\n"]
    counts = Counter()
    out = tmp_path / "f.pubtator"
    out.write_text("".join(mod["filter_pubtator"](lines, {"DiseaseOrPhenotypicFeature"}, counts)))
    assert counts == {"multi id": 1, "kept": 1, "no id": 1, "other type": 1, "non-mention line": 1}
    d = Preprocessor.load_pubtator_file(str(out))
    assert d["labels"] == ["D010051"] and d["corpus"][0].startswith("ovarian cancer [SEP] ")


def test_bioc(tmp_path):
    p0, p1 = "Aspirin study", "We gave aspirin\nand MESH-less X."
    xml = f"""<collection><document><id>42</id>
      <passage><offset>0</offset><text>{p0}</text>
        <annotation id="1"><infon key="type">Chemical</infon><infon key="identifier">MESH:D001241</infon>
          <location offset="0" length="7"/><text>Aspirin</text></annotation></passage>
      <passage><offset>20</offset><text>{p1}</text>
        <annotation id="2"><infon key="type">Chemical</infon><infon key="identifier">MESH:D001241,MESH:D2</infon>
          <location offset="28" length="7"/><text>aspirin</text></annotation>
        <annotation id="3"><infon key="type">Chemical</infon><infon key="identifier">-</infon>
          <location offset="50" length="1"/><text>X</text></annotation>
        <annotation id="4"><infon key="type">Chemical</infon><infon key="identifier">MESH:D3</infon>
          <location offset="45" length="4"/><location offset="50" length="1"/><text>less X</text></annotation>
        <annotation id="5"><infon key="type">Chemical</infon><infon key="identifier">MESH:D4</infon>
          <location offset="28" length="7"/><text>Aspirin</text></annotation>
        <annotation id="6"><infon key="type">OTHER</infon><infon key="identifier">MESH:D5</infon>
          <location offset="0" length="7"/><text>Aspirin</text></annotation>
        <annotation id="7"><infon key="type">Chemical</infon><infon key="identifier">MESH:D001241</infon>
          <location offset="45" length="4"/><text>less</text></annotation></passage>
    </document></collection>"""
    (tmp_path / "42_v1.xml").write_text(xml)
    counts = Counter()
    out = tmp_path / "b.pubtator"
    out.write_text(mod["bioc_document"](str(tmp_path / "42_v1.xml"), "Chemical", counts))
    assert counts == {"kept": 2, "multi id": 1, "no id": 1, "discontinuous": 1, "text mismatch": 1, "other type": 1}
    d = Preprocessor.load_pubtator_file(str(out))
    assert [c.split(" [SEP]")[0] for c in d["corpus"]] == ["Aspirin", "less"]
    assert d["labels"] == ["D001241", "D001241"]
    doc = d["corpus"][0].split("[SEP] ", 1)[1]
    assert doc[20:20 + len(p1)] == p1.replace("\n", " ") and doc[d["spans"][1][0]:d["spans"][1][1]] == "less"
