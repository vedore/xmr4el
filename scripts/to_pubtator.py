"""Corpus -> PubTator in the form the PubTator loader reads (one id per mention, "-1" = no concept).

filter: PubTator -> PubTator, keeping the annotations of the given entity types (all if none given); relation and other
        non-mention lines are dropped. Ids: spaces and "MESH:" prefixes stripped, "-" -> "-1", the "," / "+" multi-id
        separators -> "|" (the loader drops a multi-id mention without per-part texts, as for BC5CDR composites).
        NCBI-disease: -types SpecificDisease DiseaseClass Modifier CompositeMention; BioRED:
        -types DiseaseOrPhenotypicFeature or -types ChemicalEntity.
bioc:   BioC XML (NLM-Chem full texts, one <pmcid>_v1.xml per document) -> PubTator. The whole article is the title text
        (each passage at its BioC offset, newlines/tabs as spaces), so the BioC offsets are the PubTator offsets.
        Annotations of -type with one location whose text matches the article; ids normalised as in filter.
Both print the kept / skipped counts.
"""
import argparse
import os
import xml.etree.ElementTree as ET
from collections import Counter


def norm_id(raw):
    """"MESH:D1, MESH:D2" -> "D1|D2"; "-" or empty -> "-1"; other prefixes (OMIM:) kept."""
    parts = [p.strip() for p in raw.replace("+", ",").replace("|", ",").split(",")]
    parts = [p[len("MESH:"):] if p.startswith("MESH:") else p for p in parts if p]
    return "|".join(parts) if parts and parts != ["-"] else "-1"


def filter_pubtator(lines, types, counts):
    """Yield the PubTator lines of `lines` with only the mentions of `types` (None = all), ids normalised."""
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if "|t|" in line or "|a|" in line:
            yield line.rstrip("\n") + "\n"
        elif not line.strip():
            yield "\n"
        elif len(cols) >= 6 and cols[1].isdigit():
            if types and cols[4] not in types:
                counts["other type"] += 1
                continue
            cols[5] = norm_id(cols[5])
            counts["multi id" if "|" in cols[5] else "no id" if cols[5] == "-1" else "kept"] += 1
            yield "\t".join(cols) + "\n"
        else:
            counts["non-mention line"] += 1


def bioc_document(path, entity_type, counts):
    """One PubTator block for the first document of the BioC file `path`."""
    doc = ET.parse(path).getroot().find("document")
    doc_id = doc.findtext("id")
    passages = [(int(p.findtext("offset")), p.findtext("text") or "", p) for p in doc.iter("passage")]
    size = max((o + len(t) for o, t, _ in passages), default=0)
    buf = [" "] * size
    for o, t, _ in passages:
        buf[o:o + len(t)] = t.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = "".join(buf)
    out = [f"{doc_id}|t|{text}\n", f"{doc_id}|a|\n"]
    for _, _, p in passages:
        for a in p.findall("annotation"):
            infons = {i.get("key"): i.text or "" for i in a.findall("infon")}
            if infons.get("type") != entity_type:
                counts["other type"] += 1
                continue
            locs = a.findall("location")
            if len(locs) != 1:
                counts["discontinuous"] += 1
                continue
            start, length = int(locs[0].get("offset")), int(locs[0].get("length"))
            mention = a.findtext("text") or ""
            if text[start:start + length] != mention.replace("\n", " ").replace("\t", " "):
                counts["text mismatch"] += 1
                continue
            cid = norm_id(infons.get("identifier", ""))
            counts["multi id" if "|" in cid else "no id" if cid == "-1" else "kept"] += 1
            out.append(f"{doc_id}\t{start}\t{start + length}\t{text[start:start + length]}\t{entity_type}\t{cid}\n")
    return "".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("filter")
    f.add_argument("-input", required=True)
    f.add_argument("-out", required=True)
    f.add_argument("-types", nargs="*", help="entity types to keep (default all)")
    b = sub.add_parser("bioc")
    b.add_argument("-input_dir", required=True, help="folder of <pmcid>_v1.xml BioC files")
    b.add_argument("-pmcids", required=True, help="file with one PMC id per line (the split)")
    b.add_argument("-out", required=True)
    b.add_argument("-type", default="Chemical")
    args = parser.parse_args()
    if os.path.exists(args.out):
        parser.error(f"{args.out} exists")
    counts = Counter()
    with open(args.out, "w", encoding="utf-8") as fout:
        if args.mode == "filter":
            with open(args.input, encoding="utf-8") as fin:
                fout.writelines(filter_pubtator(fin, set(args.types or []), counts))
        else:
            with open(args.pmcids) as fin:
                pmcids = fin.read().split()
            for pmc in pmcids:
                fout.write(bioc_document(os.path.join(args.input_dir, f"{pmc}_v1.xml"), args.type, counts))
            counts["documents"] = len(pmcids)
    print(args.out, dict(counts))


if __name__ == "__main__":
    main()
