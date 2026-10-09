"""CTD vocabulary (CTD_diseases / CTD_chemicals .tsv.gz) -> PubTator pseudo-documents, one per (name, MeSH id).

Each name/synonym becomes a document whose title is the name and whose single mention spans it, so the
standard PubTator loader trains on dictionary names with no reader changes. Append to a training corpus:
    cat CDR_TrainingSet.PubTator.txt ctd_disease.pubtator > train_plus_dict.txt
Only MESH ids are kept (prefix stripped, matching BC5CDR); OMIM-only concepts are skipped.
"""
import argparse
import gzip
import os


def convert(lines, entity_type):
    """Yield PubTator blocks; names deduplicated case-insensitively per id."""
    n = 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        cols = line.rstrip("\n").split("\t")
        if not cols[1].startswith("MESH:"):
            continue
        cid = cols[1][len("MESH:"):]
        seen = set()
        for name in [cols[0], *(cols[7].split("|") if len(cols) > 7 and cols[7] else [])]:
            name = name.strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            pid = f"ctd{n}"
            n += 1
            yield f"{pid}|t|{name}\n{pid}|a|\n{pid}\t0\t{len(name)}\t{name}\t{entity_type}\t{cid}\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ctd_path", required=True, help="CTD_diseases.tsv.gz or CTD_chemicals.tsv.gz")
    parser.add_argument("-out", required=True)
    parser.add_argument("-type", required=True, help="entity type column, e.g. Disease or Chemical")
    args = parser.parse_args()
    if os.path.exists(args.out) and os.path.samefile(args.out, args.ctd_path):
        parser.error(f"output {args.out} is the input file")
    with gzip.open(args.ctd_path, "rt", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        fout.writelines(convert(fin, args.type))


if __name__ == "__main__":
    main()
