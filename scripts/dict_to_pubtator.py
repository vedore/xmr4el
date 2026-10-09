"""CTD vocabulary (CTD_diseases / CTD_chemicals .tsv.gz) -> PubTator pseudo-documents, one per (name, MeSH id).

Each name/synonym becomes a document whose title is the name and whose single mention spans it, so the
standard PubTator loader trains on dictionary names with no reader changes. Append to a training corpus:
    cat CDR_TrainingSet.PubTator.txt ctd_disease.pubtator > train_plus_dict.txt
Only MESH ids are kept (prefix stripped, matching BC5CDR); OMIM-only concepts are skipped unless `-omim` (id kept as
"OMIM:<n>", matching NCBI-disease). Names = the name column + every "*Synonyms" column of the file's "# Fields:" header
(diseases: Synonyms; chemicals: MESHSynonyms, CTDCuratedSynonyms).
"""
import argparse
import gzip
import os


def convert(lines, entity_type, omim=False):
    """Yield PubTator blocks; names deduplicated case-insensitively per id. The column header ("# <Name>\t<ID>...",
    the last tab-separated comment line before the data) picks the synonym columns; no header is an error."""
    n, syn_cols = 0, None
    for line in lines:
        if line.startswith("#"):
            if "\t" in line:
                fields = line[1:].strip().split("\t")
                syn_cols = [i for i, f in enumerate(fields) if f.endswith("Synonyms")]
            continue
        if not line.strip():
            continue
        if syn_cols is None:
            raise ValueError("no '# <Name>\\t<ID>...' column header before the data")
        cols = line.rstrip("\n").split("\t")
        if cols[1].startswith("MESH:"):
            cid = cols[1][len("MESH:"):]
        elif omim and cols[1].startswith("OMIM:"):
            cid = cols[1]
        else:
            continue
        seen = set()
        synonyms = [x for i in syn_cols if i < len(cols) and cols[i] for x in cols[i].split("|")]
        for name in [cols[0], *synonyms]:
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
    parser.add_argument("-omim", action="store_true", help="also keep OMIM-id concepts (as OMIM:<n>)")
    args = parser.parse_args()
    if os.path.exists(args.out) and os.path.samefile(args.out, args.ctd_path):
        parser.error(f"output {args.out} is the input file")
    with gzip.open(args.ctd_path, "rt", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        fout.writelines(convert(fin, args.type, omim=args.omim))


if __name__ == "__main__":
    main()
