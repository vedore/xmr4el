"""Split a local PubTator corpus into train/dev/test files."""
import argparse
import os
from xmr4el.data.splits import (deterministic_split, parse_pubtator, write_pubtator_file,
                                emit_jsonl_for_split, load_pmids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full PubTator corpus file")
    parser.add_argument("--outdir", required=True, help="Output directory to write splits")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--train_pmids", help="Optional file with official train PMIDs")
    parser.add_argument("--dev_pmids", help="Optional file with official dev PMIDs")
    parser.add_argument("--test_pmids", help="Optional file with official test PMIDs")
    parser.add_argument("--emit_jsonl", action="store_true", help="Also emit JSONL for each split")
    args = parser.parse_args()

    names = ["corpus_pubtator_%s.txt" % s for s in ("train", "dev", "test")]
    if args.emit_jsonl:
        names += ["%s.jsonl" % s for s in ("train", "dev", "test")]
    for name in names:
        out = os.path.join(args.outdir, name)
        if os.path.exists(out) and os.path.samefile(out, args.input):
            parser.error(f"output {out} is the input file")

    pmid_blocks = parse_pubtator(args.input)
    pmids = sorted(pmid_blocks.keys())

    splits = {"train": [], "dev": [], "test": []}

    if args.train_pmids and args.dev_pmids and args.test_pmids:
        # Use official PMID splits
        splits["train"] = load_pmids(args.train_pmids)
        splits["dev"] = load_pmids(args.dev_pmids)
        splits["test"] = load_pmids(args.test_pmids)
        seen = {}
        for name, ids in splits.items():
            for pmid in ids:
                if pmid in seen:
                    parser.error(f"PMID {pmid} listed in {seen[pmid]} and in {name}")
                seen[pmid] = name
    else:
        # Deterministic split
        for pmid in pmids:
            s = deterministic_split(pmid, args.train_ratio, args.dev_ratio, args.test_ratio)
            splits[s].append(pmid)

    # Write PubTator files
    os.makedirs(args.outdir, exist_ok=True)
    write_pubtator_file(splits["train"], pmid_blocks, os.path.join(args.outdir, "corpus_pubtator_train.txt"))
    write_pubtator_file(splits["dev"], pmid_blocks, os.path.join(args.outdir, "corpus_pubtator_dev.txt"))
    write_pubtator_file(splits["test"], pmid_blocks, os.path.join(args.outdir, "corpus_pubtator_test.txt"))

    # Optional JSONL
    if args.emit_jsonl:
        emit_jsonl_for_split(splits["train"], pmid_blocks, os.path.join(args.outdir, "train.jsonl"))
        emit_jsonl_for_split(splits["dev"], pmid_blocks, os.path.join(args.outdir, "dev.jsonl"))
        emit_jsonl_for_split(splits["test"], pmid_blocks, os.path.join(args.outdir, "test.jsonl"))

    print("Done. Split sizes: train=%d dev=%d test=%d" %
          (len(splits["train"]), len(splits["dev"]), len(splits["test"])))

if __name__ == "__main__":
    main()
