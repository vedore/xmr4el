"""Local data loading (`Preprocessor`) and PubTator train/dev/test splitting.

Split usage:
  python -m xmr4el.data \
    --input MedMentions/st21pv/corpus_pubtator.txt \
    --outdir datasets/medmentions/st21pv/ \
    --train_pmids train_pmids.txt \
    --dev_pmids dev_pmids.txt \
    --test_pmids test_pmids.txt \
    --emit_jsonl
"""
import argparse
import hashlib
import json
import os
import re

import pandas as pd

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

class Preprocessor:
    """Preprocess text to numerical values."""

    @staticmethod
    def load_data_labels_from_file(
        train_filepath: str,
        labels_filepath: str,
        truncate_data: int = 0,
    ) -> Dict[str, List[List[str]]]:
        """Load texts and their corresponding labels from file."""
        
        train_df = pd.read_csv(
            train_filepath,
            header=None,
            names=["group_id", "text"],
            delimiter="\t",
            dtype={"group_id": int, "text": str},
        )
        
        train_df["text"] = train_df["text"].fillna("")

        grouped_texts = train_df.groupby("group_id")["text"].apply(list).reset_index()
        
        with open(labels_filepath, 'r') as f:
            raw_labels = [line.strip() for line in f if line.strip()]

        if truncate_data > 0:
            grouped_texts = grouped_texts.head(truncate_data)
            raw_labels = raw_labels[:truncate_data]

        if len(raw_labels) > len(grouped_texts):
            raw_labels = raw_labels[:len(grouped_texts)]
            print(
                f"Warning: Truncated labels from {len(raw_labels)} to {len(grouped_texts)} to match text groups"
            )
        elif len(raw_labels) < len(grouped_texts):
            raise Exception(
                f"Mismatch: Not enough labels ({len(raw_labels)}) for text groups ({len(grouped_texts)})"
            )

        grouped_texts["concept_id"] = raw_labels

        grouped_texts["original_texts"] = grouped_texts["text"]

        return {
            "corpus": grouped_texts["original_texts"].tolist(),
            "labels": grouped_texts["concept_id"].tolist(),
        }

    @staticmethod
    def prepare_data(
        X_train: Sequence[Sequence[str]],
        Y_train: Sequence[str],
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        """Flatten synonyms and build reverse label index mapping."""
        trn_corpus: List[str] = []
        label_to_indices: Dict[str, List[int]] = defaultdict(list)

        for label, synonyms in zip(Y_train, X_train):
            for synonym in synonyms:
                idx = len(trn_corpus)
                trn_corpus.append(synonym)
                label_to_indices[label].append(idx)

        return trn_corpus, label_to_indices

    @staticmethod
    def context_window(document: str, span: Tuple[int, int], window: int) -> str:
        """Up to `window` words left and right of the mention at `span`, mention excluded."""
        start, end = span
        return " ".join(document[:start].split()[-window:] + document[end:].split()[:window])

    @staticmethod
    def best_long_form(sf, lf):
        """Schwartz & Hearst 2003: match the short form's characters right to left inside the candidate
        long form; its first character must start a word. Returns the long form or None."""
        s, l = len(sf) - 1, len(lf) - 1
        while s >= 0:
            c = sf[s].lower()
            if not c.isalnum():
                s -= 1
                continue
            while l >= 0 and (lf[l].lower() != c or (s == 0 and l > 0 and lf[l - 1].isalnum())):
                l -= 1
            if l < 0:
                return None
            l -= 1
            s -= 1
        return lf[lf.rfind(" ", 0, l + 1) + 1:]

    @staticmethod
    def abbreviations(document):
        """{short form: long form} for every "long form (SF)" in the document; first definition wins."""
        out = {}
        for m in re.finditer(r"\(([^()]+)\)", document):
            sf = re.split(r"[,;]", m.group(1))[0].strip()
            if not (2 <= len(sf) <= 10 and len(sf.split()) <= 2 and sf[0].isalnum()
                    and any(ch.isalpha() for ch in sf)) or sf in out:
                continue
            words = re.split(r"[.;!?]\s", document[:m.start()])[-1].split()
            lf = Preprocessor.best_long_form(sf, " ".join(words[-min(len(sf) + 5, 2 * len(sf)):]))
            if lf and len(lf) > len(sf) and sf not in lf.split():
                out[sf] = lf
        return out

    @staticmethod
    def load_pubtator_file(pubtator_filepath: str, window: Optional[int] = None,
                           abbrev: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Load a PubTator file and return flattened corpus and labels lists
        suitable for entity linking training.

        Each entry:
            corpus[i] = "mention [SEP] context"; context = title + " " + abstract, or with
                        `window`, the words around the mention (`context_window`)
            labels[i] = CUI
            spans[i] = (start, end) of the mention in title + " " + abstract (PubTator offsets)
        `abbrev` "append" / "replace": a mention that is a short form defined in its document
        (`abbreviations`) becomes "SF long form" / "long form"; None keeps it.
        """
        assert abbrev in (None, "append", "replace"), abbrev
        assert os.path.exists(pubtator_filepath), f"{pubtator_filepath} does not exist"

        corpus: List[str] = []
        labels: List[str] = []
        spans: List[Tuple[int, int]] = []  # mention offsets into the context after "[SEP] "

        title, abstract, abbrs = "", "", None

        with open(pubtator_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # --- Title line ---
                if "|t|" in line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        title = parts[2]
                        abstract = ""
                        abbrs = None

                # --- Abstract line ---
                elif "|a|" in line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        abstract = parts[2]

                # --- Annotation line (mentions) ---
                elif "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        mention_text = parts[3]
                        cui = parts[5]
                        context = " ".join(x for x in [title, abstract] if x)

                        span = (int(parts[1]), int(parts[2]))
                        if abbrev is not None:
                            if abbrs is None:
                                abbrs = Preprocessor.abbreviations(context)
                            lf = abbrs.get(mention_text.strip())
                            if lf is not None:
                                mention_text = lf if abbrev == "replace" else f"{mention_text.strip()} {lf}"
                        if window is not None:
                            context = Preprocessor.context_window(context, span, window)
                        corpus.append(f"{mention_text} [SEP] {context}")
                        labels.append(cui)
                        spans.append(span)

        return {"corpus": corpus, "labels": labels, "spans": spans}

    @staticmethod
    def organize_pubtator_output(pub_output: Dict) -> Tuple[List[List[str]], List[str]]:
        """
        Group flat pubtator output by label (CUI) and return:
        - corpus_grouped: List[List[str]] where each inner list contains mentions for the same label
        - labels_grouped: List[str] where labels_grouped[i] is the label for corpus_grouped[i]

        The order is deterministic: first-seen label order in the input is preserved.
        """
        corpus = pub_output["corpus"]
        ids = pub_output["labels"]

        groups: "OrderedDict[str, List[str]]" = OrderedDict()
        for mention, id_ in zip(corpus, ids):
            if id_ not in groups:
                groups[id_] = []
            groups[id_].append(mention)

        corpus_grouped = list(groups.values())
        labels_grouped = list(groups.keys())

        return corpus_grouped, labels_grouped


def deterministic_split(pmid: str, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1) -> str:
    h = hashlib.md5(pmid.encode("utf8")).hexdigest()
    val = int(h[:8], 16) / float(0xFFFFFFFF)
    if val < train_ratio:
        return "train"
    elif val < train_ratio + dev_ratio:
        return "dev"
    else:
        return "test"

def parse_pubtator(path: str) -> dict:
    pmid_blocks = defaultdict(list)
    current_pmid = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "|t|" in line or "|a|" in line:
                pmid = line.split("|", 1)[0]
                current_pmid = pmid
                pmid_blocks[pmid].append(line)
            elif "\t" in line:
                pmid = line.split("\t")[0]
                current_pmid = pmid
                pmid_blocks[pmid].append(line)
            elif current_pmid:
                pmid_blocks[current_pmid].append(line)
    return pmid_blocks

def write_pubtator_file(pmid_list, pmid_blocks, out_path):
    with open(out_path, "w", encoding="utf-8") as fw:
        for pmid in pmid_list:
            if pmid in pmid_blocks:
                for line in pmid_blocks[pmid]:
                    fw.write(line + "\n")
                fw.write("\n")

def emit_jsonl_for_split(pmid_list, pmid_blocks, out_jsonl_path):
    out = []
    for pmid in pmid_list:
        lines = pmid_blocks.get(pmid, [])
        title = ""
        abstract = ""
        for l in lines:
            if "|t|" in l:
                parts = l.split("|", 2)
                if len(parts) == 3:
                    title = parts[2]
            elif "|a|" in l:
                parts = l.split("|", 2)
                if len(parts) == 3:
                    abstract = parts[2]
        context = " ".join(x for x in [title.strip(), abstract.strip()] if x)
        for l in lines:
            if "\t" in l:
                parts = l.split("\t")
                if len(parts) >= 6:
                    pmid_p, start, end, mention, sem_types, cui = parts[:6]
                    entry = {
                        "pmid": pmid_p,
                        "mention": mention,
                        "start": int(start),
                        "end": int(end),
                        "cui": cui,
                        "sem_types": sem_types.split(",") if sem_types else [],
                        "context": context
                    }
                    out.append(entry)
    with open(out_jsonl_path, "w", encoding="utf-8") as fw:
        for e in out:
            fw.write(json.dumps(e, ensure_ascii=False) + "\n")

def load_pmids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

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

    os.makedirs(args.outdir, exist_ok=True)
    pmid_blocks = parse_pubtator(args.input)
    pmids = sorted(pmid_blocks.keys())

    splits = {"train": [], "dev": [], "test": []}

    if args.train_pmids and args.dev_pmids and args.test_pmids:
        # Use official PMID splits
        splits["train"] = load_pmids(args.train_pmids)
        splits["dev"] = load_pmids(args.dev_pmids)
        splits["test"] = load_pmids(args.test_pmids)
    else:
        # Deterministic split
        for pmid in pmids:
            s = deterministic_split(pmid, args.train_ratio, args.dev_ratio, args.test_ratio)
            splits[s].append(pmid)

    # Write PubTator files
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
