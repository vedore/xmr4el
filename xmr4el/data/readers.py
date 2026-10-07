"""Local readers, label grouping and text preparation."""
import re
import os
import pandas as pd
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

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
        if len(X_train) != len(Y_train):
            raise ValueError("Training text groups and labels must have the same length")
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
        BC5CDR: "-1" ids are skipped; composite "D1|D2" mentions are split via column 7.
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
                        # BC5CDR: "-1" = no concept (dropped, also as a composite part); a composite
                        # mention "D1|D2" becomes one row per part, its text from column 7
                        # ("t1|t2"); a composite without column 7 has no per-part text and is dropped.
                        cuis = parts[5].split("|")
                        texts = parts[6].split("|") if len(cuis) > 1 and len(parts) > 6 else [parts[3]]
                        if len(texts) != len(cuis):
                            continue
                        span = (int(parts[1]), int(parts[2]))
                        doc = " ".join(x for x in [title, abstract] if x)
                        if abbrev is not None and abbrs is None:
                            abbrs = Preprocessor.abbreviations(doc)
                        context = doc if window is None else Preprocessor.context_window(doc, span, window)

                        for mention_text, cui in zip(texts, cuis):
                            if cui == "-1":
                                continue
                            if abbrev is not None:
                                lf = abbrs.get(mention_text.strip())
                                if lf is not None:
                                    mention_text = lf if abbrev == "replace" else f"{mention_text.strip()} {lf}"
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
