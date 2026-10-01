import os

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
    def prepare_data_older(
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
    def load_pubtator_file(pubtator_filepath: str, window: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Load a PubTator file and return flattened corpus and labels lists
        suitable for entity linking training.

        Each entry:
            corpus[i] = "mention [SEP] context"; context = title + " " + abstract, or with
                        `window`, the words around the mention (`context_window`)
            labels[i] = CUI
            spans[i] = (start, end) of the mention in title + " " + abstract (PubTator offsets)
        """
        assert os.path.exists(pubtator_filepath), f"{pubtator_filepath} does not exist"

        corpus: List[str] = []
        labels: List[str] = []
        spans: List[Tuple[int, int]] = []  # mention offsets into the context after "[SEP] "

        title, abstract = "", ""

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
