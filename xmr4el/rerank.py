"""Cross-encoder reranker over the tree's top-K candidates (STATUS.md § R).

One shared model scores every (query text, label text) pair, so logits compare across labels.
Query text = the PubTator row "mention [SEP] context"; label text = `label_texts`."""
import json
import logging
import os
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from xmr4el import torch_device
from xmr4el.eval import mention_key

logger = logging.getLogger(__name__)


def label_texts(training_set, Y, n_strings=5):
    """Label text j (column j of Y, `initial_labels[j]`): its `n_strings` most frequent distinct training mention
    strings, `; `-joined. Ties keep first-seen row order, so on train_plus_ctd the annotators' strings (BC5CDR rows
    come first) precede the CTD names. training_set = the tree's grouped texts (`XModel.training_set`)."""
    flat = [t for group in training_set for t in group]  # row i = row i of X and Y (Preprocessor.prepare_data)
    Y = Y.tocsc()
    assert Y.shape[0] == len(flat), f"{len(flat)} training texts vs {Y.shape[0]} Y rows"
    out = []
    for j in range(Y.shape[1]):
        rows = np.sort(Y.indices[Y.indptr[j]:Y.indptr[j + 1]])
        counts = Counter(mention_key(flat[r]) for r in rows)
        out.append("; ".join(s for s, _ in counts.most_common(n_strings)))
    return out


def top_k(score_csr, k):
    """Per row: label indices of the k best scores, best first (fewer if the row has fewer candidates), and scores."""
    idx, val = [], []
    for i in range(score_csr.shape[0]):
        s, e = score_csr.indptr[i], score_csr.indptr[i + 1]
        order = np.argsort(-score_csr.data[s:e], kind="stable")[:k]
        idx.append(score_csr.indices[s:e][order])
        val.append(score_csr.data[s:e][order])
    return idx, val


def rerank_order(tree_scores, logits, w):
    """Top-K positions reordered by log(tree score) + w * logit, best first (w = inf: logit, ties by tree score).
    Ranks below K are the tree's."""
    if np.isinf(w):
        return np.lexsort((-tree_scores, -logits))
    return np.argsort(-(np.log(tree_scores) + w * logits), kind="stable")


class CrossEncoderReranker:
    """BERT-style encoder + linear head -> one logit per (query, label text) pair."""

    def __init__(self, model, tokenizer, meta, max_length=96):
        self.model, self.tokenizer, self.meta, self.max_length = model, tokenizer, meta, max_length
        self.device = torch_device()
        self.model.to(self.device)

    @classmethod
    def from_pretrained(cls, name, meta, max_length=96):
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
        return cls(model, AutoTokenizer.from_pretrained(name), meta, max_length)

    def _logits(self, queries, labels):
        enc = self.tokenizer(queries, labels, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        return self.model(**enc).logits.squeeze(-1)

    def fit(self, queries, candidates, label_text, epochs=2, rows_per_batch=4, lr=2e-5, seed=0):
        """queries[i] with candidates[i] = K label indices, gold first; listwise softmax CE over the K."""
        k = len(candidates[0])
        assert all(len(c) == k for c in candidates), "every row needs exactly K candidates"
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        n_steps = epochs * -(-len(queries) // rows_per_batch)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * n_steps), n_steps)
        target = torch.zeros(rows_per_batch, dtype=torch.long, device=self.device)  # gold = position 0
        self.model.train()
        for epoch in range(epochs):
            start, total = time.perf_counter(), 0.0
            order = rng.permutation(len(queries))
            for step, b in enumerate(range(0, len(order), rows_per_batch)):
                rows = order[b:b + rows_per_batch]
                logits = self._logits([queries[i] for i in rows for _ in range(k)],
                                      [label_text[j] for i in rows for j in candidates[i]]).view(len(rows), k)
                loss = torch.nn.functional.cross_entropy(logits, target[:len(rows)])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item()
                if (step + 1) % 100 == 0:
                    logger.info("Reranker epoch %d step %d: loss=%.4f pairs/s=%.1f", epoch + 1, step + 1,
                                total / (step + 1), (b + len(rows)) * k / (time.perf_counter() - start))
            logger.info("Reranker epoch %d done: loss=%.4f elapsed=%.1fs", epoch + 1, total / (step + 1),
                        time.perf_counter() - start)
        self.model.eval()

    @torch.no_grad()
    def score(self, queries, labels, batch_size=64):
        """Logit per (queries[i], labels[i]) pair, float64 array."""
        self.model.eval()
        out = [self._logits(queries[b:b + batch_size], labels[b:b + batch_size]).float().cpu().numpy()
               for b in range(0, len(queries), batch_size)]
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=False)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "reranker.json"), "w") as f:
            json.dump({**self.meta, "max_length": self.max_length}, f, indent=2)

    @classmethod
    def load(cls, load_dir, tree=None):
        """tree (saved tree directory name): refuse a reranker trained on another tree's candidates/label texts."""
        with open(os.path.join(load_dir, "reranker.json")) as f:
            meta = json.load(f)
        if tree is not None and meta["tree"] != tree:
            raise ValueError(f"reranker {load_dir} was trained on tree {meta['tree']}, not {tree}")
        max_length = meta.pop("max_length")
        model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        return cls(model, AutoTokenizer.from_pretrained(load_dir), meta, max_length)
