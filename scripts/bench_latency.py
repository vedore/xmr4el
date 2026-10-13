"""Plan I3: latency and memory of a saved tree vs SapBERT 1-NN and char TF-IDF 1-NN over the same training rows.

All three search the same in-vocabulary eval rows. Encoding is timed once and reported apart: the tree's features
are [SapBERT mention | char SVD | context SVD], and the SapBERT forward on the mention (shared with SapBERT 1-NN)
dominates. Search times exclude encoding:
  xmr4el  `XModel.predict_encoded` with the tree's predict_config (routing + leaves + knn)
  sapbert 1-NN over the mention block of the training rows: dense matmul + top-1 on the torch device (GPU if any)
  tfidf   1-NN with the char n-gram TF-IDF of the mention string (sparse matmul, CPU; transform included)
Batch = all rows at once (chunked); single = one row per call over the first -single rows (median ms).
Memory: index sizes in bytes (tree directory on disk, 1-NN matrices) and the process RSS after each step
(Linux VmRSS; the tree is loaded first, so 1-NN steps add to it). Writes -out JSON.
"""
import json
import os
import statistics
import time
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from xmr4el import torch_device
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import filter_labels_and_inputs
from xmr4el.xmodel import XModel


def rss_gb():
    """Current resident set size in GB (Linux /proc), None elsewhere."""
    try:
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1e6
    except OSError:
        return None


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def single_ms(fn, n):
    """Median ms of fn(i) over i < n (one warm-up call first). Run before the batch timing: it warms up too."""
    fn(0)
    return 1000 * statistics.median(timed(lambda: fn(i))[1] for i in range(n))


def main():
    ap = ArgumentParser()
    ap.add_argument("-xmodel_path", required=True)
    ap.add_argument("-test_path", required=True)
    ap.add_argument("-single", type=int, default=100, help="rows timed one at a time")
    ap.add_argument("-chunk", type=int, default=1024, help="1-NN query rows per matmul")
    ap.add_argument("-out", required=True)
    args = ap.parse_args()
    m = {"tree": args.xmodel_path, "test_path": args.test_path, "rss_gb": {}, "bytes": {}, "batch_s": {}, "single_ms": {}}

    xm, m["load_s"] = timed(lambda: XModel.load(args.xmodel_path))
    m["labels"], m["rss_gb"]["tree_loaded"] = len(xm.initial_labels), rss_gb()
    m["bytes"]["tree_dir"] = dir_bytes(args.xmodel_path)
    test = Preprocessor.load_pubtator_file(args.test_path, window=xm.context_window, abbrev=xm.abbrev_expansion)
    _, texts = filter_labels_and_inputs(test["corpus"], test["labels"], xm.initial_labels)
    n1 = min(args.single, len(texts))
    m["rows"], m["train_rows"], m["search"] = len(texts), xm.X.shape[0], xm.resolve_predict_config()

    m["single_ms"]["encode"] = single_ms(lambda i: xm.text_encoder.predict(texts[i:i + 1]), n1)
    Xq, m["batch_s"]["encode"] = timed(lambda: xm.text_encoder.predict(texts))
    Xq = Xq.tocsr()

    m["single_ms"]["xmr4el"] = single_ms(lambda i: xm.predict_encoded(Xq[i:i + 1]), n1)
    _, m["batch_s"]["xmr4el"] = timed(lambda: xm.predict_encoded(Xq))
    m["rss_gb"]["after_xmr4el"] = rss_gb()

    dev = torch_device()
    block = xm.mention_block()
    B = torch.from_numpy(normalize(xm.X[:, block].toarray().astype(np.float32))).to(dev)
    Q = torch.from_numpy(normalize(Xq[:, block].toarray().astype(np.float32))).to(dev)
    m["bytes"]["sapbert_index"], m["device"] = B.element_size() * B.nelement(), str(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def sap(lo, hi):  # .cpu() waits for the device
        return torch.cat([(Q[s:min(s + args.chunk, hi)] @ B.T).argmax(1) for s in range(lo, hi, args.chunk)]).cpu()
    m["single_ms"]["sapbert"] = single_ms(lambda i: sap(i, i + 1), n1)
    _, m["batch_s"]["sapbert"] = timed(lambda: sap(0, len(texts)))
    if dev.type == "cuda":
        m["sapbert_gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    m["rss_gb"]["after_sapbert"] = rss_gb()
    del B, Q

    train_mentions = [t.split("[SEP]", 1)[0] for g in xm.training_set for t in g]
    queries = [t.split("[SEP]", 1)[0] for t in texts]
    vec = TfidfVectorizer(**{k: tuple(v) if isinstance(v, list) else v
                             for k, v in xm.vectorizer_config["kwargs"].items()}, dtype=np.float32)
    T = normalize(vec.fit_transform(train_mentions)).T.tocsr()
    m["bytes"]["tfidf_index"] = T.data.nbytes + T.indices.nbytes + T.indptr.nbytes

    def tfidf(lo, hi):
        Qt = normalize(vec.transform(queries[lo:hi]))
        return np.concatenate([np.asarray((Qt[s:s + args.chunk] @ T).argmax(1)).ravel()
                               for s in range(0, hi - lo, args.chunk)])
    m["single_ms"]["tfidf"] = single_ms(lambda i: tfidf(i, i + 1), n1)
    _, m["batch_s"]["tfidf"] = timed(lambda: tfidf(0, len(texts)))
    m["rss_gb"]["after_tfidf"] = rss_gb()

    m["batch_ms_per_query"] = {k: 1000 * v / len(texts) for k, v in m["batch_s"].items()}
    print(json.dumps(m, indent=1))
    with open(args.out, "w") as f:
        json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
