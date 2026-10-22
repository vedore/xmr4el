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
Encoder calls go through the persistent embedding cache: "encoder_calls" counts transforms vs real encoder forwards
per encode step (a cache hit times an np.load, not the encoder; delete outputs/cache/transformers to time forwards).
-leaf_candidates N...: the tree only, timed once per value (keys xmr4el_lc<N>); the 1-NN baselines are skipped.
"""
import json
import os
import statistics
import time
from argparse import ArgumentParser

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.preprocessing import normalize

from xmr4el import torch_device
from xmr4el.data.readers import Preprocessor
from xmr4el.eval import filter_labels_and_inputs
from xmr4el.features.transformers import Transformer
from xmr4el.features.vectorizers import Vectorizer
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


CALLS = {"transform": 0, "forward": 0}


def counted(fn, key):
    def wrapper(*a, **k):
        CALLS[key] += 1
        return fn(*a, **k)
    return staticmethod(wrapper)


Transformer.transform, Transformer._predict = counted(Transformer.transform, "transform"), \
    counted(Transformer._predict, "forward")


def encoder_calls():
    """Transforms and encoder forwards since the last call (cache hits = transforms - forwards)."""
    out = {**CALLS, "cache_hits": CALLS["transform"] - CALLS["forward"]}
    CALLS.update(transform=0, forward=0)
    return out


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
    ap.add_argument("-leaf_candidates", type=int, nargs="+", default=None,
                    help="time the tree at each value only (no 1-NN baselines)")
    ap.add_argument("-out", required=True)
    args = ap.parse_args()
    m = {"tree": args.xmodel_path, "test_path": args.test_path, "rss_gb": {}, "bytes": {}, "batch_s": {}, "single_ms": {}}

    xm, m["load_s"] = timed(lambda: XModel.load(args.xmodel_path, research=True))
    m["labels"], m["rss_gb"]["tree_loaded"] = len(xm.initial_labels), rss_gb()
    m["bytes"]["tree_dir"] = dir_bytes(args.xmodel_path)
    test = Preprocessor.load_pubtator_file(args.test_path, window=xm.context_window, abbrev=xm.abbrev_expansion)
    _, texts = filter_labels_and_inputs(test["corpus"], test["labels"], xm.initial_labels)
    n1 = min(args.single, len(texts))
    m["rows"], m["train_rows"], m["search"] = len(texts), xm.X.shape[0], xm.resolve_predict_config()

    encoder_calls()
    m["single_ms"]["encode"] = single_ms(lambda i: xm.text_encoder.predict(texts[i:i + 1]), n1)
    m["encoder_calls"] = {"single": encoder_calls()}
    Xq, m["batch_s"]["encode"] = timed(lambda: xm.text_encoder.predict(texts))
    m["encoder_calls"]["batch"] = encoder_calls()
    _, m["first_call_s"] = timed(lambda: xm.predict_encoded(Xq[:1]))  # one-off: builds the knn index

    for lc in args.leaf_candidates or [None]:
        key = "xmr4el" if lc is None else f"xmr4el_lc{lc}"
        m["single_ms"][key] = single_ms(lambda i: xm.predict_encoded(Xq[i:i + 1], leaf_candidates=lc), n1)
        _, m["batch_s"][key] = timed(lambda: xm.predict_encoded(Xq, leaf_candidates=lc))
    m["rss_gb"]["after_xmr4el"] = rss_gb()
    if args.leaf_candidates:
        return finish(m, args.out, len(texts))

    if xm.features == "sapbert_char_context":
        sapbert_baseline(m, xm, Xq, args.chunk, n1, len(texts))
    else:  # no mention-encoder block to compare against
        m["sapbert"] = f"skipped: features {xm.features!r} have no mention encoder"

    train_mentions = [t.split("[SEP]", 1)[0] for g in xm.training_set for t in g]
    queries = [t.split("[SEP]", 1)[0] for t in texts]
    vec = Vectorizer.fit(train_mentions, xm.vectorizer_config)  # the encoder's defaults
    T = normalize(vec.transform(train_mentions)).T.tocsr()
    m["bytes"]["tfidf_index"] = T.data.nbytes + T.indices.nbytes + T.indptr.nbytes

    def tfidf(lo, hi):
        Qt = normalize(vec.transform(queries[lo:hi]))
        return np.concatenate([np.asarray((Qt[s:s + args.chunk] @ T).argmax(1)).ravel()
                               for s in range(0, hi - lo, args.chunk)])
    m["single_ms"]["tfidf"] = single_ms(lambda i: tfidf(i, i + 1), n1)
    _, m["batch_s"]["tfidf"] = timed(lambda: tfidf(0, len(texts)))
    m["rss_gb"]["after_tfidf"] = rss_gb()

    finish(m, args.out, len(texts))


def sapbert_baseline(m, xm, Xq, chunk, n1, n):
    """SapBERT 1-NN over the tree's training rows (mention block; dense even when X is CSR) on the device."""
    dev = torch_device()
    block = xm.mention_block()
    dense = lambda a: normalize(np.asarray(a.toarray() if issparse(a) else a, dtype=np.float32))
    B = torch.from_numpy(dense(xm.X[:, block])).to(dev)
    Q = torch.from_numpy(dense(Xq[:, block])).to(dev)
    m["bytes"]["sapbert_index"], m["device"] = B.element_size() * B.nelement(), str(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def sap(lo, hi):  # .cpu() waits for the device
        return torch.cat([(Q[s:min(s + chunk, hi)] @ B.T).argmax(1) for s in range(lo, hi, chunk)]).cpu()
    m["single_ms"]["sapbert"] = single_ms(lambda i: sap(i, i + 1), n1)
    _, m["batch_s"]["sapbert"] = timed(lambda: sap(0, n))
    if dev.type == "cuda":
        m["sapbert_gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    m["rss_gb"]["after_sapbert"] = rss_gb()


def finish(m, out, n):
    m["batch_ms_per_query"] = {k: 1000 * v / n for k, v in m["batch_s"].items()}
    print(json.dumps(m, indent=1))
    with open(out, "w") as f:
        json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
