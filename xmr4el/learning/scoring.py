"""Label scores shared by prediction and the diagnostic scripts."""
import numpy as np
from sklearn.preprocessing import normalize


def label_max_cos(Q, B, b_labels, n_labels, chunk=256, rows=None, cols=None):
    """(n_Q, n_labels): max cosine of each Q row to the B rows of each label (nearest-synonym score, as SapBERT
    links mentions); -1 for labels with no B row.

    With `rows`, `cols` (query and label index per pair, `rows` ascending, e.g. CSR order): only those pairs, as a
    1-D array, so the (n_Q, n_labels) matrix is never built; each Q chunk is multiplied only with the B rows of
    the labels its pairs name (`label_rows`)."""
    order = np.argsort(b_labels, kind="stable")
    present, starts = np.unique(b_labels[order], return_index=True)
    B = normalize(np.asarray(B[order].toarray() if hasattr(B, "toarray") else B[order], dtype=np.float32))
    if rows is None:
        out = np.full((Q.shape[0], n_labels), -1.0, dtype=np.float32)
    else:
        out = np.full(len(rows), -1.0, dtype=np.float32)
        col_of = np.full(n_labels, -1)
        col_of[present] = np.arange(len(present))
        bounds = np.searchsorted(rows, np.arange(0, Q.shape[0] + chunk, chunk))
    for i, s in enumerate(range(0, Q.shape[0], chunk)):
        if rows is not None and bounds[i] == bounds[i + 1]:
            continue
        q = Q[s:s + chunk]
        q = normalize(np.asarray(q.toarray() if hasattr(q, "toarray") else q, dtype=np.float32))
        if rows is None:
            out[s:s + len(q), present] = np.maximum.reduceat(q @ B.T, starts, axis=1)
        else:
            lo, hi = bounds[i], bounds[i + 1]
            c = col_of[cols[lo:hi]]
            used = np.unique(c[c >= 0])  # indices into `present`
            if not len(used):
                continue
            idx, local = label_rows(starts, len(B), used)
            m = np.maximum.reduceat(q @ B[idx].T, local, axis=1)  # (len(q), len(used))
            out[lo:hi] = np.where(c >= 0, m[rows[lo:hi] - s, np.searchsorted(used, np.maximum(c, 0))], -1.0)
    return out


def label_rows(starts, n_rows, groups):
    """Row indices of the contiguous groups `groups` (ascending) of a matrix whose group g starts at row
    `starts[g]`, and each selected group's start inside that selection (`np.maximum.reduceat` offsets)."""
    ends = np.append(starts[1:], n_rows)[groups]
    lengths = ends - starts[groups]
    local = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return np.repeat(starts[groups] - local, lengths) + np.arange(lengths.sum()), local
