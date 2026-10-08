"""Label scores shared by prediction and the diagnostic scripts."""
import numpy as np
from sklearn.preprocessing import normalize


def label_max_cos(Q, B, b_labels, n_labels, chunk=256):
    """(n_Q, n_labels): max cosine of each Q row to the B rows of each label (nearest-synonym score, as SapBERT
    links mentions); -1 for labels with no B row."""
    order = np.argsort(b_labels, kind="stable")
    present, starts = np.unique(b_labels[order], return_index=True)
    B = normalize(np.asarray(B[order].toarray() if hasattr(B, "toarray") else B[order], dtype=np.float32))
    out = np.full((Q.shape[0], n_labels), -1.0, dtype=np.float32)
    for s in range(0, Q.shape[0], chunk):
        q = Q[s:s + chunk]
        q = normalize(np.asarray(q.toarray() if hasattr(q, "toarray") else q, dtype=np.float32))
        out[s:s + len(q), present] = np.maximum.reduceat(q @ B.T, starts, axis=1)
    return out
