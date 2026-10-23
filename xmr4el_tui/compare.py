"""Comparability verdict and metric rows for two selected evaluations (tui.md §5.8, §11). Stdlib only.

Never reads a corpus or loads a pickle: hashes come from metrics_<split>.json and provenance.json; runs recorded
before the population hash are compared by derivation through the tree's training run.
"""
from dataclasses import dataclass
from pathlib import Path

from xmr4el_tui.fmt import DASH, count, metric
from xmr4el_tui.runs import training_run

RECALL = ("R@5", "R@10", "R@20", "R@50", "R@100")
PREPROCESSING = ("context_window", "abbrev_expansion")


@dataclass(frozen=True)
class Verdict:
    level: str                # "hash" | "derived" | "unknown" | "different"
    reasons: tuple[str, ...]  # header lines
    differs: tuple[str, ...]  # tree, beam_size 10 -> 5, packages, ...
    subset_deltas: bool       # acc@1, MRR, R@k, R@cand
    all_rows_delta: bool      # acc@1_all
    breakdown_same: bool      # hybrid inputs (eval train_path) hash equal
    timing_delta: bool
    devices: tuple[str | None, str | None] = (None, None)


def _sha(run, path):
    return ((run.provenance() or {}).get("inputs") or {}).get(path) if path else None


def _preprocessing(run, metrics, runs):
    """Resolved preprocessing of the evaluated tree: from the metrics, else from its training run's config."""
    if isinstance(metrics.get("preprocessing"), dict):
        return metrics["preprocessing"]
    train = training_run(runs, run.tree_id)
    cfg = train.model_config() if train else None
    return {k: cfg.get(k) for k in PREPROCESSING} if isinstance(cfg, dict) else None


def _train_sha(run, runs):
    """sha256 of the training file of the run's tree, None if its training run is not here."""
    train = training_run(runs, run.tree_id)
    spec = train.spec() if train else None
    return _sha(train, spec.get("train_path")) if isinstance(spec, dict) else None


def _clean_commit(runs_involved):
    """The one clean commit of every run, else None."""
    gits = [r.git() or {} for r in runs_involved]
    commits = {g.get("commit") for g in gits}
    ok = len(commits) == 1 and None not in commits and all(g.get("dirty") is False for g in gits)
    return commits.pop() if ok else None


def _differs(a, ma, b, mb):
    out = ["tree"] if ma.get("tree") != mb.get("tree") else []
    sa, sb = ma.get("search") or {}, mb.get("search") or {}
    out += [f"{k} {sa.get(k)} -> {sb.get(k)}" for k in dict.fromkeys([*sa, *sb]) if sa.get(k) != sb.get(k)]
    if ma.get("features") != mb.get("features"):
        out.append(f"features {ma.get('features')} -> {mb.get('features')}")
    pa, pb = ((r.provenance() or {}).get("packages") or {} for r in (a, b))
    out += [f"{k} {pa.get(k)} -> {pb.get(k)}" for k in dict.fromkeys([*pa, *pb]) if pa.get(k) != pb.get(k)]
    return tuple(out)


def verdict(a, a_split, b, b_split, runs):
    """§11 for evaluation `a_split` of run `a` against `b_split` of `b`; `runs` = the scan (training runs)."""
    ea, eb = a.eval(a_split), b.eval(b_split)
    missing = [f"{side}: no metrics for {split} (not produced yet)"
               for side, e, split in (("A", ea, a_split), ("B", eb, b_split)) if e is None or e.metrics is None]
    if missing:
        return Verdict("unknown", tuple(missing), (), False, False, False, False)
    ma, mb = ea.metrics, eb.metrics
    differs = _differs(a, ma, b, mb)
    devices = tuple((r.provenance() or {}).get("torch_device") for r in (a, b))
    same_rows = ma.get("rows_total") == mb.get("rows_total")
    breakdown = _sha(a, ea.train_path)
    breakdown_same = breakdown is not None and breakdown == _sha(b, eb.train_path)

    def result(level, reasons, subset=False, all_rows=False):
        return Verdict(level, tuple(reasons), differs, subset, all_rows, breakdown_same,
                       subset and None not in devices and devices[0] == devices[1], devices)

    side_by_side = "Values side by side; no differences computed."
    sa, sb = _sha(a, ea.test_path), _sha(b, eb.test_path)
    if sa is None or sb is None:
        return result("unknown", ["Comparability unknown: evaluation file hash not recorded.", side_by_side])
    if sa != sb:
        na, nb = Path(ea.test_path or "").name, Path(eb.test_path or "").name
        files = f"{na} vs {nb}" if na != nb else f"{na}, contents differ"
        return result("different", [f"Not comparable: different evaluation files ({files}).", side_by_side])
    pa, pb = _preprocessing(a, ma, runs), _preprocessing(b, mb, runs)
    if pa is None or pb is None:
        return result("unknown", ["Comparability unknown: preprocessing not recorded "
                                  "(training run of the tree not found).", side_by_side])
    if pa != pb:
        diff = ", ".join(f"{k} {pa.get(k)} vs {pb.get(k)}" for k in PREPROCESSING if pa.get(k) != pb.get(k))
        return result("different", [f"Not comparable: different preprocessing ({diff}).", side_by_side])

    all_rows_note = "acc@1_all compared over all loaded rows (out-of-vocabulary gold = miss)."
    ha, hb = ma.get("population_sha256"), mb.get("population_sha256")
    if ha and hb:
        if ha == hb:
            return result("hash", ["Evaluation file, preprocessing, and evaluated population: hashes match"],
                          subset=True, all_rows=True)
        return result("different", ["Same evaluation file; in-vocabulary populations differ.", all_rows_note],
                      all_rows=same_rows)
    ta, tb = _train_sha(a, runs), _train_sha(b, runs)
    if ta is None or tb is None:
        return result("unknown", ["Comparability unknown: training run of the tree not found; "
                                  "no subset differences computed.", all_rows_note], all_rows=same_rows)
    if ta != tb:
        return result("different", ["Same evaluation file; in-vocabulary populations differ "
                                    "(different training files).", all_rows_note], all_rows=same_rows)
    involved = {r.id: r for r in (a, b, training_run(runs, a.tree_id), training_run(runs, b.tree_id))}
    if _clean_commit(involved.values()) is None:
        return result("unknown", ["Comparability unknown: runs are not all at one clean commit; "
                                  "no subset differences computed.", all_rows_note], all_rows=same_rows)
    return result("derived", ["Populations equal by derivation (same files, preprocessing, commit)"],
                  subset=True, all_rows=True)


def default_splits(a, b):
    """First split of A (spec order) that B also has; else the first evaluation on each side."""
    in_b = {e.split for e in b.evals}
    shared = next((e.split for e in a.evals if e.split in in_b), None)
    if shared:
        return shared, shared
    return (a.evals[0].split if a.evals else None), (b.evals[0].split if b.evals else None)


def rows(ma, mb, v):
    """[(metric, A, B, B - A or "")] for every recorded metric; a delta only where the verdict allows it."""
    def num(key, allowed, fmt=metric, dfmt="{:+.4f}"):
        x, y = ma.get(key), mb.get(key)
        d = dfmt.format(y - x) if allowed and isinstance(x, (int, float)) and isinstance(y, (int, float)) else ""
        return key, fmt(x), fmt(y), d

    def n_rows(m):
        return f"{count(m.get('rows'))} / {count(m.get('rows_total'))}" if "rows" in m else DASH

    out = [("rows", n_rows(ma), n_rows(mb), ""), num("acc@1", v.subset_deltas), num("acc@1_all", v.all_rows_delta),
           num("MRR", v.subset_deltas)]
    out += [num(k, v.subset_deltas) for k in RECALL if k in ma or k in mb]
    out.append(num("R@cand", v.subset_deltas))
    if "hybrid" in ma or "hybrid" in mb:
        out += [num("hybrid", v.subset_deltas and v.breakdown_same),
                num("hybrid_all", v.all_rows_delta and v.breakdown_same)]
    key, x, y, d = num("ms_per_query", v.timing_delta, fmt=lambda t: DASH if t is None else f"{t:.2f}",
                       dfmt="{:+.2f}")
    da, db = v.devices
    if d:
        d += "   same device"
    elif da and db and da != db:
        d = f"devices differ: {da} vs {db}"
    out.append(("ms/query", x, y, d))
    return out
