"""Results comparison (tui.md §5.12, §11): what is known about each saved evaluation, how a challenger relates to the
baseline (VERIFIED / CAUTION / INCOMPATIBLE, with the reason), and the metric rows of the four analysis groups with
the deltas the evidence allows. Stdlib only, no UI.

Reads only small run files (provenance, git, spec, model_config) of the evaluation's run and of its tree's training
run; never a corpus, never a pickle, never re-evaluates. Identity is the explorer key ("eval", run id, split).
"""
from dataclasses import dataclass
from pathlib import Path

from xmr4el_tui.compare import RECALL, _preprocessing, _sha, _train_sha
from xmr4el_tui.fmt import count, number
from xmr4el_tui.runs import dig, training_run

VERIFIED, CAUTION, INCOMPATIBLE = "VERIFIED", "CAUTION", "INCOMPATIBLE"
MAX_CHALLENGERS = 3
NA = "N/A"
STRING_GROUPS = ("all", "seen, 1 label", "seen, >1 label", "unseen string")
STRING_COLS = (("tree", "model"), ("dict", "dictionary"), ("either", "either"), ("routed", "routed"))
SUBSET = "acc@1, MRR, R@k, R@cand, hybrid"  # in-vocabulary metrics: deltas need equal populations


@dataclass(frozen=True)
class Facts:
    """One evaluation as recorded: its metrics and the hashes / settings that decide comparability."""
    key: tuple
    run: object               # runs.Run
    split: str
    metrics: dict | None      # metrics_<split>.json, None = not written
    test_sha: str | None      # sha256 of the evaluated file (provenance.json)
    strings_sha: str | None   # sha256 of the seen/unseen reference file (the evaluation's train_path)
    tree_train_sha: str | None  # sha256 of the tree's training file (its training run's provenance)
    preprocessing: dict | None
    commits: tuple            # (commit, dirty) of this run and its tree's training run; None = no git.json
    beam: int | None = None


def read_facts(run, e, all_runs):
    """Facts of evaluation `e` of `run` (reads small run files: call off the UI thread)."""
    metrics = e.metrics if isinstance(e.metrics, dict) else None
    train = training_run(all_runs, run.tree_id)
    gits = [r.git() for r in {r.id: r for r in (run, train) if r}.values()]
    return Facts(("eval", run.id, e.split), run, e.split, metrics, _sha(run, e.test_path), _sha(run, e.train_path),
                 _train_sha(run, all_runs), _preprocessing(run, metrics, all_runs) if metrics else None,
                 tuple((g.get("commit"), g.get("dirty")) if isinstance(g, dict) else None for g in gits), beam(e))


@dataclass(frozen=True)
class Compat:
    """A challenger against the baseline. `reason`: why it is CAUTION / INCOMPATIBLE (what is missing or differs);
    `matches`: what is known to match; `affects`: which deltas are withheld because of it."""
    level: str
    reason: str = ""
    matches: str = ""
    affects: str = ""
    population: bool = False  # in-vocabulary populations equal: subset deltas allowed
    strings: bool = False     # seen/unseen breakdown inputs equal: Generalization deltas allowed
    notes: tuple = ()         # known differences that withhold deltas (no uncertainty, so not CAUTION)


BASELINE = Compat(VERIFIED, population=True, strings=True)


def _incompatible(reason):
    return Compat(INCOMPATIBLE, reason=reason, affects="cannot be compared with the baseline")


def assess(base, other):
    """How `other` compares with `base` (both Facts). Only the evaluated file (sha256) and its loaded row count decide
    eligibility; vocabularies, beams and search settings never do. Deltas of in-vocabulary metrics need a population
    hash match or its derivation (tui.md §11)."""
    if other.key == base.key:
        return BASELINE
    if not other.metrics:
        return _incompatible(f"no metrics recorded (metrics_{other.split}.json not written)")
    if not base.metrics:
        return _incompatible("the baseline has no metrics recorded")
    unknown = [name for name, f in (("baseline", base), ("this evaluation", other)) if f.test_sha is None]
    if unknown:
        return _incompatible(f"test-file sha256 not recorded for {' and '.join(unknown)}: dataset identity unknown")
    if base.test_sha != other.test_sha:
        pa, pb = (dig(f.metrics, "test_path") or "" for f in (base, other))
        na, nb = (Path(p).name for p in (pa, pb))
        if pa == pb:
            return _incompatible(f"test file {pa} has different contents (sha256 differs)")
        return _incompatible(f"different test file ({na} vs {nb})" if na != nb else f"different test file ({pa} vs {pb})")
    rows = [f.metrics.get("rows_total") for f in (base, other)]
    if not all(number(n) for n in rows):
        return _incompatible("loaded row count (rows_total) not recorded")
    if rows[0] != rows[1]:
        return _incompatible(f"loaded rows differ ({count(rows[0])} vs {count(rows[1])}) on the same test file")
    matches = f"test file sha256, {count(rows[0])} loaded rows"

    missing, notes = [], []  # uncertain -> CAUTION; known differences -> notes
    hb, ho = base.metrics.get("population_sha256"), other.metrics.get("population_sha256")
    if hb and ho:
        population = hb == ho
        if not population:
            notes.append(f"in-vocabulary populations differ (population hash): no deltas for {SUBSET}")
    elif base.tree_train_sha and other.tree_train_sha and base.tree_train_sha != other.tree_train_sha:
        population = False
        notes.append(f"trees trained on different files, so in-vocabulary populations differ: no deltas for {SUBSET}")
    else:
        why = []
        if not (base.tree_train_sha and other.tree_train_sha):
            why.append("training run of a tree not found")
        pb, po = base.preprocessing, other.preprocessing
        if pb is None or po is None:
            why.append("preprocessing not recorded")
        elif pb != po:
            why.append("preprocessing differs (" + ", ".join(f"{k} {pb.get(k)} vs {po.get(k)}"
                                                             for k in dict.fromkeys([*pb, *po]) if pb.get(k) != po.get(k)) + ")")
        commits = {*base.commits, *other.commits}
        commit = next(iter(commits)) if len(commits) == 1 else None
        if not (commit and commit[0] and commit[1] is False):
            why.append("runs not all at one clean git commit")
        population = not why
        if why:
            missing.append("in-vocabulary population not verified: no population hash, and " + "; ".join(why))
    population_unknown = bool(missing)
    strings = False
    if "strings" in base.metrics and "strings" in other.metrics:
        if base.strings_sha and other.strings_sha:
            strings = population and base.strings_sha == other.strings_sha
            if base.strings_sha != other.strings_sha:
                notes.append("seen/unseen reference files differ: no Generalization deltas")
        else:
            missing.append("seen/unseen reference file sha256 not recorded")
    if not missing:
        return Compat(VERIFIED, matches=matches, population=population, strings=strings, notes=tuple(notes))
    affects = [f"no deltas for {SUBSET}" if population_unknown else "",
               "no Generalization deltas" if not strings else "", "acc@1_all and coverage compared"]
    return Compat(CAUTION, reason="; ".join(missing), matches=matches, affects="; ".join(a for a in affects if a),
                  population=population, strings=strings, notes=tuple(notes))


# --- analysis rows ----------------------------------------------------------------------------------------------
# A row: (label, value of a metrics dict, kind, which Compat flag allows its delta). kind: "pct" (fraction shown as %,
# delta in percentage points), "mrr" (native scale), "count", "ms", "s", "text" (no delta).
def _m(key):
    return lambda m: m.get(key)


def coverage(m):
    rows, total = m.get("rows"), m.get("rows_total")
    return rows / total if number(rows) and number(total) and total else None


def _string(group, col):
    return lambda m: dig(m, "strings", group, col)


def _search(key):
    return lambda m: dig(m, "search", key)


GROUPS = {
    "Performance": (("Acc@1_all", _m("acc@1_all"), "pct", "all"), ("Acc@1", _m("acc@1"), "pct", "population"),
                    ("Coverage", coverage, "pct", "all"), ("MRR", _m("MRR"), "mrr", "population"),
                    ("Rows evaluated", _m("rows"), "count", None), ("Rows loaded", _m("rows_total"), "count", None),
                    ("Labels", _m("labels"), "count", None), ("Distinct gold", _m("distinct_gold"), "count", None)),
    "Ranking": (*((k, _m(k), "pct", "population") for k in RECALL), ("R@cand", _m("R@cand"), "pct", "population"),
                ("Candidates/query", _m("candidates_per_query"), "count", None)),
    "Generalization": (("Hybrid", _m("hybrid"), "pct", "hybrid"), ("Hybrid_all", _m("hybrid_all"), "pct", "hybrid_all"),
                       *((f"{g} · {name}", _string(g, col), kind, flag) for g in STRING_GROUPS
                         for col, name, kind, flag in (("n", "n", "count", None),
                                                       *((c, n, "pct", "strings") for c, n in STRING_COLS)))),
    "Efficiency": (("Prediction time", _m("predict_seconds"), "s", None), ("ms/query", _m("ms_per_query"), "ms", None),
                   ("Eval time", _m("eval_seconds"), "s", None),
                   ("Candidates/query", _m("candidates_per_query"), "count", None),
                   *((k, _search(k), "text", None) for k in ("beam_size", "topk", "knn_beta", "leaf_candidates")),
                   ("Features", _m("features"), "text", None)),
}
GROUP_NOTES = {"Efficiency": "No timing deltas: equal hardware and load are not recorded.",
               "Generalization": "Seen/unseen groups are mention strings of the evaluation's train_path."}


def allowed(flag, c):
    """Whether Compat `c` allows a delta for a row with `flag`."""
    if c.level == INCOMPATIBLE or flag is None:
        return False
    return {"all": True, "population": c.population, "strings": c.strings,
            "hybrid": c.population and c.strings, "hybrid_all": c.strings}[flag]


def value_text(x, kind):
    if x is None or (kind != "text" and not number(x)):
        return NA
    return {"pct": lambda: f"{x * 100:.2f}%", "mrr": lambda: f"{x:.4f}", "count": lambda: count(x),
            "ms": lambda: f"{x:.2f}", "s": lambda: f"{x:.1f} s", "text": lambda: str(x)}[kind]()


def delta_text(x, y, kind):
    """`y - x` as shown: percentage points for "pct", the metric's scale for "mrr"; None when not a number pair."""
    if not (number(x) and number(y)) or kind not in ("pct", "mrr"):
        return None
    return f"{(y - x) * 100:+.2f}" if kind == "pct" else f"{y - x:+.4f}"


def rows(group, entries, compats):
    """[(label, [(value, delta or None) per entry])]; entries[0] is the baseline (no delta). A delta is given only
    where the row's Compat flag allows it; a missing metric is N/A, never 0."""
    out = []
    for label, get, kind, flag in GROUPS[group]:
        base = get(entries[0].metrics or {})
        cells = []
        for i, (f, c) in enumerate(zip(entries, compats)):
            x = get(f.metrics or {})
            cells.append((value_text(x, kind), delta_text(base, x, kind) if i and allowed(flag, c) else None))
        if group != "Generalization" or any(v != NA for v, _ in cells):  # old runs record no breakdown
            out.append((label, cells))
    return out


def beam(e):
    """Beam size of evaluation `e` (runs.Eval): its recorded search, else the spec's override, else None."""
    b = dig(e.metrics, "search", "beam_size")
    b = e.overrides.get("beam_size") if b is None else b
    return int(b) if number(b) and float(b).is_integer() else None


def beam_label(b):
    return f"B{b}" if b is not None else "B?"


def summary_cells(entries, compats):
    """Acc@1_all, Acc@1, Coverage of each entry: [(value, delta or None)] x 3."""
    return [[cells[i] for _, cells in rows("Performance", entries, compats)[:3]] for i in range(len(entries))]
