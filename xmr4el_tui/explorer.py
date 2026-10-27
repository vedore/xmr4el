"""Dashboard explorer model: saved trees grouped by dataset, their evaluations, and runs without a saved tree. No UI.

Identity is the key, never the label: a tree is its tree_id (path when it has none), an evaluation is (run id,
split), a run is its directory name. Labels are display names, made unique among siblings with a short suffix.
"""
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from xmr4el_tui import fmt
from xmr4el_tui.runs import state_word

ACTIVE = ("RUNNING", "UNVERIFIED")  # recorded running and not proven gone
UNKNOWN_DATASET = "Unknown dataset"


@dataclass(frozen=True)
class Node:
    key: tuple
    kind: str                 # group | dataset | tree | eval | run
    label: str
    state: str | None = None  # state word of the run (tree: its training run)
    note: str = ""            # missing, moved, eval running
    ref: object = None        # runs.Tree | (Run, Eval) | Run | None
    children: tuple = ()


def dataset_of(tree):
    """Directory of the tree's training file (from its training run's spec), or None. Shown as recorded, not
    parsed: a dataset is only as named as its path."""
    train = tree.train_path()
    return str(Path(train).parent) if train else None


def eval_state(run, e):
    """COMPLETED once its metrics file exists; before that, its run's state (a completed run without one: UNKNOWN)."""
    if e.metrics:
        return "COMPLETED"
    word = state_word(run)
    return "UNKNOWN" if word == "COMPLETED" else word


def unique(nodes, suffix):
    """Siblings with equal labels get ` · suffix(node)`."""
    seen = Counter(n.label for n in nodes)
    return tuple(replace(n, label=f"{n.label} · {suffix(n)}") if seen[n.label] > 1 else n for n in nodes)


def tree_node(t, key):
    evals = unique([Node(("eval", r.id, e.split), "eval", e.split if r.kind == "train" else f"{e.split} · {r.name}",
                         eval_state(r, e), ref=(r, e)) for r in t.used_by for e in r.evals],
                   lambda n: fmt.stamp(n.ref[0].started))
    notes = (["missing"] if not t.here else []) + (["moved"] if t.recorded_path else []) + (
        ["eval running"] if any(r.kind == "eval" and state_word(r) in ACTIVE for r in t.used_by) else [])
    return Node(key, "tree", t.trained_by.name if t.trained_by else Path(t.path).name,
                state_word(t.trained_by) if t.trained_by else None, " · ".join(notes), t, evals)


def build(all_runs, trees):
    """Top-level nodes: Models (dataset -> tree -> evaluation), then active runs without a saved tree, then the
    other runs without one (failed before tree.json, no spec)."""
    datasets, keys = {}, set()
    for t in trees:
        key = ("tree", t.id or t.path)
        key = key if key not in keys else ("tree", t.id, t.path)  # a copied tree directory: same tree_id
        keys.add(key)
        datasets.setdefault(dataset_of(t), []).append(tree_node(t, key))
    groups = [Node(("dataset", d), "dataset", d.removeprefix("datasets/") if d else UNKNOWN_DATASET,
                   children=unique(nodes, lambda n: fmt.tree(n.ref.id) if n.ref.id else n.ref.path))
              for d, nodes in sorted(datasets.items(), key=lambda kv: (kv[0] is None, (kv[0] or "").lower()))]
    out = [Node(("group", "models"), "group", "Models", children=tuple(groups))]
    loose = [r for r in all_runs if not r.tree_path]
    for gid, label, picked in (("active", "Active, no saved tree yet", [r for r in loose if state_word(r) in ACTIVE]),
                               ("no-tree", "Runs without a tree", [r for r in loose if state_word(r) not in ACTIVE])):
        if picked:
            out.append(Node(("group", gid), "group", label, children=unique(
                [Node(("run", r.id), "run", r.name, state_word(r), ref=r) for r in picked],
                lambda n: fmt.stamp(n.ref.started))))
    return tuple(out)


def index(roots):
    """{key: (node, ancestor keys root first)}."""
    out = {}

    def walk(node, path):
        out[node.key] = (node, path)
        for c in node.children:
            walk(c, (*path, node.key))
    for r in roots:
        walk(r, ())
    return out


def signature(roots):
    """What the explorer draws: unchanged signature = no rebuild."""
    return tuple((n.key, n.label, n.state, n.note, signature(n.children)) for n in roots)
