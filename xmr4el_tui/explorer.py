"""Dashboard explorer model: saved trees grouped by dataset, their evaluations, and runs without a saved tree. No UI.

Identity is the key, never the label: a tree is its tree_id (path when it has none), an evaluation is (run id,
split), a run is its directory name. Labels are display names, made unique among siblings with a short suffix; a
tree shows two lines (`tree_title`, then `detail`: training date and a short tree id that no other tree shares).
"""
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from xmr4el_tui import fmt
from xmr4el_tui.runs import dig, state_word

ACTIVE = ("RUNNING", "UNVERIFIED")  # recorded running and not proven gone
UNKNOWN_DATASET = "Uncategorized"
FEATURE_NAMES = {"tfidf": "TF-IDF", "sapbert_char_context": "SapBERT"}  # model_config "features"
SHORT_ID = 8


@dataclass(frozen=True)
class Node:
    key: tuple
    kind: str                 # group | dataset | tree | eval | run
    label: str
    state: str | None = None  # state word of the run (tree: its training run)
    note: str = ""            # missing, moved, eval running
    ref: object = None        # runs.Tree | (Run, Eval) | Run | None
    children: tuple = ()
    descriptor: str = ""      # tree: features and depth of its training run ("SapBERT D2")
    detail: str = ""          # tree: second line, "Oct 28 · a64127ef"
    short: str = ""           # tree: its displayed id (short_ids), the path's name without one


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


def unique(nodes, *suffixes):
    """Siblings with equal labels get ` · suffix(node)`: the first of `suffixes` that tells them apart, else the last
    (the run id: one minute can hold two runs of one spec)."""
    twins = {}
    for n in nodes:
        twins.setdefault(n.label, []).append(n)

    def suffix(n):
        same = twins[n.label]
        return next((f for f in suffixes if len({f(m) for m in same}) == len(same)), suffixes[-1])(n)
    return tuple(replace(n, label=f"{n.label} · {suffix(n)}") if len(twins[n.label]) > 1 else n for n in nodes)


def short_ids(ids):
    """{id: its first SHORT_ID characters, or more until no other id starts the same}."""
    s = sorted(set(ids))
    common = [len(os.path.commonprefix(pair)) for pair in zip(s, s[1:])]
    return {x: x[:max(SHORT_ID, 1 + max(common[i - 1] if i else 0, common[i] if i < len(common) else 0))]
            for i, x in enumerate(s)}


def descriptor(run):
    """Features and depth recorded by the tree's training run, "" when it recorded neither."""
    if run is None:
        return ""
    return " ".join(p for p in (FEATURE_NAMES.get(run.features, run.features),
                                f"D{run.depth}" if run.depth is not None else None) if p)


BATCH_RUN = re.compile(r"\d{8}-(\d{6})-[0-9a-f]{12}_([0-9a-f]+)")


def display_name(name):
    """A batch run is named `<batch id>_<tree id>` (review.run_name): shown as `<tree id> · batch <time>`, so the part
    that tells siblings apart comes first when the explorer is narrow."""
    m = BATCH_RUN.fullmatch(name)
    return f"{m.group(2)[:SHORT_ID]} · batch {m.group(1)}" if m else name


def tree_title(node):
    """First line of a tree: name, then its descriptor."""
    return f"{node.label} · {node.descriptor}" if node.descriptor else node.label


def tree_node(t, key, shorts):
    evals = unique([Node(("eval", r.id, e.split), "eval", e.split if r.kind == "train" else f"{e.split} · {r.name}",
                         eval_state(r, e), ref=(r, e)) for r in t.used_by for e in r.evals],
                   lambda n: fmt.stamp(n.ref[0].started), lambda n: n.ref[0].id)
    notes = (["missing"] if not t.here else []) + (["moved"] if t.recorded_path else []) + (
        ["eval running"] if any(r.kind == "eval" and state_word(r) in ACTIVE for r in t.used_by) else [])
    short = shorts[t.id] if t.id else Path(t.path).name
    started = t.trained_by.started if t.trained_by else None
    return Node(key, "tree", display_name(t.trained_by.name) if t.trained_by else Path(t.path).name,
                state_word(t.trained_by) if t.trained_by else None, " · ".join(notes), t, evals,
                descriptor(t.trained_by), f"{started:%b %d} · {short}" if started else short, short)


def build(all_runs, trees):
    """Top-level nodes: Models (dataset -> tree -> evaluation), then active runs without a saved tree, then the
    other runs without one (failed before tree.json, no spec)."""
    datasets, keys, shorts = {}, set(), short_ids(t.id for t in trees if t.id)
    for t in trees:
        key = ("tree", t.id or t.path)
        key = key if key not in keys else ("tree", t.id, t.path)  # a copied tree directory: same tree_id
        keys.add(key)
        datasets.setdefault(dataset_of(t), []).append(tree_node(t, key, shorts))
    groups = [Node(("dataset", d), "dataset", d.removeprefix("datasets/") if d else UNKNOWN_DATASET,
                   children=tuple(nodes))
              for d, nodes in sorted(datasets.items(), key=lambda kv: (kv[0] is None, (kv[0] or "").lower()))]
    out = [Node(("group", "models"), "group", "Models", children=tuple(groups))]
    loose = [r for r in all_runs if not r.tree_path]
    for gid, label, picked in (("active", "Active, no saved tree yet", [r for r in loose if state_word(r) in ACTIVE]),
                               ("no-tree", "Runs without a tree", [r for r in loose if state_word(r) not in ACTIVE])):
        if picked:
            out.append(Node(("group", gid), "group", label, children=unique(
                [Node(("run", r.id), "run", r.name, state_word(r), ref=r) for r in picked],
                lambda n: fmt.stamp(n.ref.started), lambda n: n.ref.id)))
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
    return tuple((n.key, n.label, n.descriptor, n.detail, n.state, n.note, signature(n.children)) for n in roots)


def prune(roots, keep):
    """(`roots` cut to the nodes `keep(node, ancestors)` accepts, each with its whole subtree, and their ancestors;
    keys of the ancestors kept only for a descendant, so a search can open them)."""
    opened = set()

    def walk(nodes, ancestors):
        out = []
        for n in nodes:
            if keep(n, ancestors):
                out.append(n)
            elif kids := walk(n.children, (*ancestors, n)):
                opened.add(n.key)
                out.append(replace(n, children=kids))
        return tuple(out)
    return walk(roots, ()), opened


def search_text(node):
    """What a search matches (lowercase): names, descriptor, date and id, dataset path, run id, split, features."""
    ref = node.ref
    parts = [node.label, node.descriptor, node.detail]
    if node.kind == "tree":
        parts += [ref.id, ref.path, dataset_of(ref), ref.trained_by and ref.trained_by.id]
    elif node.kind == "eval":
        run, e = ref
        parts += [run.id, run.name, e.split, e.test_path, dig(e.metrics, "features")]
    elif node.kind == "run":
        parts += [ref.id]
    elif node.kind == "dataset":
        parts += [node.key[1]]
    return " ".join(p for p in parts if isinstance(p, str)).lower()


def matches(query):
    """`keep` for prune: every word of `query` in the search_text of the node or of an ancestor ("ncbi dev")."""
    words = query.lower().split()
    return lambda node, ancestors: all(w in " ".join(map(search_text, (*ancestors, node))) for w in words)
