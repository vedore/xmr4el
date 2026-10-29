"""Review Batch model (tui.md §5.15): the checks run on the experiment draft before launch, the configuration
differences between trees, and the plan Step 5's launcher reads. No UI, stdlib only.

Everything reuses the existing rules: a tree's model configuration is checked by `components.problems` (the Model
Builder's own validators), the shared dataset by `datasets.validate` (Dataset Setup's, bounded reads, cached by
modification time). Nothing here trains, hashes or downloads: input provenance (sha256) stays with the runner's
`check_inputs` at launch, so a Ready batch can still fail there.
"""
import copy
import json
import re
from dataclasses import dataclass

from xmr4el_tui import components, datasets

ERROR, CAUTION = "error", "caution"
READY_WORD, CAUTION_WORD, INVALID_WORD = "Ready", "Caution", "Invalid"
MODEL_KEYS = set(components.ORDER) | {"verbose"}  # the keys `XModel(**config)` accepts; others make the loader fail
RUN_NAME = re.compile(r"[A-Za-z0-9_.-]+")          # what the runner accepts as a split / run name
STAGE_NAMES = {"Feature construction": "Features"}
STAGES = ("Features", "Hierarchy", "Matcher", "Search", "Other")
MISSING = object()  # a key the configuration does not have (distinct from an explicit null)
DF_PATHS = {components.path_of(s, f) for s in components.SLOTS for c in s.components for f in c.fields if f.kind == "df"}


@dataclass(frozen=True)
class Issue:
    """One finding with where to fix it. `dest`: slot (a Model Builder component, `field` when known) | tree (the
    tree in the Model Builder) | rename (its rename box) | dataset (Dataset Setup, `target` = its field) |
    batch-name (the Review Batch name box)."""
    id: str
    severity: str
    scope: str                 # batch | dataset | tree
    message: str
    tree: str | None = None
    path: str | None = None    # dotted JSON path in the tree's configuration
    slot: str | None = None
    field: components.Field | None = None
    dest: str = "tree"
    target: str | None = None


# --- validation ---------------------------------------------------------------------------------------------------
def status(issues):
    return INVALID_WORD if any(i.severity == ERROR for i in issues) else CAUTION_WORD if issues else READY_WORD


def tree_issues(draft):
    """{tree id: [Issue]} for every tree of the batch: name, model configuration, duplicate configuration."""
    out = {t.id: [] for t in draft.trees}
    for t in draft.trees:
        found = out[t.id]

        def add(key, message, severity=ERROR, **kw):
            found.append(Issue(f"tree:{t.id}:{key}", severity, "tree", message, tree=t.id, **kw))

        if (problem := draft.name_problem(t.id, t.name)):
            add("name", f"Tree name: {problem}", dest="rename")
        for key in t.config:
            if key not in MODEL_KEYS:
                add(f"key:{key}", f"{key!r} is not a model setting: the loader refuses it.", path=key)
        for sid, items in components.problems(t.config).items():
            slot = components.BY_ID.get(sid)
            for f, message in items:
                if slot is None:  # the feature recipe
                    add("features", message, path="features")
                elif f is None:
                    add(f"{sid}:slot", message, path=slot.key, slot=sid, dest="slot")
                else:
                    path = components.json_path(slot, f)
                    add(f"{sid}:{path}", f"{f.label}: {message}", path=path, slot=sid, field=f, dest="slot")
    names = {t.id: t.name for t in draft.trees}
    for a in draft.trees:
        twins = [b.id for b in draft.trees if b.id != a.id and not diff(a.config, b.config)]
        if twins and not any(i.severity == ERROR for i in out[a.id]):
            same = ", ".join(names[i] for i in twins)
            out[a.id].append(Issue(f"tree:{a.id}:twin", CAUTION, "tree", f"Same configuration as {same}.",
                                   tree=a.id))
    return out


def dataset_issues(root, selection):
    """Problems of the batch's shared dataset (`selection`: the draft's confirmed one), reported once. Every
    Dataset Setup check blocks here: launch needs a training file and an evaluation split that all pass."""
    if selection is None:
        return [Issue("dataset:none", ERROR, "dataset", "No dataset confirmed.", dest="dataset")]
    out = [Issue(f"dataset:{f}:{n}", ERROR, "dataset", message, dest="dataset", target=f)
           for n, (f, _, message) in enumerate(datasets.validate(
               root, datasets.Dataset(selection.key, selection.name, selection.variant), selection))]
    for e in selection.enabled():
        if not RUN_NAME.fullmatch(e.split):
            out.append(Issue(f"dataset:name:{e.split}", ERROR, "dataset",
                             f"Evaluation split name {e.split!r}: use letters, digits, _ . -", dest="dataset",
                             target=e.split))
    return out


def batch_issues(draft):
    problem = draft.batch_name_problem(draft.name)
    if problem:
        return [Issue("batch:name", ERROR, "batch", f"Batch name: {problem}", dest="batch-name")]
    return []


def validate(root, draft):
    """Every issue of the batch (dataset, batch, trees in queue order). The launcher's entry point."""
    trees = tree_issues(draft)
    return [*batch_issues(draft), *dataset_issues(root, draft.confirmed), *(i for t in draft.trees for i in trees[t.id])]


def blocking(issues):
    return [i for i in issues if i.severity == ERROR]


# --- configuration differences ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Change:
    stage: str
    path: tuple
    label: str
    old: object
    new: object


def stage_of(path):
    """Stage of a configuration path: its slot's, else Features (the recipe) or Other."""
    for s in components.SLOTS:
        if path[0] in s.keys:
            return STAGE_NAMES.get(s.stage, s.stage)
    return "Features" if path[0] == "features" else "Other"


def effective(config):
    """{path: value} of what the config means: every field the Model Builder knows with the backend default where
    the config leaves it out (so an omitted default equals the same value written out), then every other key of the
    config as written (a missing key stays MISSING in the comparison, an explicit null stays None)."""
    out, labels = {}, {}
    for s in components.SLOTS:
        comp = components.installed(config, s)
        for f in (comp.fields if comp else ()) + s.extras:
            path = components.path_of(s, f)
            out[path], labels[path] = components.value(config, s, f), f.label
    def leaves(path, node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from leaves((*path, k), v)
        elif node != {}:
            yield path, node
    for path, v in leaves((), config):
        if path not in out:
            slot = next((s for s in components.SLOTS if path[0] in s.keys), None)
            out[path] = v
            labels[path] = f"{slot.name} component" if slot and path == (slot.key, "type") else ".".join(path)
    return out, labels


def same(a, b):
    """Equal values: numbers by value (1 == 1.0), booleans only as booleans, lists and dicts element-wise."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a is b if MISSING in (a, b) else type(a) is type(b) and a == b


def diff(old, new):
    """[Change] between two configurations, grouped by stage (then pipeline order): effective values, nested keys
    by path, never compared as text or by key order."""
    (a, la), (b, lb) = effective(old), effective(new)
    changes = [Change(stage_of(p), p, la.get(p) or lb[p], a.get(p, MISSING), b.get(p, MISSING))
               for p in dict.fromkeys([*a, *b]) if not (components.same if p in DF_PATHS else same)(
                   a.get(p, MISSING), b.get(p, MISSING))]
    return sorted(changes, key=lambda c: STAGES.index(c.stage))  # stable: pipeline order inside a stage


def show(v):
    """A configuration value as one short text: strings plain, JSON for the rest, MISSING as `not set`."""
    return "not set" if v is MISSING else v if isinstance(v, str) else json.dumps(v)


# --- the plan Step 5 launches -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannedTree:
    id: str
    name: str
    config: dict


@dataclass(frozen=True)
class Plan:
    """A reviewed batch, copied out of the draft: later edits of the draft never reach it."""
    id: str
    name: str
    dataset: datasets.Selection
    trees: tuple


def plan(draft):
    """The draft as it stands: batch identity and name, the confirmed shared dataset, the trees in execution order
    (id, name, applied configuration). Raises ValueError without a confirmed dataset or a tree; validity is
    `validate`'s job."""
    if draft.confirmed is None or not draft.trees:
        raise ValueError("a batch needs a confirmed dataset and at least one tree")
    return Plan(draft.id, draft.name, draft.confirmed,
                tuple(PlannedTree(t.id, t.name, copy.deepcopy(t.config)) for t in draft.trees))


def run_name(plan_, tree):
    """The run name of a tree: ids only (display names are not filesystem safe)."""
    return f"{plan_.id}_{tree.id}"


def run_spec(plan_, tree, model_config_path):
    """The experiment runner's spec (scripts/run_experiment.py) for one tree: the shared training file and the enabled
    evaluation splits; the model config file is the caller's to write from `tree.config`. Each evaluation's
    `train_path` is the training file the tree is trained on (the mention strings it has seen), as in
    configs/experiments/*.json; the draft has no per-evaluation override."""
    return {"name": run_name(plan_, tree), "model_config": model_config_path, "train_path": plan_.dataset.train,
            "evals": [{"split": e.split, "test_path": e.path, "train_path": plan_.dataset.train}
                      for e in plan_.dataset.enabled()]}
