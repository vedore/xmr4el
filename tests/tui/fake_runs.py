"""Run directories with the file shapes scripts/run_experiment.py writes (copied from real outputs/runs)."""
import hashlib
import json
from pathlib import Path

METRICS = {"labels": 13385, "features": "sapbert_char_context", "rows": 729, "rows_total": 757,
           "distinct_gold": 300, "search": {"beam_size": 10, "topk": 0, "knn_beta": 10.0, "leaf_candidates": 100},
           "predict_seconds": 2.2, "ms_per_query": 2.99, "acc@1": 0.9246, "MRR": 0.9459, "R@5": 0.9671,
           "R@10": 0.9739, "R@20": 0.9739, "R@50": 0.9822, "R@100": 0.9835, "R@cand": 0.9835, "acc@1_all": 0.8904,
           "candidates_per_query": 100.0, "eval_protocol": 1}
COMMIT = "b10a922a7734f5844d8e30b9b928b740923a3299"


def sha(label):
    """A real-looking sha256 standing for `label` (equal labels, equal hashes); None stays None."""
    return hashlib.sha256(label.encode()).hexdigest() if label is not None else None


def make_run(root, name, *, stamp="2026-10-20_13-36-20", kind="train", status="completed", splits=("dev",),
             metrics=None, tree_id="8528b624aaaa", tree_here=False, events=None, identity=None, corrupt=(),
             test_sha="sha-dev", train_sha="sha-train", eval_train_sha=None, git=None, device="cuda",
             model_config=None, packages=None):
    """Write outputs/runs/<stamp>_<name>/ under root and return its path. `metrics`: {split: overrides} (None =
    METRICS for every split; a split mapped to None writes no metrics file). `corrupt`: file names to garble."""
    run = Path(root, "outputs", "runs", f"{stamp}_{name}")
    run.mkdir(parents=True)
    tree_path = f"outputs/saved_trees/xmodel_{tree_id}"
    train = kind == "train"
    evals = [{"split": s, "test_path": f"datasets/x/{s}.pubtator", "train_path": "datasets/x/train.pubtator"}
             for s in splits]
    files = {
        "status.json": {"status": status, "finished": "2026-10-20T13:38:27", **(identity or {}),
                        **({"error": "ValueError: boom"} if status == "failed" else {})},
        "spec.json": {"name": name, **({"model_config": "configs/c.json", "train_path": "datasets/x/train.pubtator"}
                                       if train else {}), "evals": evals, "xmodel_path": None if train else tree_path},
        "git.json": git if git is not None else {"commit": COMMIT, "dirty": False, "untracked": []},
        "provenance.json": {"inputs": {**{e["test_path"]: sha(test_sha if e["split"] == "dev" else f"sha-{e['split']}")
                                          for e in evals},
                                       "datasets/x/train.pubtator": sha(eval_train_sha or train_sha)},
                            "python": "3.12.15", "packages": packages or {"numpy": "2.2.6", "torch": "2.14.1"},
                            "torch_device": device},
    }
    if tree_id:
        files["tree.json"] = {"xmodel_path": tree_path, "tree_id": tree_id, "trained": train}
    if train:
        files["model_config.json"] = model_config or {"features": "sapbert_char_context", "context_window": None}
    if status == "running":
        files["status.json"] = {"status": "running", "started": "2026-10-23T14:10:00", **(identity or {})}
    for s in splits:
        over = {} if metrics is None else metrics.get(s, {})
        if over is None:
            continue
        files[f"metrics_{s}.json"] = {"tree": f"xmodel_{tree_id}", "test_path": f"datasets/x/{s}.pubtator",
                                      "train_path": "datasets/x/train.pubtator", "split": s, **METRICS, **over}
    for fname, obj in files.items():
        (run / fname).write_text(json.dumps(obj, indent=2))
    for fname in corrupt:
        (run / fname).write_text("{not json")
    if tree_here and tree_id:
        Path(root, tree_path).mkdir(parents=True, exist_ok=True)
        Path(root, tree_path, "xmodel.pkl").write_bytes(b"")
    if events is not None:
        (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "run.log").write_text("2026-10-20 13:36:21 [INFO] xmr4el.run_experiment: Run started\n")
    return run


def patch_json(path, **fields):
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def bad_fields_run(root):
    """A train run whose JSON files parse but hold fields of the wrong type; one time has an offset."""
    path = make_run(root, "bad", splits=("dev",), tree_id=None)
    patch_json(path / "status.json", status=[], started="2026-10-20T13:36:20+05:00", finished=7, error={"x": 1})
    patch_json(path / "spec.json", name=3, model_config=[], evals=[{"split": "dev", "test_path": 5}])
    patch_json(path / "metrics_dev.json", **{"acc@1": "0.9", "search": [], "strings": [1], "rows": "7", "labels": []})
    patch_json(path / "provenance.json", packages=[], inputs=[], torch_device=1)
    patch_json(path / "git.json", commit=["c"])
    return path
