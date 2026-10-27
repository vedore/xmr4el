"""Description texts of the Dashboard (tui.md §5.3, §5.4): one Rich Text per explorer item."""
import re
from datetime import datetime
from pathlib import Path

from rich.text import Text

from xmr4el_tui import explorer, fmt, runs
from xmr4el_tui.compare import RECALL
from xmr4el_tui.fmt import STATE_SHORT, STATE_STYLE
from xmr4el_tui.progress import event_time, read_events
from xmr4el_tui.runs import as_dict, elapsed, heartbeat_age, state_word
from xmr4el_tui.widgets.description import HEAD

STRING_GROUPS = ("all", "seen, 1 label", "seen, >1 label", "unseen string")
STRING_COLS = ("tree", "dict", "either", "routed")
KIND = {"train": "Training", "eval": "Evaluation", "unknown": "unknown"}
DESCRIPTION_TITLE = {"tree": "Trained tree", "eval": "Evaluation", "dataset": "Dataset", "group": "Group"}


def bold_heads(t, *heads):
    """Bold `heads` where they start a line of `t` (section and field labels)."""
    t.highlight_regex(r"(?m)^(?:" + "|".join(map(re.escape, heads)) + r")", "bold")
    return t


def banner_text(run):
    """§5.4 banner for runs that did not complete, or None."""
    word = state_word(run)
    if word == "FAILED":
        return Text.assemble((f"{fmt.marker('failed', fmt.use_ascii())} {run.error or 'no error recorded'}", "red"),
                             f"\n  recorded in status.json at {fmt.stamp(run.finished, long=True)}    2 full log")
    if word == "STOPPED":
        return Text(f"Stopped by {run.error or 'a signal'} at {fmt.stamp(run.finished, long=True)}. "
                    "Completed stages and saved files are kept.", style="yellow")
    if word == "LOST":
        events, _ = read_events(run.path / "events.jsonl")
        last = next((e for e in reversed(events) if e.get("stage") and event_time(e)), None)
        where = f" Last event: {last['stage']} at {fmt.stamp(event_time(last), long=True)}." if last else ""
        return Text(f"Process {run.identity['pid']} is gone and wrote no final status (killed, e.g. out of "
                    f"memory).{where}", style="red")
    if word == "UNVERIFIED":
        age = heartbeat_age(run)
        beat = f" Heartbeat {fmt.duration(age)} ago." if age is not None else ""
        if run.live and run.live.other_runtime:
            return Text(f"Recorded running on {run.identity.get('host') or 'another host'}, another machine or "
                        f"container.{beat}", style="yellow")
        if run.identity:
            reason = run.live.reason if run.live else "not checked"
            return Text(f"Recorded running here, but the process cannot be inspected ({reason}). "
                        f"Stop unavailable.{beat}", style="yellow")
        return Text("Recorded running, but this run records no process identity (written before the runner "
                    "change), so liveness cannot be checked.", style="yellow")
    if word == "UNKNOWN":
        reason = next((p for p in run.problems if p.startswith("status.json")), "status.json: missing")
        return Text(f"status.json is missing, unreadable or has no usable status: {reason.split(': ', 1)[1]}. "
                    "Showing the files that exist.")
    return None


def metrics_text(m, split):
    """Evaluation block of §5.3 for one metrics dict."""
    if not m:
        return Text(f"{split}: no metrics (not produced yet)")
    search = as_dict(m.get("search"))
    lines = ["Search      " + " · ".join(f"{k} {v}" for k, v in search.items()),
             f"Rows        {fmt.count(m.get('rows'))} / {fmt.count(m.get('rows_total'))} have a gold label in the "
             f"tree ({fmt.pct(m.get('rows'), m.get('rows_total'))}) · {fmt.count(m.get('labels'))} labels", ""]
    cells = [(k, m.get(k)) for k in ("acc@1", "acc@1_all", "MRR", *[k for k in RECALL if k in m], "R@cand")]
    if "hybrid" in m:
        cells += [("hybrid", m.get("hybrid")), ("hybrid_all", m.get("hybrid_all"))]
    row = []
    for k, v in cells:
        row.append(f"{k:<10} {fmt.metric(v):>6}")
        if len(row) == 3:
            lines.append("      ".join(row))
            row = []
    lines += ["      ".join(row)] if row else []
    lines += [f"{fmt.ms(m.get('ms_per_query'))}, batch incl. encoding", ""]
    strings = as_dict(m.get("strings"))
    if strings:
        lines.append(f"{'Mention strings':<16} {'n':>6}" + "".join(f"{k:>10}" for k in STRING_COLS))
        for g in STRING_GROUPS:
            s = as_dict(strings.get(g))
            lines.append(f"{g:<16} {fmt.count(s.get('n')):>6}" + "".join(
                f"{fmt.metric(s.get(k)):>10}" for k in STRING_COLS))
    else:
        lines.append("Mention strings  not recorded (no train_path in this evaluation)")
    t = bold_heads(Text("\n".join(lines)), "Search", "Rows", "Mention strings")
    t.highlight_regex(r"(?m)^acc@1 +[\d.]+", "bold")
    return t


def _rows(t, rows):
    for k, v in rows:
        if v is None or v == "":
            continue
        t.append(f"{k:<14}", style="dim")
        t.append_text(v if isinstance(v, Text) else Text(str(v)))
        t.append("\n")


def _config(cfg):
    """{"type": t, "kwargs": {...}} -> `t (k=v, ...)`."""
    if not isinstance(cfg, dict):
        return None if cfg is None else str(cfg)
    kwargs = cfg.get("kwargs") if isinstance(cfg.get("kwargs"), dict) else {}
    return f"{cfg.get('type')}" + (f" ({', '.join(f'{k}={v}' for k, v in kwargs.items())})" if kwargs else "")


def _search(search):
    return " · ".join(f"{k} {v}" for k, v in search.items()) if isinstance(search, dict) and search else None


def state_text(word):
    return Text(word, style=STATE_STYLE[word])


def _provenance(t, run):
    git, prov = run.git() or {}, run.provenance() or {}
    if git or prov:
        t.append("\nProvenance\n", style=HEAD)
        _rows(t, [("Git", f"{str(git.get('commit'))[:12]} · {'dirty' if git.get('dirty') else 'clean'}"
                   if git.get("commit") else None),
                  ("Python", prov.get("python")), ("Device", prov.get("torch_device")),
                  ("Packages", " · ".join(f"{k} {v}" for k, v in as_dict(prov.get("packages")).items()) or None),
                  ("Files", str(run.path / "provenance.json") if prov else None)])


def tree_text(node, root):
    tr, t = node.ref, Text()
    trained = tr.trained_by
    t.append(f"{node.label}\n", style=HEAD)
    dataset = explorer.dataset_of(tr)
    place = Text("on this machine", style="green") if tr.here else Text(
        f"missing: {tr.path}/xmodel.pkl (moved, deleted or not synced)", style="yellow")
    _rows(t, [("Tree id", tr.id or "none recorded"),
              ("Dataset", dataset or f"{explorer.UNKNOWN_DATASET} (no training run recorded)"),
              ("Saved at", tr.path), ("", place),
              ("Recorded at", f"{tr.recorded_path} (runs' tree.json; found by tree_id at Saved at)"
               if tr.recorded_path else None),
              ("Labels", fmt.count(tr.labels) if tr.labels is not None else None)])
    if trained:
        spec = trained.spec() or {}
        snapshot = trained.path / "model_config.json"
        try:
            saved = fmt.stamp(datetime.fromtimestamp((trained.path / "tree.json").stat().st_mtime), long=True)
        except OSError:
            saved = None
        _rows(t, [("Training run", f"{trained.name} · {trained.id}"), ("Run state", state_text(state_word(trained))),
                  ("Training file", trained.train_path), ("Model config", spec.get("model_config")),
                  ("Config copy", str(snapshot) if snapshot.is_file() else None), ("Saved", saved)])
    else:
        _rows(t, [("Training run", "none recorded (trained outside run_experiment, or its run is not synced)")])
    view = runs.read_dict(Path(root, tr.path, "xmodel.json"))
    if view:
        t.append("\nModel (xmodel.json)\n", style=HEAD)
        _rows(t, [("Features", view.get("features")), ("Transformer", _config(view.get("transformer_config"))),
                  ("Vectorizer", _config(view.get("vectorizer_config"))),
                  ("Reduction", _config(view.get("dimension_config"))), ("Depth", view.get("depth")),
                  ("Clustering", _config(view.get("clustering_config"))),
                  ("Matcher", _config(view.get("matcher_config"))), ("Min leaf", view.get("min_leaf_size")),
                  ("Search", _search(view.get("predict_config")))])
    t.append("\nEvaluations\n", style=HEAD)
    if not node.children:
        t.append("none recorded\n", style="dim")
    for c in node.children:
        run, e = c.ref
        t.append(f"{c.label:<28} acc@1 {fmt.metric((e.metrics or {}).get('acc@1'))}  ")
        t.append_text(Text(STATE_SHORT[c.state], style=STATE_STYLE[c.state]))
        t.append("\n")
    if trained:
        _provenance(t, trained)
    return t


def eval_text(node, tree_node):
    (run, e), t = node.ref, Text()
    t.append(f"{e.split} · {run.name}\n", style=HEAD)
    metrics = run.path / f"metrics_{e.split}.json"
    _rows(t, [("Run", f"{run.id} · {KIND[run.kind]} run"), ("State", state_text(node.state)),
              ("Tree", f"{tree_node.label} · {fmt.tree(tree_node.ref.id)}" if tree_node else None),
              ("Test file", e.test_path), ("Strings from", e.train_path),
              ("Started", fmt.stamp(run.started, long=True)), ("Finished", fmt.stamp(run.finished, long=True)
                                                              if run.finished else None),
              ("Duration", fmt.duration(elapsed(run, datetime.now()))),
              ("Metrics file", str(metrics) if e.metrics else "not written yet")])
    if (banner := banner_text(run)) is not None:
        t.append("\n")
        t.append_text(banner)
        t.append("\n")
    t.append("\n")
    if e.metrics:
        t.append_text(metrics_text(e.metrics, e.split))
        t.append("\n")
    else:
        _rows(t, [("Search", _search(e.overrides) or "the tree's saved defaults")])
    _provenance(t, run)
    return t


def run_text(run, info):
    t = Text()
    t.append(f"{run.name}\n", style=HEAD)
    spec = run.spec() or {}
    stage, device = info.get(run.id, (None, None))
    _rows(t, [("Run id", run.id), ("Type", KIND[run.kind]), ("State", state_text(state_word(run))),
              ("Started", fmt.stamp(run.started, long=True)),
              ("Elapsed", fmt.duration(elapsed(run, datetime.now()))),
              ("Stage", stage or ("not recorded" if run.recorded == "running" else None)), ("Device", device),
              ("Tree", spec.get("xmodel_path") if run.kind == "eval" else "none saved"),
              ("Training file", run.train_path), ("Model config", spec.get("model_config")),
              ("Evaluations", ", ".join(f"{e.split} ({e.test_path})" for e in run.evals) or None),
              ("Log", str(run.path / "run.log"))])
    if (banner := banner_text(run)) is not None:
        t.append("\n")
        t.append_text(banner)
        t.append("\n")
    if run.problems:
        t.append("\nproblems: " + "; ".join(run.problems) + "\n", style="yellow")
    return t


def group_text(node, snapshot, root):
    t = Text()
    t.append(f"{node.label}\n", style=HEAD)
    if node.kind == "dataset":
        _rows(t, [("Path", node.key[1] or "no training file recorded for these trees"),
                  ("Trees", len(node.children))])
    elif node.key == ("group", "models"):
        n = sum(len(d.children) for d in node.children)
        _rows(t, [("Trees", n), ("Datasets", len(node.children)), ("Runs", len(snapshot.runs))])
        if snapshot.problem:
            t.append(f"\n{snapshot.problem}\n", style="yellow")
        if not snapshot.scanned:
            t.append("\nScanning outputs/runs/ …\n")
        elif not snapshot.found:
            t.append(f"\noutputs/runs/ not found under {root}. Start in the repository or pass --root.\n")
        elif not n:
            t.append("\nNo saved trees yet: n starts a training run.\n")
    else:
        _rows(t, [("Runs", len(node.children))])
    t.append("\nenter describes the selected item · left results · right run monitor\n", style="dim")
    return t
