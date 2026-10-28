"""Texts of the Run monitor (tui.md §5.11): overview, stage rows, resource summaries. No file reads: the data comes
from the scan's Run, the stage list (progress.stages) and resources.json as the runner wrote it."""
from datetime import datetime
from pathlib import Path

from rich.text import Text

from xmr4el_tui import fmt
from xmr4el_tui.fmt import STATE_STYLE
from xmr4el_tui.runs import as_dict, elapsed, state_word
from xmr4el_tui.screens.dashboard_details import banner_text
from xmr4el_tui.services.experiment_repository import Activity
from xmr4el_tui.theme import style

NA = "N/A"
RUN_ROW = ("run",)  # the Pipeline row of the whole run
STAGE_NAMES = {"load": "Load data", "encode": "Encode features", "labels": "Label embeddings",
               "hierarchy": "Hierarchy training", "save": "Save tree"}
PART_NAMES = {"mention_tfidf": "mention TF-IDF → SVD", "context_tfidf": "context TF-IDF → SVD",
              "transformer": "transformer", "assemble": "assemble features", "data": "load data",
              "predict": "prediction", "metrics": "metrics", "write": "write results"}
STAGE_STYLE = {"done": style("success"), "current": style("info", "bold"), "pending": "dim",
               "failed": style("error"), "stopped": style("warning")}
STAGE_WORD = {"done": "completed", "current": "running", "pending": "pending", "failed": "failed",
              "stopped": "stopped"}


def kind_text(run):
    if run.kind == "train":
        return "Training + Evaluation" if run.evals else "Training"
    return {"eval": "Evaluation"}.get(run.kind, "unknown")


def dataset(run):
    path = run.train_path or next((e.test_path for e in run.evals if e.test_path), None)
    return str(Path(path).parent) if path else None


def overview_text(run, device, cancelling):
    word = state_word(run)
    state = Text(word, style=STATE_STYLE[word])
    if cancelling and word == "RUNNING":
        state.append("  cancellation requested", style=style("warning"))
    t = Text()  # one line per field: the panel cuts long ones (app.tcss #overview-text)
    t.append(f"{run.name}\n", style="bold")
    for k, v in (("Type", kind_text(run) + (f" · {', '.join(e.split for e in run.evals)}" if run.evals else "")),
                 ("Dataset", dataset(run)), ("State", state),
                 ("Elapsed", fmt.duration(elapsed(run, datetime.now()))), ("Device", device or NA),
                 ("Run id", Text(run.id, style="dim"))):
        if v is None:
            continue
        t.append(f"{k:<9}", style="dim")
        t.append_text(v if isinstance(v, Text) else Text(v))
        t.append("\n")
    t.rstrip()
    return t


def banner(run):
    """Why the run is not running (failed, stopped, lost, unverified, unknown), or None."""
    return banner_text(run, Activity()) if state_word(run) != "COMPLETED" else None


def stage_name(key, kind):
    if key.startswith("eval:"):
        return f"Evaluate {key[5:]}"
    if key == "load" and kind == "eval":
        return "Load tree"
    return STAGE_NAMES.get(key, PART_NAMES.get(key, key))


def stage_row(stage, kind, width=20):
    """`+ Encode features    0:02  344 rows`: marker, name padded to `width`, duration when ended, emitted detail."""
    marker = fmt.marker("failed" if stage.state == "stopped" else stage.state, fmt.use_ascii())
    t = Text(no_wrap=True, overflow="ellipsis")
    t.append(f"{marker} ", style=STAGE_STYLE[stage.state])
    t.append(f"{stage_name(stage.key, kind):<{width}}", style="dim" if stage.state == "pending" else "")
    t.append(f"{fmt.duration(stage.seconds) if stage.seconds is not None else '':>8}")
    if stage.state in ("failed", "stopped"):
        t.append(f"  {stage.state}", style=STAGE_STYLE[stage.state])
    if stage.detail:
        t.append(f"  {stage.detail}", style="dim")
    return t


def run_row(run):
    word = state_word(run)
    t = Text(no_wrap=True, overflow="ellipsis")
    t.append(f"{'Whole run':<22}", style="bold")
    t.append(f"{fmt.duration(elapsed(run, datetime.now())):>8}  ")
    t.append(fmt.STATE_SHORT[word], style=STATE_STYLE[word])
    return t


def stage_summaries(doc):
    """resources.json's stage summaries: the objects of its `stages` list ([] when it is not a list)."""
    found = as_dict(doc).get("stages")
    return [s for s in found if isinstance(s, dict)] if isinstance(found, list) else []


def summary_of(doc, key):
    """resources.json summary of top-level stage `key` (load, …, eval:<split>), the last one recorded for that
    (stage, split), or None."""
    stage, split = ("eval", key[5:]) if key.startswith("eval:") else (key, None)
    found = [s for s in stage_summaries(doc) if (s.get("stage"), s.get("split")) == (stage, split)]
    return found[-1] if found else None


def _pct(x):
    return f"{x:.1f} %" if fmt.number(x) else NA


def _size(x):
    return fmt.size(x) if fmt.number(x) else NA


def _rows(t, rows):
    for k, v, scope in rows:
        t.append(f"{k:<12}", style="dim")
        t.append(v, style="bold" if v != NA else "dim")
        if scope:
            t.append(f"   {scope}", style="dim")
        t.append("\n")


def resources_block(t, res, scope):
    """Scopes are resources.json's own: device-wide GPU utilisation is never called per process."""
    gpu_util, gpu_mem, util = scope.get("gpu_util"), scope.get("gpu_memory"), res.get("gpu_util_avg_percent")
    _rows(t, [("CPU avg", _pct(res.get("cpu_avg_percent")), scope.get("cpu")),
              ("RAM avg", _size(res.get("rss_avg_bytes")), scope.get("rss")),
              ("RAM peak", _size(res.get("rss_peak_bytes")), None),
              ("GPU util", _pct(util), "not measured on this device" if gpu_util is None else
               gpu_util if util is not None else f"{gpu_util}: no NVML samples (optional nvml extra)"),
              ("GPU memory", _size(res.get("torch_gpu_allocated_peak_bytes")),
               f"peak, {gpu_mem}" if gpu_mem else "not measured on this device")])


def resource_text(key, run, stages, doc):
    """(title, text) of the Resource monitor for Pipeline row `key`: RUN_ROW, (stage,) or (stage, part)."""
    doc = doc if isinstance(doc, dict) else None
    scope = {k: fmt.clean(v) for k, v in as_dict((doc or {}).get("scope")).items() if isinstance(v, str)}
    t = Text()
    if key is None or not stages and key != RUN_ROW:
        t.append("Select a stage in the pipeline to see its resources.", style="dim")
        return "Resources", t
    if key == RUN_ROW:
        return "Resources — whole run", overall_text(run, doc, scope)
    stage = next((s for s in stages if s.key == key[0]), None)
    if stage is None:
        return "Resources", Text("This stage is no longer in the run's events.", style=style("warning"))
    part = next((p for p in stage.parts if p.key == key[1]), None) if len(key) > 1 else None
    shown = part or stage
    name = stage_name(stage.key, run.kind)
    title = f"Resources — {name}" + (f" › {stage_name(part.key, run.kind)}" if part else "")
    t.append(STAGE_WORD[shown.state].capitalize(), style=STAGE_STYLE[shown.state])
    if shown.seconds is not None:
        t.append(f"  {fmt.duration(shown.seconds)}")
    if shown.error:
        t.append(f"\n{shown.error}", style=style("error"))
    t.append("\n\n")
    summary = summary_of(doc, stage.key)
    if part and summary:
        t.append(f"Parts are not sampled separately: the figures are the whole {name} stage.\n", style="dim")
    if summary:
        n, every = summary.get("samples"), (doc or {}).get("interval_seconds")
        t.append(f"{n if fmt.number(n) else NA} samples" + (f", every {every} s" if fmt.number(every) else "")
                 + " plus stage start and end\n\n", style="dim")
        resources_block(t, as_dict(summary.get("resources")), scope)
    elif stage.state == "pending":
        t.append("Not started.", style="dim")
    elif stage.state == "current":
        t.append("Collecting resource statistics…\nThe runner writes this stage's summary when it ends.", style="dim")
    elif doc is None:
        t.append("No resources.json for this run (written before resource telemetry, or -resource_interval 0).",
                 style="dim")
    else:
        t.append("No summary recorded for this stage.", style="dim")
    t.rstrip()
    return title, t


def overall_text(run, doc, scope):
    """Only what the stage summaries give exactly: total stage time, CPU time / wall time over them, the highest
    peaks. Averages of RAM and GPU utilisation are not derivable from per-stage averages."""
    t = Text()
    word = state_word(run)
    t.append(word, style=STATE_STYLE[word])
    t.append(f"  {fmt.duration(elapsed(run, datetime.now()))}\n\n")
    found = stage_summaries(doc)
    if doc is None:
        t.append("No resources.json for this run (written before resource telemetry, or -resource_interval 0).",
                 style="dim")
        return t
    if not found:
        t.append("No stage has ended yet: summaries are written per stage.", style="dim")
        return t
    res = [(as_dict(s.get("resources")), s.get("duration_seconds")) for s in found]
    timed = [(r.get("cpu_avg_percent"), d) for r, d in res if fmt.number(d) and d > 0]
    cpu = sum(c * d for c, d in timed) / sum(d for _, d in timed) \
        if timed and all(fmt.number(c) for c, _ in timed) else None
    peak = lambda k: max((r[k] for r, _ in res if fmt.number(r.get(k))), default=None)
    t.append(f"{len(found)} ended stages · {fmt.duration(sum(d for _, d in timed))} recorded\n\n", style="dim")
    _rows(t, [("CPU avg", _pct(cpu), "CPU time / wall time over the ended stages"),
              ("RAM peak", _size(peak("rss_peak_bytes")), scope.get("rss")),
              ("GPU memory", _size(peak("torch_gpu_allocated_peak_bytes")),
               f"peak, {scope['gpu_memory']}" if scope.get("gpu_memory") else "not measured on this device")])
    t.append("RAM and GPU utilisation averages are per stage: select a stage.", style="dim")
    return t
