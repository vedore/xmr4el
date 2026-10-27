"""Widgets and the text blocks screens show (tui.md §4, §5.1-5.4)."""
import json
import os
import re
from datetime import datetime

from rich.highlighter import Highlighter
from rich.text import Text
from textual.widgets import Log, Static

from xmr4el_tui import fmt, runs
from xmr4el_tui.compare import RECALL
from xmr4el_tui.progress import event_time, read_events, stages

STATE_STYLE = {"RUNNING": "bold blue", "UNVERIFIED": "yellow", "LOST": "red", "COMPLETED": "green",
               "FAILED": "red", "STOPPED": "yellow", "UNKNOWN": ""}
STATE_SHORT = {"RUNNING": "RUN", "UNVERIFIED": "UNVER", "LOST": "LOST", "COMPLETED": "DONE", "FAILED": "FAIL",
               "STOPPED": "STOP", "UNKNOWN": "?"}  # Runs table: all columns fit 80 wide
LOG_TAIL = 256 * 1024
STRING_GROUPS = ("all", "seen, 1 label", "seen, >1 label", "unseen string")
STRING_COLS = ("tree", "dict", "either", "routed")
LOG_LINE = re.compile(r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d),\d+ \[(\w+)\] (?:xmr4el\.)?(?:[\w.]*\.)?(\w+): (.*)")
LOG_STYLES = ((r"^\d\d:\d\d:\d\d \w \w+ *", "dim"), (r"(?<=^\d\d:\d\d:\d\d )W(?= )", "bold yellow"),
              (r"(?<=^\d\d:\d\d:\d\d )[EC](?= )", "bold red"))


def bold_heads(t, *heads):
    """Bold `heads` where they start a line of `t` (section and field labels)."""
    t.highlight_regex(r"(?m)^(?:" + "|".join(map(re.escape, heads)) + r")", "bold")
    return t


def state_word(run):
    """Shown state: the live check for recorded-running runs (UNVERIFIED without one), else the recorded state."""
    if run.recorded == "running":
        return run.live.state if run.live else "UNVERIFIED"
    return run.recorded.upper()


def heartbeat_age(run, now=None):
    """Seconds since the runner last touched `heartbeat`, None without one."""
    try:
        return (now or datetime.now()).timestamp() - (run.path / "heartbeat").stat().st_mtime
    except OSError:
        return None


def state_text(run, short=False):
    word = state_word(run)
    return Text(STATE_SHORT[word] if short else word, style=STATE_STYLE[word])


def log_line(line):
    """`15:04:44 I tree           Routing started…` from the logging format; other lines unchanged."""
    m = LOG_LINE.fullmatch(line)
    return f"{m[1]} {m[2][0]} {m[3]:<14} {m[4]}" if m else line


def elapsed(run, now=None):
    end = run.finished or (now or datetime.now() if run.recorded == "running" else None)
    return (end - run.started).total_seconds() if end and run.started else None


def first_eval(run):
    return run.evals[0] if run.evals else None


def acc1(run):
    e = first_eval(run)
    return e.metrics.get("acc@1") if e and e.metrics else None


def split_cell(run):
    if not run.evals:
        return fmt.DASH
    return run.evals[0].split + (f" +{len(run.evals) - 1}" if len(run.evals) > 1 else "")


def tree_place(run):
    return "on this machine" if run.tree_here else "not on this machine"


def stage_list(run):
    """Stages from events.jsonl, or None when the run has no events file (written before the runner change)."""
    path = run.path / "events.jsonl"
    if not path.exists():
        return None
    events, _ = read_events(path)
    recorded = "failed" if state_word(run) == "LOST" else run.recorded  # a lost run's current stage failed
    return stages(events, run.kind, [e.split for e in run.evals], recorded)


class TopBar(Static):
    """`XMR4EL  <place>` left, state · host right (§4)."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.left, self.right = "", Text()

    def set(self, left, right):
        self.left, self.right = left, right if isinstance(right, Text) else Text(right)
        self.refresh()

    def render(self):
        left = Text(f"XMR4EL  {self.left}", style="bold")
        gap = max(1, self.size.width - left.cell_len - self.right.cell_len)
        return Text.assemble(left, " " * gap, self.right)


def details_text(run, all_runs):
    """The selected-run panel beside the Runs table (§5.1): one field per line."""
    word = state_word(run)
    running = run.recorded == "running"
    e = first_eval(run)
    m = e.metrics if e else None
    t = Text()
    t.append(f"{run.name}\n", style="bold")
    t.append(f"{run.kind} · ")
    t.append(word, style=STATE_STYLE[word])
    t.append("\n")
    fields = [("Started", fmt.stamp(run.started, long=True)),
              ("Duration", fmt.duration(elapsed(run)) + (" so far" if running else ""))]
    if m and m.get("features"):
        fields.append(("Features", m["features"]))
    if run.tree_id:
        fields.append(("Tree", fmt.tree(run.tree_id)))
        if m and "labels" in m:
            fields.append(("Labels", fmt.count(m["labels"])))
        fields.append(("Tree file", tree_place(run)))
        if also := runs.same_tree(all_runs, run):
            fields.append(("Also", ", ".join(r.name for r in also)))
    if running:
        fields.append(("Stage", "not recorded" if stage_list(run) is None else "see run (enter)"))
        if prev := runs.previous_completed(all_runs, run):
            fields.append(("Previous", f"{prev.name} took {fmt.duration(elapsed(prev))}"))
    _fields(t, fields)
    for e in run.evals:
        t.append(f"\n\n{e.split}", style="bold")
        if e.metrics:
            em = e.metrics
            _fields(t, [("acc@1", fmt.metric(em.get("acc@1"))), ("acc@1_all", fmt.metric(em.get("acc@1_all"))),
                        ("Rows", f"{fmt.count(em.get('rows'))} / {fmt.count(em.get('rows_total'))} in vocab")])
        else:
            t.append("\nno metrics")
    if run.error:
        t.append(f"\n\n{fmt.marker('failed', fmt.use_ascii())} {run.error}", style="red")
    if run.problems:
        t.append("\n\nproblems: " + "; ".join(run.problems), style="yellow")
    return t


def _fields(t, fields):
    for k, v in fields:
        t.append(f"\n{k:<10}", style="dim")
        t.append(f"{v}")


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
        return Text(f"status.json is missing or unreadable: {reason.split(': ', 1)[1]}. "
                    "Showing the files that exist.")
    return None


def _stages_text(run):
    found = stage_list(run)
    if found is None:
        return Text(f"Stages not recorded (run predates events.jsonl) · total {fmt.duration(elapsed(run))}")
    if not found:
        return Text(f"{fmt.marker('pending', fmt.use_ascii())} Starting…")
    if run.recorded == "completed":
        return Text("Took   " + " · ".join(f"{s.key.replace(':', ' ')} {fmt.duration(s.seconds)}" for s in found))
    ascii = fmt.use_ascii()
    lines = []
    for s in found:
        lines.append(f"{fmt.marker(s.state, ascii)} {s.key.replace(':', ' ')}  {s.detail}".rstrip()
                     + (f"  {fmt.duration(s.seconds)}" if s.seconds is not None else ""))
        lines += [f"  {fmt.marker(p.state, ascii)} {p.key}  {p.detail}".rstrip() for p in s.parts]
    return Text("\n".join(lines))


def metrics_text(m, split):
    """Evaluation block of §5.3 for one metrics dict."""
    if not m:
        return Text(f"{split}: no metrics (not produced yet)")
    search = m.get("search") or {}
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
    strings = m.get("strings")
    if strings:
        lines.append(f"{'Mention strings':<16} {'n':>6}" + "".join(f"{k:>10}" for k in STRING_COLS))
        for g in STRING_GROUPS:
            s = strings.get(g, {})
            lines.append(f"{g:<16} {fmt.count(s.get('n')):>6}" + "".join(
                f"{fmt.metric(s.get(k)):>10}" for k in STRING_COLS))
    else:
        lines.append("Mention strings  not recorded (no train_path in this evaluation)")
    t = bold_heads(Text("\n".join(lines)), "Search", "Rows", "Mention strings")
    t.highlight_regex(r"(?m)^acc@1 +[\d.]+", "bold")
    return t


def overview_text(run, all_runs, split):
    """Overview tab (§5.2-5.4) for the selected evaluation `split`."""
    t = Text()
    banner = banner_text(run)
    if banner:
        t.append_text(banner)
        t.append("\n\n")
    if run.tree_id:
        also = runs.same_tree(all_runs, run)
        t.append(f"Tree   {fmt.tree(run.tree_id)} · {'trained here' if run.kind == 'train' else 'evaluated'}"
                 f" · {tree_place(run)}" + (" · also: " + ", ".join(r.name for r in also) if also else "") + "\n")
    else:
        t.append(f"Tree   {fmt.DASH} (no tree saved yet)\n")
    t.append_text(_stages_text(run))
    t.append("\n\n")
    e = run.eval(split)
    if e:
        t.append(f"Evaluation  {e.split}  {e.test_path or ''}\n")
        t.append_text(metrics_text(e.metrics, e.split))
    else:
        t.append("No evaluations in spec.json")
    if run.problems:
        t.append("\n\nproblems: " + "; ".join(run.problems), style="yellow")
    return bold_heads(t, "Tree", "Took", "Evaluation")


def inputs_text(run):
    """Inputs tab: what was run, with the equivalent command (§10 round trip)."""
    lines = [f"Run dir      {run.path}", f"Run id       {run.id}",
             f"Tree id      {run.tree_id or fmt.DASH}", f"Tree path    {run.tree_path or fmt.DASH} ({tree_place(run)})"]
    spec = run.spec()
    if isinstance(spec, dict):
        normalized, xmodel_path = runs.normalize_spec(spec)
        lines += ["", "Command", runs.command_line(normalized, xmodel_path)]
    prov = run.provenance() or {}
    if prov:
        lines += ["", "Inputs (sha256)"] + [f"  {sha[:12]}  {p}" for p, sha in (prov.get("inputs") or {}).items()]
        lines.append(f"Python       {prov.get('python')} · {prov.get('platform')} · device {prov.get('torch_device')}")
        lines.append("Packages     " + " · ".join(f"{k} {v}" for k, v in (prov.get("packages") or {}).items()))
    git = run.git() or {}
    if git:
        lines.append(f"Git          {git.get('commit')} · {'dirty' if git.get('dirty') else 'clean'}"
                     + (f" · untracked: {', '.join(git['untracked'])}" if git.get("untracked") else ""))
    cfg = run.model_config()
    if cfg is not None:
        lines += ["", "Model config", json.dumps(cfg, indent=2)]
    return bold_heads(Text("\n".join(lines)), "Run dir", "Run id", "Tree id", "Tree path", "Command",
                      "Inputs (sha256)", "Python", "Packages", "Git", "Model config")


class LogHighlighter(Highlighter):
    """Dim time, level and logger; colour WARNING / ERROR."""

    def highlight(self, text):
        for pattern, style in LOG_STYLES:
            text.highlight_regex(pattern, style)


class LogView(Log):
    """Bounded tail of run.log (last 256 KB), polled every second; `warnings` keeps WARNING/ERROR lines only."""

    def __init__(self, path, **kwargs):
        super().__init__(highlight=True, **kwargs)
        self.highlighter = LogHighlighter()
        self.path, self.read_pos, self.lines_seen, self.warnings = path, None, [], False

    def on_mount(self):
        self.poll()
        self.set_interval(1, self.poll)

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            if self.read_pos is None:
                self.write_line(f"{self.path.name} not found")
                self.read_pos = 0
            return
        if self.read_pos is None or size < self.read_pos:  # first read or truncated: tail
            self.read_pos, skip = max(0, size - LOG_TAIL), size > LOG_TAIL
            self.lines_seen = []
            self.clear()
        else:
            skip = False
        if size == self.read_pos:
            return
        with open(self.path, "rb") as f:
            f.seek(self.read_pos)
            data = f.read(size - self.read_pos)
        end = data.rfind(b"\n") + 1
        lines = data[:end].decode("utf-8", "replace").splitlines()
        self.read_pos += end
        if skip and lines:  # the first line of a tail starts mid-line
            lines = lines[1:]
        self.lines_seen += lines
        self.write_lines([log_line(ln) for ln in lines if self.keep(ln)])

    def keep(self, line):
        return not self.warnings or "[WARNING]" in line or "[ERROR]" in line or "[CRITICAL]" in line

    def toggle_warnings(self):
        self.warnings = not self.warnings
        self.clear()
        self.write_lines([log_line(ln) for ln in self.lines_seen if self.keep(ln)])
        return self.warnings
