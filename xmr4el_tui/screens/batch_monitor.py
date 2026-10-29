"""Batch monitor and batch list (tui.md §5.16): a running or finished batch, read from its files.

Read-only except Stop Batch, which writes the stop request (`batch.request_stop`); the coordinator does the stopping.
Nothing here owns a process: closing the TUI leaves the batch running. Left: batch overview + execution queue; right:
the selected tree's state and, once it has a run, that run's log (the existing LogPanel). `o` / enter opens the
tree's run in the ordinary Run Monitor (pipeline, resources, cancel).
"""
from datetime import datetime

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Static
from textual.widgets.option_list import Option

from xmr4el_tui import batch, fmt, theme
from xmr4el_tui.screens.monitor_screen import RunMonitorScreen
from xmr4el_tui.screens.stop_dialog import StopBatchDialog
from xmr4el_tui.services.experiment_repository import Activity
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.description import Rows
from xmr4el_tui.widgets.log_panel import LogPanel
from xmr4el_tui.widgets.top_bar import TopBar

POLL_SECONDS = 1
STYLE = {"pending": "text-muted", "running": "accent", "succeeded": "success", "failed": "error",
         "cancelled": "warning", "interrupted": "warning", "preparing": "text-muted", "stopping": "warning",
         "completed": "success", "completed_with_failures": "warning", "stopped": "warning", "blocked": "error",
         "lost": "error", "unreadable": "error"}


def word(state, tree=False):
    return (batch.TREE_WORDS if tree else batch.WORDS).get(state, str(state).upper())


def duration(info):
    st = info.status
    try:
        start = datetime.fromisoformat(st["started"] or st["created"])
        end = datetime.fromisoformat(st["finished"]) if st.get("finished") else datetime.now()
    except (KeyError, TypeError, ValueError):
        return fmt.DASH
    return fmt.duration((end - start).total_seconds())


def overview_text(info):
    st, trees = info.status, info.status.get("trees", [])
    done = sum(t["state"] in batch.TREE_WORDS and t["state"] not in ("pending", "running") for t in trees)
    current = info.current
    index = {t["id"]: i + 1 for i, t in enumerate(trees)}
    muted = theme.style("text-muted")
    lines = [Text.assemble(("Batch    ", muted), (info.name, "bold")),
             Text.assemble(("Status   ", muted), (word(info.state), theme.style(STYLE.get(info.state, "text"), "bold"))),
             Text.assemble(("Trees    ", muted), f"{len(trees)}   ",
                           ("Succeeded ", muted), f"{info.count('succeeded')}  ", ("Failed ", muted),
                           f"{info.count('failed')}  ", ("Pending ", muted), f"{info.count('pending')}  ",
                           ("Cancelled ", muted), f"{info.count('cancelled')}"),
             Text.assemble(("Progress ", muted), f"{done} of {len(trees)} finished"),
             Text.assemble(("Current  ", muted), f"{index[current['id']]:02d} {current['name']}" if current
                           else fmt.DASH),
             Text.assemble(("Elapsed  ", muted), duration(info))]
    if st.get("error"):
        lines.append(Text(st["error"], style=theme.style("error")))
    if info.state == "stopping":
        lines.append(Text("Stop requested: waiting for the running tree to exit.", style=theme.style("warning")))
    return Text("\n").join(lines)


def tree_text(tree, run, activity, position):
    muted = theme.style("text-muted")
    lines = [Text.assemble(("Tree     ", muted), (f"{position:02d} {tree['name']}", "bold")),
             Text.assemble(("State    ", muted), (word(tree["state"], True), theme.style(STYLE[tree["state"]], "bold"))),
             Text.assemble(("Run      ", muted), tree["run_id"] or "not started"),
             Text.assemble(("Started  ", muted), tree["started"] or fmt.DASH, ("   Finished ", muted),
                           tree["finished"] or fmt.DASH)]
    if activity and activity.stage:
        lines.append(Text.assemble(("Stage    ", muted), activity.stage))
    if tree["exit_code"] is not None:
        lines.append(Text.assemble(("Exit     ", muted), str(tree["exit_code"])))
    if tree["error"]:
        lines.append(Text(fmt.clean(str(tree["error"])), style=theme.style("error")))
    if run is None and tree["run_id"]:
        lines.append(Text("The run directory has not been scanned yet.", style=muted))
    return Text("\n").join(lines)


class BatchRead(Message):
    def __init__(self, info):
        super().__init__()
        self.info = info


class Queue(Rows):
    BINDINGS = [Binding("enter", "screen.open_run", "run monitor")]


class BatchMonitorScreen(Screen):
    crumbs = ("Batch Monitor",)
    BINDINGS = [Binding("escape", "back", "dashboard"), Binding("s", "stop", "stop batch"),
                Binding("o", "open_run", "run monitor", show=False)]

    def __init__(self, batch_id, info=None):
        super().__init__()
        self.batch_id = batch_id
        self.info = info or batch.info(batch.directory(self.app.root, batch_id))
        self.selected = None      # tree id under the queue cursor
        self.shown = None         # (tree id, run id) the log panel follows
        self.reading = False

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="bm"):
            with Vertical(id="bm-left"):
                yield Static(id="bm-overview")
                yield Queue(id="bm-queue")
                yield Button("Stop Batch", id="bm-stop", variant="error", compact=True)
            with Vertical(id="bm-right"):
                yield Static(id="bm-detail")
                yield Vertical(id="bm-logs")
        yield ContextFooter()

    def on_mount(self):
        for wid, title in (("bm-overview", "Batch overview"), ("bm-queue", "Execution queue"),
                           ("bm-detail", "Selected tree"), ("bm-logs", "Live logs")):
            self.query_one(f"#{wid}").border_title = title
        trees = self.info.status.get("trees", [])
        self.selected = (self.info.current or next(iter(trees), {})).get("id")
        self.fill()
        self.query_one(Queue).focus()
        self.set_interval(POLL_SECONDS, self.tick)

    # --- reading -----------------------------------------------------------------------------------------------
    def tick(self):
        if not self.reading:
            self.reading = True
            self.read()

    @work(thread=True, group="batch-read")
    def read(self):
        self.post_message(BatchRead(batch.info(batch.directory(self.app.root, self.batch_id))))

    def on_batch_read(self, message):
        self.reading = False
        if message.info != self.info:
            self.info = message.info
            self.fill()

    def on_runs_changed(self, _):
        self.fill_detail()

    # --- drawing -----------------------------------------------------------------------------------------------
    def tree(self):
        return next((t for t in self.info.status.get("trees", ()) if t["id"] == self.selected), None)

    def run_of(self, tree):
        return next((r for r in self.app.snapshot.runs if tree and r.id == tree["run_id"]), None)

    def fill(self):
        self.query_one("#bm-overview", Static).update(overview_text(self.info))
        queue = self.query_one(Queue)
        trees = self.info.status.get("trees", [])
        scroll = queue.scroll_y
        rows = [Option(Text.assemble(f" {i + 1:02d}  {t['name']}\n      ",
                                     (word(t["state"], True), theme.style(STYLE[t["state"]])) ), id=t["id"])
                for i, t in enumerate(trees)]
        queue.set_options(rows)
        if self.selected in [t["id"] for t in trees]:
            queue.highlighted = [t["id"] for t in trees].index(self.selected)
        self.call_after_refresh(queue.scroll_to, y=scroll, animate=False)
        self.query_one("#bm-stop", Button).disabled = self.info.state not in ("running", "preparing")
        self.fill_detail()
        self.refresh_bindings()

    def fill_detail(self):
        tree = self.tree()
        if tree is None:
            self.query_one("#bm-detail", Static).update("No tree.")
            return
        position = [t["id"] for t in self.info.status["trees"]].index(tree["id"]) + 1
        run = self.run_of(tree)
        activity = self.app.snapshot.active.get(run.id, Activity()) if run else None
        self.query_one("#bm-detail", Static).update(tree_text(tree, run, activity, position))
        key = (tree["id"], run.id if run else None)
        if tree["state"] == "running" and tree["run_id"] and run is None:
            self.app.rescan()  # the scan that knows this run lands in a few seconds
        if key != self.shown:
            self.shown = key
            box = self.query_one("#bm-logs")
            box.remove_children()
            if run is not None:
                box.mount(LogPanel(run, self.app.root, named=False))
            else:
                box.mount(Static(Text("Nothing to show: this tree has not started." if not tree["run_id"] else
                                      "Waiting for the run to appear.", style=theme.style("text-muted"))))
        elif run is not None:
            for panel in self.query_one("#bm-logs").query(LogPanel):
                panel.set_run(run)

    def on_option_list_option_highlighted(self, event):
        if event.option and event.option.id != self.selected:
            self.selected = event.option.id
            self.fill_detail()

    # --- actions -----------------------------------------------------------------------------------------------
    def action_open_run(self):
        tree = self.tree()
        if tree is None or self.run_of(tree) is None:
            self.app.say("This tree has no run to open yet.", screen=self)
        else:
            self.app.push_screen(RunMonitorScreen(tree["run_id"]))

    def check_action(self, action, parameters):
        return action != "stop" or self.info.state in ("running", "preparing")

    def action_stop(self):
        if self.info.state in ("running", "preparing"):
            self.app.push_screen(StopBatchDialog(self.info), self.confirmed)

    def on_button_pressed(self, event):
        if event.button.id == "bm-stop":
            self.action_stop()

    def confirmed(self, ok):
        if ok:
            self.send_stop()

    @work(thread=True, group="batch-stop")
    def send_stop(self):
        self.post_message(StopSent(batch.request_stop(self.app.root, self.batch_id)))

    def on_stop_sent(self, message):
        severity = "warning" if message.result.startswith("not requested") else "information"
        self.app.say(f"{self.info.name}: {message.result}.", severity, self)
        self.tick()

    def action_back(self):
        self.app.go_dashboard()


class StopSent(Message):
    def __init__(self, result):
        super().__init__()
        self.result = result


class BatchesScreen(Screen):
    """Every batch in outputs/batches/, newest first; enter opens its monitor."""
    crumbs = ("Batches",)
    BINDINGS = [Binding("escape", "back", "dashboard")]

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        yield Rows(id="bm-list")
        yield ContextFooter()

    def on_mount(self):
        self.query_one("#bm-list").border_title = "Batches"
        self.fill()
        self.query_one(Rows).focus()

    def on_runs_changed(self, _):
        self.fill()

    def fill(self):
        box = self.query_one(Rows)
        keep = box.highlighted_option.id if box.highlighted is not None else None
        found = self.app.snapshot.batches
        box.set_options([Option(Text.assemble(
            f" {i.status.get('created', '')[:16].replace('T', ' ')}  ", (i.name, "bold"), "  ",
            (word(i.state), theme.style(STYLE.get(i.state, "text"))),
            f"  {len(i.status.get('trees', ()))} trees, {i.count('succeeded')} ok, {i.count('failed')} failed"),
            id=i.id) for i in found] or [Option(Text(" No batches yet.", style=theme.style("text-muted")),
                                                disabled=True)])
        ids = [i.id for i in found]
        if keep in ids:
            box.highlighted = ids.index(keep)
        elif ids:
            box.highlighted = 0

    def on_option_list_option_selected(self, event):
        info = next((i for i in self.app.snapshot.batches if i.id == event.option.id), None)
        if info is not None:
            self.app.push_screen(BatchMonitorScreen(info.id, info))

    def action_back(self):
        self.dismiss()
