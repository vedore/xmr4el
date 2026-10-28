"""Run monitor (tui.md §5.11): Run overview over Pipeline | Resource monitor over Live logs, for one run (by run id).

Read-only except Cancel Run, which goes through process.stop after a confirmation. Data: the app's scan (state,
every RESCAN_SECONDS), events.jsonl (EventTail) and resources.json (reread only when changed) every POLL_SECONDS
in a worker while the run is active, the log through LogPanel. Nothing is sampled here: resources are the runner's.
"""
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Input

from xmr4el_tui import explorer, process, runs
from xmr4el_tui.progress import EventTail, stages
from xmr4el_tui.runs import state_word
from xmr4el_tui.screens.monitor_details import banner, overview_text, resource_text
from xmr4el_tui.screens.stop_dialog import StopDialog
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.log_panel import LogPanel
from xmr4el_tui.widgets.log_view import LogView
from xmr4el_tui.widgets.monitor_panels import ResourcePanel, RunOverview
from xmr4el_tui.widgets.pipeline import Pipeline
from xmr4el_tui.widgets.resize import column_width
from xmr4el_tui.widgets.top_bar import TopBar

POLL_SECONDS = 1
LEFT_SHARE, LEFT_MIN, LEFT_MAX_SHARE, RIGHT_MIN, STEP = 0.35, 30, 0.6, 44, 2
SINGLE_HEIGHT = 20  # below LEFT_MIN + RIGHT_MIN columns or this many rows: one panel at a time
TITLES = {"overview": "Run overview", "pipeline": "Pipeline", "resources": "Resources", "logs": "Live logs"}


class TelemetryRead(Message):
    """events.jsonl and resources.json as read by the telemetry worker."""

    def __init__(self, events, resources, device, final):
        super().__init__()
        self.events, self.resources, self.device, self.final = events, resources, device, final


class StopSent(Message):
    def __init__(self, result):
        super().__init__()
        self.result = result


class RunMonitorScreen(Screen):
    """Hidden keys (in the ? panel): ctrl+tab / ctrl+shift+tab (or ] / [) cycle panels, < / > resize the columns."""
    ALLOW_IN_MAXIMIZED_VIEW = "TopBar, ContextFooter"
    crumbs = ("Run Monitor",)
    BINDINGS = [Binding("escape", "back", "dashboard"),
                Binding("s", "cancel", "cancel run"),
                Binding("c", "follow", "follow stage"),
                Binding("f", "fullscreen", "fullscreen"),
                Binding("ctrl+tab,right_square_bracket", "panel(1)", "next panel", show=False),
                Binding("ctrl+shift+tab,left_square_bracket", "panel(-1)", "previous panel", show=False),
                Binding("less_than_sign", f"resize({-STEP})", "narrower left", show=False),
                Binding("greater_than_sign", f"resize({STEP})", "wider left", show=False)]

    def __init__(self, run_id, split=None):
        """`run_id` must be in the app's snapshot; `split`: select that evaluation's stage first."""
        super().__init__()
        self.run = next(r for r in self.app.snapshot.runs if r.id == run_id)
        self.initial = (f"eval:{split}",) if split else None
        # worker-only state
        self.tail, self.resources_sig, self.provenance = EventTail(self.run.path / "events.jsonl"), None, None
        self.events, self.resources, self.device, self.stages = [], None, None, []
        self.reading, self.timer, self.cancelling, self.left_want, self.single = False, None, False, None, False
        self.again = False  # a tick came while reading: read once more when that read lands

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Horizontal(id="mon"):
            with Vertical(id="mon-left"):
                yield RunOverview(id="overview")
                yield Pipeline(id="pipeline")
            with Vertical(id="mon-right"):
                yield ResourcePanel(id="resources")
                yield LogPanel(self.run, self.app.root, named=False, id="logs")
        yield ContextFooter()

    def on_mount(self):
        for wid, title in TITLES.items():
            self.query_one(f"#{wid}").border_title = title
        self.layout_columns()
        self.show_overview()
        self.query_one(Pipeline).fill(self.run, [])
        self.focus_panel(self.query_one(Pipeline))
        self.tick()
        if self.active():
            self.timer = self.set_interval(POLL_SECONDS, self.tick)
        self.set_interval(1, self.clock)

    def active(self):
        return state_word(self.run) in explorer.ACTIVE

    def stoppable(self):
        return state_word(self.run) == "RUNNING" and self.run.live.stoppable and not self.cancelling

    # --- telemetry ------------------------------------------------------------------------------------------
    def tick(self):
        """One read at a time (the tails are not shared between threads); a tick during a read is not dropped."""
        if self.reading:
            self.again = True
        else:
            self.reading = True
            self.read_telemetry(not self.active())

    @work(thread=True, group="telemetry")
    def read_telemetry(self, final):
        """`final`: started after the run ended (its runner wrote the final status after every event)."""
        self.tail.read()
        path = self.run.path / "resources.json"
        try:
            st = path.stat()
            sig = (st.st_ino, st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
        resources = self.resources if sig == self.resources_sig else runs.read_dict(path)
        self.resources_sig = sig
        if self.provenance is None:  # read once
            self.provenance = self.run.provenance() or {}
        device = runs.text(self.provenance, "torch_device") or runs.text(resources, "device")
        self.post_message(TelemetryRead(self.tail.events, resources, device, final))

    def on_telemetry_read(self, message):
        self.reading = False
        changed = (message.events, message.resources, message.device) != (self.events, self.resources, self.device)
        self.events, self.resources, self.device = message.events, message.resources, message.device
        if changed:
            self.refresh_stages()
            self.show_overview()
        if self.again:
            self.again = False
            self.tick()
        elif message.final and self.timer:  # this read started after the run ended: stop polling
            self.timer.stop()
            self.timer = None

    def refresh_stages(self):
        recorded = "failed" if state_word(self.run) == "LOST" else self.run.recorded  # as the Dashboard's stage
        self.stages = stages(self.events, self.run.kind, [e.split for e in self.run.evals], recorded)
        pipeline = self.query_one(Pipeline)
        pipeline.fill(self.run, self.stages, self.initial if self.initial and self.stages else None)
        if self.stages:
            self.initial = None
        self.show_resources()

    def show_resources(self):
        key = self.query_one(Pipeline).selected_key()
        self.query_one(ResourcePanel).show(*resource_text(key, self.run, self.stages, self.resources))

    def show_overview(self):
        self.query_one(RunOverview).show(overview_text(self.run, self.device, self.cancelling), banner(self.run),
                                         self.stoppable())

    def clock(self):
        if self.active():
            self.show_overview()
            self.query_one(Pipeline).fill(self.run, self.stages)  # the Whole run row's elapsed time

    def on_runs_changed(self, _):
        """The monitored run as the scan sees it. A vanished run directory keeps the last state shown."""
        run = next((r for r in self.app.snapshot.runs if r.id == self.run.id), None)
        if run is None or run == self.run:
            return
        ended = self.active() and state_word(run) not in explorer.ACTIVE
        self.run = run
        self.query_one(LogPanel).set_run(run)
        dialog = self.app.screen
        if ended and isinstance(dialog, StopDialog) and dialog.run.id == run.id:
            dialog.dismiss(False)
            self.app.say(f"{run.name} ended ({state_word(run)}) before it was cancelled: nothing was sent.")
        self.refresh_stages()
        self.show_overview()
        self.tick()  # ended: one more read for the final events, then polling stops
        self.refresh_bindings()

    def on_tree_node_highlighted(self, _):
        self.show_resources()

    # --- cancel -----------------------------------------------------------------------------------------------
    def action_cancel(self):
        if self.stoppable():
            self.app.push_screen(StopDialog(self.run), self.confirmed)

    def on_button_pressed(self, event):
        if event.button.id == "cancel":
            self.action_cancel()

    def confirmed(self, ok):
        if not ok:
            return
        if not self.stoppable():  # ended or already being cancelled while the dialog was open
            self.app.say(f"{self.run.name} is {state_word(self.run)}: nothing was sent.")
            return
        self.cancelling = True
        self.show_overview()
        self.refresh_bindings()
        self.send_stop(self.run)

    @work(thread=True, group="stop")
    def send_stop(self, run):
        self.post_message(StopSent(process.stop(run)))

    def on_stop_sent(self, message):
        if message.result.startswith("not signalled"):
            self.cancelling = False
            self.app.say(f"{self.run.name}: {message.result}", "warning", self)
        else:
            self.app.say(f"{self.run.name}: {message.result}; it records STOPPED when it exits.", screen=self)
        self.show_overview()
        self.refresh_bindings()
        self.app.rescan()

    # --- layout and keys --------------------------------------------------------------------------------------
    def panels(self):
        return [self.query_one(w) for w in ("#overview", "#pipeline", "#resources", "#logs")]

    def focus_panel(self, panel):
        target = panel.query_one(LogView) if isinstance(panel, LogPanel) else panel
        if isinstance(panel, RunOverview) and self.stoppable():
            target = panel.query_one("#cancel")
        if self.single:
            self.maximize(panel, container=False)
        target.focus()

    def current_panel(self):
        return next((p for p in self.panels() if self.focused is not None
                     and p in self.focused.ancestors_with_self), None)

    def action_panel(self, step):
        """Next / previous panel that can take focus (Live logs cannot while it shows a missing log)."""
        panels, here = self.panels(), self.current_panel()
        i = panels.index(here) if here in panels else -step
        for k in range(1, len(panels) + 1):
            panel = panels[(i + k * step) % len(panels)]
            if not isinstance(panel, LogPanel) or panel.query_one(LogView).display:
                self.focus_panel(panel)
                return

    def action_fullscreen(self):
        """The focused panel fills the screen (Live logs: the fullscreen log, same reader and search); esc restores."""
        if self.maximized is None and (panel := self.current_panel()):
            self.maximize(panel, container=False)

    def action_follow(self):
        self.query_one(Pipeline).follow()
        self.query_one(Pipeline).fill(self.run, self.stages)
        self.show_resources()

    def action_back(self):
        self.dismiss()

    def action_resize(self, step):
        self.set_left_width(column_width(self.left_want, self.size.width, LEFT_SHARE, LEFT_MIN, LEFT_MAX_SHARE,
                                         RIGHT_MIN) + step)

    def set_left_width(self, want):
        """Keep `want` cells (clamped) for this session; terminal resizes re-clamp it."""
        self.left_want = None if want is None else column_width(want, self.size.width, LEFT_SHARE, LEFT_MIN,
                                                                LEFT_MAX_SHARE, RIGHT_MIN)
        self.layout_columns()

    def layout_columns(self):
        """35 / 65 by default; on a small terminal one panel at a time (the focused one, maximized)."""
        width, height = self.size
        self.query_one("#mon-left").styles.width = column_width(self.left_want, width, LEFT_SHARE, LEFT_MIN,
                                                                 LEFT_MAX_SHARE, RIGHT_MIN)
        single = width < LEFT_MIN + RIGHT_MIN or height < SINGLE_HEIGHT
        if single != self.single:
            self.single = single
            self.ESCAPE_TO_MINIMIZE = not single  # esc goes back, not to a layout that does not fit
            if single:
                self.maximize(self.current_panel() or self.query_one(Pipeline), container=False)
            else:
                self.minimize()
        hint = "  ] next panel" if single else ""
        for wid, title in TITLES.items():
            if wid != "resources":
                self.query_one(f"#{wid}").border_title = title + hint

    def on_resize(self, _):
        self.layout_columns()

    def check_action(self, action, parameters):
        """Typing in the log's search box is text; Cancel Run shows only while it can be used."""
        if isinstance(self.focused, Input) and action in ("cancel", "follow", "fullscreen", "panel", "resize"):
            return False
        if action == "cancel":
            return self.stoppable()
        return True
