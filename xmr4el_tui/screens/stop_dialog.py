"""Cancel Run confirmation (tui.md §5.2): names the exact run; focus starts on Keep running."""
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class StopDialog(ModalScreen):
    """Dismisses with True to cancel the run."""
    BINDINGS = [Binding("escape", "dismiss(False)", "keep running")]

    def __init__(self, run):
        super().__init__()
        self.run = run

    def compose(self) -> ComposeResult:
        with Vertical(id="stop-box"):
            yield Static(Text(f"Cancel {self.run.name}?", style="bold"))
            yield Static(f"Run {self.run.id}\nSends SIGTERM to its process group (pid {self.run.identity['pid']}). "
                         "It is recorded as STOPPED. The current stage is lost; completed stages, logs, events and "
                         "saved files are kept.", id="stop-text")
            with Horizontal(id="stop-buttons"):
                yield Button("Keep running", id="keep", compact=True)
                yield Button("Cancel run", id="confirm", variant="error", compact=True)

    def on_mount(self):
        self.query_one("#keep", Button).focus()

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "confirm")


class StopBatchDialog(ModalScreen):
    """Stop Batch confirmation: dismisses with True to request the stop."""
    BINDINGS = [Binding("escape", "dismiss(False)", "keep running")]

    def __init__(self, info):
        super().__init__()
        self.info = info

    def compose(self) -> ComposeResult:
        with Vertical(id="stop-box"):
            yield Static(Text(f"Stop batch {self.info.name}?", style="bold"))
            yield Static("No further tree will start. The running tree is sent SIGTERM (SIGKILL if it does not exit "
                         "in time) and is recorded cancelled; trees that have not started are cancelled. Finished "
                         "runs and the partial run's files are kept.", id="stop-text")
            with Horizontal(id="stop-buttons"):
                yield Button("Keep running", id="keep", compact=True)
                yield Button("Stop batch", id="confirm", variant="error", compact=True)

    def on_mount(self):
        self.query_one("#keep", Button).focus()

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "confirm")


class QuitDialog(ModalScreen):
    """Quitting while a batch runs: dismisses with "leave", "stop" or "stay" (the default, also esc)."""
    BINDINGS = [Binding("escape", "dismiss('stay')", "stay"), Binding("left", "app.focus_previous", show=False),
                Binding("right", "app.focus_next", show=False)]

    def __init__(self, active):
        super().__init__()
        self.active = active

    def compose(self) -> ComposeResult:
        names = ", ".join(i.name for i in self.active)
        with Vertical(id="stop-box"):
            yield Static(Text("A batch is running", style="bold"))
            yield Static(f"{names}\nLeave it running: the coordinator keeps going after XMR4EL closes and starts the "
                         "queued trees. Stop it: no new tree starts and the running one is cancelled.", id="stop-text")
            with Horizontal(id="stop-buttons"):
                yield Button("Leave batch running", id="leave", compact=True)
                yield Button("Stop batch and exit", id="stop", variant="error", compact=True)
                yield Button("Stay in XMR4EL", id="stay", compact=True)

    def on_mount(self):
        self.query_one("#stay", Button).focus()

    def on_button_pressed(self, event):
        self.dismiss(event.button.id)
