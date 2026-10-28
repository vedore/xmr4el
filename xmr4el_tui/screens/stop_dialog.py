"""Cancel Run confirmation (tui.md §5.9): names the exact run; focus starts on Keep running."""
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
