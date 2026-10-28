"""Run monitor Run overview and Resource monitor panels: display only (the screen fills them)."""
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from xmr4el_tui.widgets.resize import DragRightBorder


class RunOverview(DragRightBorder, VerticalScroll, can_focus=True):
    """Run identity and state; `Cancel Run` (enabled only while the screen says the run is stoppable). At most half
    the column (app.tcss): a long error scrolls here instead of pushing the Pipeline off screen."""
    ALLOW_MAXIMIZE = True

    def compose(self) -> ComposeResult:
        yield Static(id="overview-text")
        yield Static(id="overview-banner")
        yield Button("Cancel Run", id="cancel", variant="error", compact=True, disabled=True)

    def show(self, text, banner, stoppable):
        self.query_one("#overview-text", Static).update(text)
        self.query_one("#overview-banner", Static).update(banner or "")
        self.query_one("#overview-banner").display = banner is not None
        self.query_one("#cancel", Button).disabled = not stoppable


class ResourcePanel(VerticalScroll, can_focus=True):
    """Resource summary of the Pipeline's selected row."""
    ALLOW_MAXIMIZE = True

    def compose(self) -> ComposeResult:
        yield Static(id="resources-text")

    def show(self, title, text):
        self.border_title = title
        self.query_one("#resources-text", Static).update(text)
