"""Results panels: Current Baseline, Comparison Summary, Detailed Analysis, Inspection. Display only (the screen fills
them from screens.results_details)."""
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Static


class BaselinePanel(Static):
    """Tree · evaluation and beam of the comparison reference (set when Results opens, never changed here)."""


class ComparisonSummary(VerticalScroll, can_focus=True):
    """Baseline and challengers with the primary metrics and Caution reasons; as tall as its content (app.tcss)."""

    def compose(self) -> ComposeResult:
        yield Static(id="summary-text")

    def show(self, renderable):
        self.query_one("#summary-text", Static).update(renderable)


class DetailedAnalysis(Vertical):
    """One metric group across every compared evaluation: a table (scrolls both ways) and its notes."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="analysis-table", cursor_type="row", header_height=2)
        yield Static(id="analysis-notes")

    def show(self, heads, body, notes):
        """Refill; the cursor row and scroll offsets are kept (same group: the same metric stays under them)."""
        table = self.query_one(DataTable)
        row, x, y = table.cursor_row, table.scroll_x, table.scroll_y
        table.clear(columns=True)
        table.add_column("Metric", key="metric")
        for i, h in enumerate(heads):
            table.add_column(h, key=str(i))
        for label, cells in body:
            table.add_row(label, *cells)
        table.move_cursor(row=min(row, max(0, table.row_count - 1)), animate=False)
        table.scroll_to(x=x, y=y, animate=False)
        notes_widget = self.query_one("#analysis-notes", Static)
        notes_widget.update("\n".join(notes))
        notes_widget.display = bool(notes)


class InspectionView(VerticalScroll, can_focus=True):
    """Read-only view of one evaluation; replaces the right column until esc."""

    def compose(self) -> ComposeResult:
        yield Static(id="inspect-text")

    def show(self, text):
        self.query_one("#inspect-text", Static).update(text)
