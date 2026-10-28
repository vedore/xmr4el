"""Bottom bar: the screen's key bindings (Textual Footer) behind a note that shows the app's footer state."""
from rich.text import Text
from textual.containers import Horizontal
from textual.widgets import Footer, Static

from xmr4el_tui import theme

LABEL = {"information": "Notice", "warning": "Notice", "error": "Error"}


class ContextFooter(Horizontal):
    """Shortcuts; a feedback note left of them for a few seconds; a confirmation in place of them until answered.
    The state lives on the app (XMR4ELApp.say / ask_quit): this only draws it for its own screen."""

    def compose(self):
        yield Static(id="footer-note")
        yield Footer(show_command_palette=False)

    def on_mount(self):
        self.update_state()

    def update_state(self):
        app, note = self.app, self.query_one("#footer-note", Static)
        confirm = app.confirming is self.screen
        feedback = app.note if app.note and app.note[2] is self.screen else None
        if confirm:
            text = Text.assemble((" Confirm ", theme.chip("warning")), " ", app.quit_question(), "   ",
                                 ("ctrl+c", theme.style("accent", "bold")), " quit  ",
                                 ("esc", theme.style("accent", "bold")), " cancel")
        elif feedback:
            severity, message, _ = feedback
            text = Text.assemble((f" {LABEL.get(severity, 'Notice')} ", theme.chip(severity)), " ", message, " ")
        note.update(text if confirm or feedback else "")
        note.display = bool(confirm or feedback)
        self.query_one(Footer).display = not confirm
        self.set_class(confirm, "-confirm")
