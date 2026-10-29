"""Bottom bar: the screen's key bindings (Textual Footer), or the app's footer state over the whole bar."""
from rich.text import Text
from textual.containers import Horizontal
from textual.widgets import Footer, Static

from xmr4el_tui import theme

LABEL = {"information": "Notice", "warning": "Notice", "error": "Error"}


class ContextFooter(Horizontal):
    """One state at a time, in this priority: a confirmation (until answered, warning tint) > feedback (a few
    seconds, the feedback-row tint) > the shortcuts. Confirmation and feedback replace every shortcut and span the
    whole bar. The state lives on the app (XMR4ELApp.say / action_quit / ask): this only draws it for its own
    screen, so an expiring feedback timer never touches a confirmation (it only clears `app.note`)."""

    def compose(self):
        yield Static(id="footer-note")
        yield Footer(show_command_palette=False, compact=True)

    def on_mount(self):
        self.update_state()

    def update_state(self):
        app, note = self.app, self.query_one("#footer-note", Static)
        confirm = app.confirming is self.screen
        feedback = app.note if app.note and app.note[2] is self.screen else None
        if confirm:
            question, key, verb = app.asking or (app.quit_question(), "ctrl+c", "quit")
            text = Text.assemble((" Confirm ", theme.chip("warning")), " ", question, "   ",
                                 (key, theme.style("accent", "bold")), f" {verb}  ",
                                 ("esc", theme.style("accent", "bold")), " cancel")
        elif feedback:
            severity, message, _ = feedback
            text = Text.assemble((f" {LABEL.get(severity, 'Notice')} ", theme.chip(severity)), " ", message, " ")
        note.update(text if confirm or feedback else "")
        note.display = bool(confirm or feedback)
        self.query_one(Footer).display = not (confirm or feedback)
        self.set_class(confirm, "-confirm")
        self.set_class(bool(feedback) and not confirm, "-feedback")
