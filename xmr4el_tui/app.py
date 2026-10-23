"""Textual app shell; screens come in later increment-1 steps."""
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header


class XMR4ELApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "XMR4EL"

    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
