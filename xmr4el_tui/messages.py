"""Messages more than one screen or widget handles."""
from textual.message import Message


class RunsChanged(Message):
    """The app rescanned outputs/runs."""
