import logging
import torch

from typing import Optional

_pkg_name = "xmr4el"
_pkg_logger = logging.getLogger(_pkg_name)
_pkg_logger.addHandler(logging.NullHandler())

# Machine-readable progress (docs/design/tui.md §8): not propagated, so run.log stays unchanged;
# scripts/run_experiment.py writes the `event` of every record to events.jsonl.
_progress_logger = _pkg_logger.getChild("progress")
_progress_logger.propagate = False


def progress(**event):
    """Emit one progress event (stage, phase, ...); stage log calls carry theirs as extra={"event": ...}."""
    _progress_logger.info("%s", event, extra={"event": event})


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return _pkg_logger.getChild(name) if name else _pkg_logger

def set_verbosity(level: int):
    """0 = WARNING, 1 = INFO, 2+ = DEBUG"""
    mapping = [logging.WARNING, logging.INFO, logging.DEBUG]
    lvl = mapping[min(max(level, 0), len(mapping) - 1)]
    _pkg_logger.setLevel(lvl)


def torch_device() -> torch.device:
    """cuda, then mps (Apple GPU), then cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
