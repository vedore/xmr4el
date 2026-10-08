import logging
import torch

from typing import Optional

_pkg_name = "xmr4el"
_pkg_logger = logging.getLogger(_pkg_name)
_pkg_logger.addHandler(logging.NullHandler())

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return _pkg_logger.getChild(name) if name else _pkg_logger

def set_logger(logger: logging.Logger):
    """Replace package logger (package-level override)."""
    global _pkg_logger
    _pkg_logger = logger

def set_verbosity(level: int):
    """0 = WARNING, 1 = INFO, 2+ = DEBUG"""
    mapping = [logging.WARNING, logging.INFO, logging.DEBUG]
    lvl = mapping[min(max(level, 0), len(mapping) - 1)]
    _pkg_logger.setLevel(lvl)


def torch_device() -> torch.device:
    """cuda, then mps (Apple GPU), then cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
