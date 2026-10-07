"""Check import-time quiet settings in fresh processes, without loading a dataset."""
import os
from pathlib import Path
import subprocess
import sys


def test_eval_quiet():
    settings = ("TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY", "HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE")
    env = {k: v for k, v in os.environ.items() if k not in settings}
    code = """
import io
import logging
import runpy
import sys
runpy.run_path('scripts/evaluate.py', run_name='__main__')
from transformers.utils import logging as hf_logging
from tqdm import tqdm
quiet = '-verbose' not in sys.argv
assert hf_logging.get_verbosity() == logging.WARNING
assert logging.getLogger('xmr4el.learning.scoring').isEnabledFor(logging.WARNING)
assert logging.getLogger('unrelated').isEnabledFor(logging.WARNING)
assert logging.getLogger('xmr4el').isEnabledFor(logging.DEBUG) != quiet
logging.getLogger('xmr4el.learning.scoring').warning('ranker fallback preserved')
logging.getLogger('unrelated').warning('unrelated warning preserved')
with tqdm(total=1, file=io.StringIO()) as bar:
    assert bool(bar.disable) == quiet
"""
    for args in ([], ["-verbose"]):
        result = subprocess.run([sys.executable, "-c", code, "-selfcheck", *args], env=env,
                                cwd=Path(__file__).resolve().parents[1], check=True,
                                capture_output=True, text=True)
        assert "ranker fallback preserved" in result.stderr
        assert "unrelated warning preserved" in result.stderr
    print("quiet/verbose imports ok")
