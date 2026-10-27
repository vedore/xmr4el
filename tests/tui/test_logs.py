"""logs.LogTail: incremental reads from an offset, torn last lines, truncation / replacement, a bounded tail of a
large file, record levels for multi-line records; log_path falls back to the launch console log."""
import os

from fake_runs import make_run
from xmr4el_tui import process, runs
from xmr4el_tui.logs import LogTail, log_path

INFO = "2026-10-27 14:30:00,001 [INFO] xmr4el.run_experiment: "
ERROR = "2026-10-27 14:30:01,002 [ERROR] xmr4el.run_experiment: "


def texts(lines):
    return [line.text for line in lines]


def test_incremental_torn_lines_and_levels(tmp_path):
    path = tmp_path / "run.log"
    tail = LogTail(path)
    assert tail.read() == ([], False) and tail.error == "not created yet"
    path.write_text(f"{INFO}Run started\n{ERROR}Run failed\nTraceback (most recent call last):\n  File")
    lines, reset = tail.read()
    assert texts(lines) == [f"{INFO}Run started", f"{ERROR}Run failed", "Traceback (most recent call last):"]
    assert [x.level for x in lines] == ["INFO", "ERROR", "ERROR"] and not reset and tail.error is None
    assert tail.read() == ([], False), "nothing new: the torn line waits"
    with open(path, "a") as f:
        f.write(' "x.py", line 1\nValueError: boom')
    assert [(x.text, x.level) for x in tail.read()[0]] == [('  File "x.py", line 1', "ERROR")]
    assert texts(tail.read(final=True)[0]) == ["ValueError: boom"], "the writer exited: the last line counts"
    path.unlink()
    assert tail.read() == ([], False) and tail.error == "deleted"


def test_truncation_and_replacement_restart(tmp_path):
    path = tmp_path / "run.log"
    path.write_text(f"{INFO}one\n{INFO}two\n")
    tail = LogTail(path)
    assert len(tail.read()[0]) == 2
    path.write_text(f"{ERROR}new\n")  # truncated in place: smaller than the offset
    lines, reset = tail.read()
    assert reset and texts(lines) == [f"{ERROR}new"]
    other = tmp_path / "other.log"
    other.write_text(f"{INFO}replaced and longer than before\n")
    os.replace(other, path)  # same size or larger, other inode
    lines, reset = tail.read()
    assert reset and texts(lines) == [f"{INFO}replaced and longer than before"]


def test_large_file_reads_a_bounded_tail(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("".join(f"{INFO}line {i}\n" for i in range(20000)))
    tail = LogTail(path, tail_bytes=4096)
    lines, _ = tail.read()
    assert 0 < len(lines) < 100 and texts(lines)[-1] == f"{INFO}line 19999"
    assert all(x.startswith(INFO) for x in texts(lines)), "no fragment of a cut line"
    assert tail.skipped == path.stat().st_size - 4096
    with open(path, "a") as f:  # a burst larger than the tail: skipped too, never all in memory
        f.write("".join(f"{INFO}burst {i}\n" for i in range(1000)))
    lines, _ = tail.read()
    assert texts(lines)[-1] == f"{INFO}burst 999" and all(x.startswith(INFO) for x in texts(lines))
    assert sum(len(x.text) + 1 for x in lines) <= 4096


def test_log_path_falls_back_to_console(tmp_path):
    run = runs.read_run(make_run(tmp_path, "a"), tmp_path)
    assert log_path(tmp_path, run) == run.path / "run.log"
    (run.path / "run.log").unlink()
    assert log_path(tmp_path, run) == run.path / "run.log", "missing: shown as unavailable"
    console = tmp_path / process.CONSOLE_DIR / f"{run.id}.log"
    console.parent.mkdir(parents=True)
    console.write_text("Traceback: crashed before logging\n")
    assert log_path(tmp_path, run) == console
