"""Texts of the Results screen (tui.md §5.12): baseline line, Comparison Summary, Detailed Analysis cells, inspection.
No file reads and no comparability decisions: Facts and Compat come from services.comparison."""
from rich.console import Group
from rich.style import Style
from rich.table import Table
from rich.text import Text

from xmr4el_tui import fmt
from xmr4el_tui.runs import state_word
from xmr4el_tui.services.comparison import (BASELINE, CAUTION, GROUP_NOTES, GROUPS, INCOMPATIBLE, MAX_CHALLENGERS,
                                            NA, VERIFIED, rows, summary_cells)
from xmr4el_tui.theme import color, style
from xmr4el_tui.widgets.explorer import CAUTION_MARK, chip

LEVEL_STYLE = {VERIFIED: style("success"), CAUTION: style("warning", "bold"), INCOMPATIBLE: style("error")}
DELTA_UNITS = "Δ vs baseline: percentage points; MRR on its own scale"
HEADER_WIDTH = 24


def role(i):
    return "Base" if i == 0 else str(i)


def baseline_text(name, beam):
    """`Tree · split` on the left, the beam chip on the right."""
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(Text(name or fmt.DASH, style="bold"), chip(beam))
    return grid


def cell(value, delta, on_selected=False):
    """`89.04%  +0.13`: the value, then its delta (muted) when the evidence allows one."""
    t = Text(value, style=style("text-disabled") if value == NA and not on_selected else "")
    if delta is not None:
        t.append(f" {delta}", style="" if on_selected else style("delta"))
    return t


def name_cell(name, compat, on_selected=False):
    t = Text(name, no_wrap=True, overflow="ellipsis")
    if compat.level == CAUTION:
        t.append(f" {CAUTION_MARK}", style=Style(color=color("warning-on-accent" if on_selected else "warning"),
                                                 bold=True))
    return t


def summary(entries, compats, names):
    """Baseline row on the accent background, challengers below in selection order, then one line per Caution
    challenger: what is uncertain, what matches, which deltas it affects."""
    table = Table(box=None, expand=True, show_edge=False, pad_edge=False, padding=(0, 0, 0, 1),
                  header_style=style("text-muted"))
    table.add_column("", no_wrap=True)
    table.add_column("", no_wrap=True)  # beam chip
    table.add_column("Evaluation", ratio=1, min_width=8, no_wrap=True, overflow="ellipsis")
    for head in ("Acc@1_all", "Acc@1", "Coverage"):  # narrow: the delta wraps under its value, never cut
        table.add_column(head, justify="right", min_width=7)
    for i, (f, c, cells) in enumerate(zip(entries, compats, summary_cells(entries, compats))):
        base = i == 0
        table.add_row(role(i), chip(f.beam), name_cell(names.get(f.key, f.key[1]), c, base),
                      *(cell(v, d, base) for v, d in cells),
                      style=Style(bgcolor=color("baseline-row"), color=color("on-selected"), bold=True) if base else None)
    lines = [table]
    for i, (f, c) in enumerate(zip(entries, compats)):
        if c.level == CAUTION:
            t = Text.assemble((f"{CAUTION_MARK} {i} ", style("warning", "bold")), (names.get(f.key, f.key[1]), "bold"),
                              f": {c.reason}. ")
            t.append(f"Matches: {c.matches}. Affects: {c.affects}.", style=style("text-muted"))
            lines.append(t)
    if len(entries) == 1:
        lines.append(Text(f"space adds an evaluation from the explorer (up to {MAX_CHALLENGERS}); "
                          "acc@1_all is the comparison metric.", style="dim"))
    else:
        lines.append(Text(DELTA_UNITS, style="dim"))
    return Group(*lines)


def header(i, f, name):
    """Two lines: role and beam chip, then the name cut to HEADER_WIDTH (the column's width stays compact)."""
    t = Text(f"{role(i)} ", style="bold")
    t.append_text(chip(f.beam))
    line = Text(name)
    line.truncate(HEADER_WIDTH, overflow="ellipsis")
    return t.append("\n").append_text(line)


def analysis(group, entries, compats, names):
    """(column headers, [(label, cells)], notes) of one metric group across every evaluation."""
    heads = [header(i, f, names.get(f.key, f.key[1])) for i, f in enumerate(entries)]
    body = [(label, [cell(v, d) for v, d in cells]) for label, cells in rows(group, entries, compats)]
    notes = [GROUP_NOTES[group]] if group in GROUP_NOTES else []
    notes += [f"{i}: {n}" for i, c in enumerate(compats) if i for n in c.notes]  # Caution reasons: Summary only
    return heads, body, notes


def inspection(f, compat, name, is_base):
    """One evaluation, read-only: identity, beam, its standing against the baseline, then every metric group."""
    t = Text()
    t.append(f"{name}\n", style="bold")
    run = f.run
    for k, v in (("Beam", chip(f.beam)), ("Run", f"{run.id} · {state_word(run)}"),
                 ("Split", f.split), ("Test file", (f.metrics or {}).get("test_path") or fmt.DASH)):
        t.append(f"{k:<12}", style="dim")
        t.append_text(v if isinstance(v, Text) else Text(str(v)))
        t.append("\n")
    t.append(f"{'Comparison':<12}", style="dim")
    if is_base:
        t.append("the baseline\n")
    else:
        t.append(compat.level, style=LEVEL_STYLE[compat.level])
        t.append(f"  {compat.reason}\n" if compat.reason else "\n")
        for k, v in (("Matches", compat.matches), ("Affects", compat.affects if compat.level != VERIFIED else ""),
                     *(("Note", n) for n in compat.notes)):
            if v:
                t.append(f"{k:<12}", style="dim")
                t.append(f"{v}\n")
    if not f.metrics:
        t.append("\nNo metrics recorded for this evaluation.\n", style=style("warning"))
        return t
    for group in GROUPS:
        t.append(f"\n{group}\n", style="bold")
        for label, cells in rows(group, [f], [BASELINE]):
            t.append(f"  {label:<26}", style="dim")
            t.append_text(cell(cells[0][0], None))
            t.append("\n")
        if group in GROUP_NOTES:
            t.append(f"  {GROUP_NOTES[group]}\n", style="dim")
    return t
