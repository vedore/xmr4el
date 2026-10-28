"""Texts of the Results screen (tui.md §5.12): baseline line, Comparison Summary, Detailed Analysis cells, inspection.
No file reads and no comparability decisions: Facts and Compat come from services.comparison."""
from rich.console import Group
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from xmr4el_tui import fmt
from xmr4el_tui.runs import state_word
from xmr4el_tui.services.comparison import (BASELINE, CAUTION, GROUP_NOTES, GROUPS, INCOMPATIBLE, NA, NO_DELTA,
                                            VERIFIED, rows, summary_cells)
from xmr4el_tui.theme import beam_color, color, style
from xmr4el_tui.widgets.explorer import CAUTION_MARK, chip

LEVEL_STYLE = {VERIFIED: style("success"), CAUTION: style("warning", "bold"), INCOMPATIBLE: style("error")}
LEGEND = f"Δ: points vs Base (MRR: own scale) · {NO_DELTA}: not verifiably comparable"
HEADER_WIDTH = 24
NAME_WIDTH = 30  # Summary evaluation column at most; longer names end in …
LABEL_WIDTH = max(len(row[0]) for group in GROUPS.values() for row in group) + 3  # inspection: values in one column


def role(i):
    return "Base" if i == 0 else str(i)


def baseline_text(name, beam):
    """`Tree · split` on the left, the beam chip on the right."""
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(Text(name or fmt.DASH, style="bold"), chip(beam))
    return grid


def cell(value, delta, on_selected=False, widths=(0, 0)):
    """`89.04%  +0.13`: the value, then its delta (muted) when the row has one; NO_DELTA dimmer still. `widths`
    (value, delta) pad both so a left-justified column lines up (Rich drops trailing pads of right-justified ones)."""
    value_width, delta_width = widths
    t = Text(f"{value:>{value_width}}", style=style("text-disabled") if value == NA and not on_selected else "")
    if delta is not None:
        shade = "" if on_selected else style("delta-unavailable" if delta == NO_DELTA else "delta")
        t.append(f" {delta:>{delta_width}}", style=shade)
    elif delta_width:
        t.append(" " * (delta_width + 1))
    return t


def role_cell(i, compat, on_selected=False):
    """`Base`, `1`, `2 !`: the Caution mark sits by the role, so a cut name never hides it."""
    t = Text(role(i))
    if compat.level == CAUTION:
        t.append(f" {CAUTION_MARK}", style=Style(color=color("warning-on-accent" if on_selected else "warning"),
                                                 bold=True))
    return t


def summary(entries, compats, names, width=80):
    """Baseline row on the accent background, challengers below in selection order: Acc@1_all, Acc@1, Coverage
    beside the name (columns sized to their content, the rest of the row empty). Below a rule, per Caution
    challenger: its reason, then what matches and which deltas it affects. `width`: the panel's, so names are cut
    (8 to NAME_WIDTH cells) before metrics wrap."""
    table = Table(box=None, expand=True, show_edge=False, pad_edge=False, padding=(0, 2, 0, 0),
                  header_style=style("text-muted"))
    roles = [role_cell(i, c, i == 0) for i, c in enumerate(compats)]
    table.add_column("", no_wrap=True, min_width=max(r.cell_len for r in roles))
    table.add_column("", no_wrap=True, min_width=3)  # beam chip
    all_cells = summary_cells(entries, compats)
    widths = [(max(len(cells[j][0]) for cells in all_cells),
               max((len(d) for cells in all_cells if (d := cells[j][1])), default=0)) for j in range(3)]
    used = max(r.cell_len for r in roles) + 3 + sum(max(v + bool(d) + d, 9) for v, d in widths) + 2 * 6
    table.add_column("Evaluation", min_width=8, max_width=max(8, min(NAME_WIDTH, width - used)), no_wrap=True,
                     overflow="ellipsis")
    for head, (value_width, _) in zip(("Acc@1_all", "Acc@1", "Coverage"), widths):
        table.add_column(head, min_width=max(value_width, len(head)))  # narrow: the delta wraps under its value
    table.add_column("", ratio=1)  # keeps the metrics by the names; the baseline's background spans the row
    for i, (f, c, cells) in enumerate(zip(entries, compats, all_cells)):
        base = i == 0
        table.add_row(roles[i], chip(f.beam), Text(names.get(f.key, f.key[1])),
                      *(cell(v, d, base, w) for (v, d), w in zip(cells, widths)), "",
                      style=Style(bgcolor=color("baseline-row"), color=color("on-selected"), bold=True) if base else None)
    notes = Table.grid(padding=(0, 1, 0, 0))  # wrapped lines stay right of the mark
    notes.add_column(no_wrap=True)
    notes.add_column(ratio=1)
    for i, (f, c) in enumerate(zip(entries, compats)):
        if c.level == CAUTION:
            notes.add_row(Text(f"{CAUTION_MARK} {i}", style=style("warning", "bold")),
                          Text.assemble((names.get(f.key, f.key[1]), "bold"), f": {c.reason}."))
            notes.add_row("", Text(f"Matches {c.matches}; {c.affects}.", style=style("text-muted")))
    if not notes.row_count:
        return table
    return Group(table, Rule(style=color("divider")), notes)


def header(i, f, name):
    """Two lines: role and beam chip, then the name cut to HEADER_WIDTH (the column's width stays compact)."""
    t = Text(f"{role(i)} ", style="bold")
    t.append_text(chip(f.beam))
    line = Text(name)
    line.truncate(HEADER_WIDTH, overflow="ellipsis")
    return t.append("\n").append_text(line)


def spec_cell(label, value, base_value, is_base):
    """A Specs value as recorded: beam_size in its beam colour, a value other than the baseline's in spec-differs."""
    if label == "beam_size" and value != NA:
        return Text(value, style=Style(color=beam_color(int(value)) if value.isdigit() else None, bold=True))
    differs = not is_base and value != base_value
    return Text(value, style=style("spec-differs", "bold") if differs else style("text-disabled")
                if value == NA else "")


def analysis(group, entries, compats, names):
    """(column headers, [(label, cells)], notes) of one group across every evaluation. Caution reasons stay in the
    Summary; notes = the group's own note, known differences that withhold deltas, and the Δ legend."""
    heads = [header(i, f, names.get(f.key, f.key[1])) for i, f in enumerate(entries)]
    found = rows(group, entries, compats)
    if group == "Specs":
        body = [(label, [spec_cell(label, v, cells[0][0], i == 0) for i, (v, _) in enumerate(cells)])
                for label, cells in found]
    else:
        body = [(label, [cell(v, d) for v, d in cells]) for label, cells in found]
    notes = [GROUP_NOTES[group]] if group in GROUP_NOTES else []
    notes += [f"{i}: {n}" for i, c in enumerate(compats) if i for n in c.notes]
    if any(d is not None for _, cells in found for _, d in cells):
        notes.append(LEGEND)
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
            t.append(f"  {label:<{LABEL_WIDTH}}", style="dim")
            t.append_text(cell(cells[0][0], None))
            t.append("\n")
        if group in GROUP_NOTES:
            t.append(f"  {GROUP_NOTES[group]}\n", style="dim")
    return t
