"""Review Batch, Review Details texts (tui.md §5.15): Overview of one tree's applied configuration by model stage, and
its Differences from the configuration reference. Pure: Rich text lines, no widgets."""
from dataclasses import dataclass

from rich.text import Text

from xmr4el_tui import components, review, theme
from xmr4el_tui.review import ERROR, MISSING

LABEL = 24  # cells of the parameter column
# What an empty value means, by field name (the backend's meaning of a None / left-out key)
NONE_TEXT = {"pooling": "checkpoint default", "max_seq_length": "checkpoint default", "revision": "main",
             "max_features": "no cap", "context_window": "whole document", "random_state": "unseeded",
             "abbrev_expansion": "off", "stop_words": "none", "norm": "none", "class_weight": "none"}
# Fields each slot shows in full; the others of its component go into one compact `More` line (a field with a
# problem is always shown)
PROMINENT = {"transformer": ("revision", "pooling", "max_seq_length", "batch_size"),
             "char_tfidf": ("analyzer", "ngram_range", "min_df", "max_df", "abbrev_expansion"),
             "context_tfidf": ("analyzer", "ngram_range", "min_df", "max_df", "context_window"),
             "mention_svd": ("n_components", "random_state"), "context_svd": ("n_components", "random_state"),
             "hierarchy": ("n_clusters", "depth", "min_leaf_size", "cut_half_cluster", "routing_features"),
             "matcher": ("C", "class_weight"), "search": ("beam_size", "topk", "knn_beta", "leaf_candidates")}


@dataclass
class Line:
    text: Text
    issue: str | None = None  # id of the Issue a row belongs to (enter = Fix Issue)
    head: bool = False        # heading or blank: shown, never under the cursor


def blank():
    return Line(Text(""), head=True)


def head(title):
    return Line(Text(title.upper(), "bold"), head=True)


def issue_line(issue, indent=2):
    style = theme.style("error" if issue.severity == ERROR else "warning")
    text = Text.assemble((" " * indent + ("Issue: " if issue.severity == ERROR else "Caution: "), style),
                         (issue.message, style), ("   [Fix Issue]", theme.style("accent")))
    return Line(text, issue.id)


def value_text(f, v):
    """(text, valid) of a field's effective value as the Overview shows it."""
    if components.check(f, v):
        raw = components.show(v) if not isinstance(v, (list, dict)) else str(v)
        return f"Invalid ({raw[:30]})" if raw else "Invalid (empty)", False
    if v is None:
        return NONE_TEXT.get(f.path[-1], "none"), True
    if f.kind == "bool":
        return ("yes" if v else "no"), True
    if f.kind == "choice":
        return next((c[0] for c in f.choices if c[1] == v), str(v)), True
    if f.kind == "range":
        return f"{v[0]}-{v[1]}", True
    return components.show(v), True


def row(label, value, valid=True, indent=2):
    return Line(Text.assemble((" " * indent + f"{label:<{LABEL - indent}}", theme.style("text-muted")),
                              (value, "" if valid else theme.style("error", "bold"))))


def overview(tree, issues):
    """Lines of the tree's applied configuration by stage, each problem under its slot with Fix Issue. `issues`: the
    tree's own Issues (the shared dataset's are drawn by `dataset`)."""
    config, lines = tree.config, []
    loose = [i for i in issues if i.slot is None and i.path != "features"]
    if loose:
        lines += [head("Tree"), *(issue_line(i) for i in loose), blank()]
    for stage in ("Features", "Hierarchy", "Matcher", "Search"):
        lines.append(head(stage))
        if stage == "Features":
            lines.append(row("Recipe", str(config.get("features", components.HYBRID))))
            lines += [issue_line(i, 4) for i in issues if i.path == "features"]
        for s in (s for s in components.SLOTS if review.STAGE_NAMES.get(s.stage, s.stage) == stage):
            lines += slot_lines(config, s, [i for i in issues if i.slot == s.id])
        lines.append(blank())
    return lines[:-1]


def slot_lines(config, s, issues):
    comp = components.installed(config, s)
    if comp is None:
        title = row(s.name, "unsupported", False, 1)
        return [title, *(issue_line(i, 4) for i in issues)] if issues else [title]
    name = components.component_label(config, s)
    if s.id == "transformer" and comp.id in components.MODEL_NAMES:
        name += f" · {components.MODEL_NAMES[comp.id]}"
    lines, more = [row(s.name, name, True, 1)], []
    bad = {i.field for i in issues if i.field is not None}
    for f in comp.fields + s.extras:
        v = components.value(config, s, f)
        text, valid = value_text(f, v)
        if f.path[-1] in PROMINENT[s.id] or not valid or f in bad:
            lines.append(row(f.label, text, valid, 4))
        else:
            more.append(f"{f.label.lower()} {text}")
    if more:
        lines.append(row("More", " · ".join(more), True, 4))
    return [*lines, *(issue_line(i, 4) for i in issues)]


def dataset(issues):
    """The shared dataset's problems, once, above both views; nothing when it is fine."""
    return [head("Shared dataset"), *(issue_line(i) for i in issues), blank()] if issues else []


def differences(tree, reference, changes, width):
    """Reference and selected names, then only the settings that differ, by stage."""
    muted = theme.style("text-muted")
    lines = [Line(Text.assemble(("REFERENCE  ", muted), reference.name), head=True),
             Line(Text.assemble(("SELECTED   ", muted), (tree.name, theme.style("accent"))), head=True), blank()]
    if tree.id == reference.id:
        return lines + [Line(Text("This tree is the configuration reference.", muted), head=True)]
    if not changes:
        return lines + [Line(Text("No configuration differences from reference.", muted), head=True)]
    col = max(8, (width - LABEL - 2) // 2)
    lines.append(Line(Text.assemble((f"  {'Parameter':<{LABEL - 2}}{'Reference':<{col}}", muted), ("Selected", muted)),
                      head=True))
    stage = None
    for c in changes:
        if c.stage != stage:
            stage = c.stage
            lines += [blank(), head(stage)]
        old, new = (review.show(v) for v in (c.old, c.new))
        lines.append(Line(Text.assemble(
            (f"  {cut(c.label, LABEL - 3):<{LABEL - 2}}", muted),
            (f"{cut(old, col - 2):<{col}}", "" if c.old is not MISSING else muted),
            (cut(new, col), theme.style("accent", "bold") if c.new is not MISSING else muted))))
    return lines


def cut(text, width):
    return text if len(text) <= width else text[:max(1, width - 1)] + "…"
