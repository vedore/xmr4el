"""New Experiment, step 2: Model Builder (tui.md §5.14). Model Builder 35 % (tree tabs, pipeline grid) over the
Component Library or the Component Editor (Attributes 40 % | JSON preview 60 %) 65 %.

Every tree's applied configuration lives in the app's draft (`app.draft.trees`). The editor works on a copy of
the slot's keys (`Editing.draft`): typing never changes the tree; Apply validates the slot and replaces the tree's
configuration in one assignment; Cancel, or esc and the footer's discard, drops the copy. The JSON preview is the
tree's applied configuration with the copy merged in (`components.merged`), the same dict Apply commits.
"""
import copy
from dataclasses import dataclass, field
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Select, Static
from textual.widgets.option_list import Option

from xmr4el_tui import components, theme
from xmr4el_tui.components import HYBRID
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.description import Rows
from xmr4el_tui.widgets.pipeline_grid import BRANCHES, PipelineGrid, TreeTabs
from xmr4el_tui.widgets.top_bar import TopBar

FEATURE_SLOTS = {sid for _, slots in BRANCHES for sid in slots}
YES_NO = (("Yes", True), ("No", False))
PLACEHOLDER = {"revision": "main", "max_seq_length": "checkpoint default", "max_features": "no cap",
               "random_state": "unseeded", "context_window": "whole document"}


@dataclass
class Editing:
    """The component being edited: its fields and a working copy of the slot's keys (never the tree's dict)."""
    slot: components.Slot
    comp: components.Component
    origin: dict                                # the slot's keys the copy started from
    draft: dict
    errors: dict = field(default_factory=dict)  # field index -> message of text that does not parse

    @property
    def fields(self):
        return self.comp.fields + self.slot.extras


class Library(Rows):
    """Components the highlighted slot can load: moving changes nothing, enter or a click opens the editor."""
    BINDINGS = [Binding("enter", "select", "edit")]


class JsonPreview(ScrollableContainer):
    """Read-only full configuration of the tree as Apply would commit it; scrolls on its own. y copies it."""
    BINDINGS = [Binding("y", "screen.copy_json", "copy JSON")]


class ModelBuilderScreen(Screen):
    """Hidden keys (in the ? panel): ctrl+tab / ctrl+shift+tab (or ] / [) cycle panels, 1-9 show tree N."""
    crumbs = ("Model Builder",)
    BINDINGS = [Binding("escape", "back", "back"), Binding("ctrl+s", "apply", "apply"),
                Binding("u", "discard", "discard changes"), Binding("f2", "rename", "rename tree"),
                Binding("ctrl+tab,right_square_bracket", "panel(1)", "next panel", show=False),
                Binding("ctrl+shift+tab,left_square_bracket", "panel(-1)", "previous panel", show=False),
                *(Binding(str(i), f"tree({i - 1})", f"tree {i}", show=False) for i in range(1, 10))]

    def __init__(self):
        super().__init__()
        self.editing = None
        self.pending = None      # what runs after the footer's discard (u)
        self.refocus = None      # control to focus again when that question is cancelled
        self.renaming = None     # focus to return to after renaming; None: not renaming

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Vertical(id="mb"):
            with Vertical(id="mb-top"):
                with Horizontal(id="mb-tabs-row"):
                    yield TreeTabs(id="mb-tabs")
                    yield Input(id="mb-rename", placeholder="Tree name", compact=True)
                with ScrollableContainer(id="mb-grid-box"):
                    yield PipelineGrid(id="mb-grid")
            with Vertical(id="mb-bottom"):
                yield Library(id="mb-library")
                with Horizontal(id="mb-editor"):
                    yield VerticalScroll(id="mb-attrs")
                    with JsonPreview(id="mb-json"):
                        yield Static(id="mb-json-text")
        yield ContextFooter()

    def on_mount(self):
        self.query_one("#mb-top").border_title = "Model Builder"
        self.query_one("#mb-rename").display = False
        self.query_one("#mb-editor").display = False
        if not self.draft.trees:
            self.draft.add_tree(self.template())
        self.fill()
        self.query_one(PipelineGrid).focus()

    def on_screen_resume(self):
        self.fill_dataset()

    @property
    def draft(self):
        return self.app.draft

    @property
    def tree(self):
        return self.draft.current_tree()

    def template(self):
        config, source, note = components.template(self.app.root, self.draft.confirmed)
        if note:
            self.app.say(f"Model config: {note}" + (f"; started from {source}" if source else ""), "warning", self)
        return config if config is not None else {"features": HYBRID}

    # --- drawing ------------------------------------------------------------------------------------------------
    def fill(self):
        self.fill_dataset()
        self.query_one(TreeTabs).show(self.draft.trees, self.tree.id)
        self.query_one(PipelineGrid).show(self.tree.config, self.draft.slot)
        if self.editing is None:
            self.fill_library()

    def fill_dataset(self):
        """Shared dataset of every tree (Dataset Setup's confirmed selection), in the top panel's border."""
        c = self.draft.confirmed
        if c is None:
            text = "No dataset confirmed: esc returns to Dataset Setup"
        else:
            name = f"{c.name} — {c.variant}" if c.variant else c.name
            evals = ", ".join(e.split for e in c.enabled()) or "none"
            text = f"{name} · Training: {Path(c.train).stem if c.train else '—'} · Evaluation: {evals}"
        self.query_one("#mb-top").border_subtitle = text

    def fill_library(self, keep=None):
        slot = components.slot(self.draft.slot)
        box = self.query_one(Library)
        box.border_title = f"Components · {slot.name}"
        here = components.installed(self.tree.config, slot)
        width = max(20, box.scrollable_content_region.width or 60)
        options = []
        for comp in slot.components:
            head = Text.assemble((comp.name, "bold"), ("  installed" if comp is here else "", theme.style("accent")))
            desc = Text(comp.description, theme.style("text-muted"))
            for t in (head, desc):
                t.truncate(width, overflow="ellipsis")
            options.append(Option(Text("\n").join([head, desc]), id=comp.id))
        box.set_options(options)
        ids = [c.id for c in slot.components]
        target = keep or (here.id if here else ids[0])
        box.highlighted = ids.index(target) if target in ids else 0

    def on_resize(self, _):
        self.query_one(PipelineGrid).show(self.tree.config, self.draft.slot)
        if self.editing is None:
            self.fill_library(self.library_choice())
        else:
            self.fill_preview()

    def library_choice(self):
        box = self.query_one(Library)
        return box.highlighted_option.id if box.highlighted_option else None

    # --- grid -----------------------------------------------------------------------------------------------------
    def select_slot(self, sid):
        """Grid cursor to `sid`; with an editor open, leaving its slot asks first when it has unapplied changes."""
        if sid == self.draft.slot and self.editing is None:
            return
        if self.editing is not None and sid == self.editing.slot.id:
            return
        self.guard(lambda: self.move_to(sid))

    def move_to(self, sid):
        self.draft.slot = sid
        if sid in FEATURE_SLOTS:
            self.draft.feature_slot = sid
        self.query_one(PipelineGrid).show(self.tree.config, sid)
        self.fill_library()

    def action_library(self):
        if self.editing is None:
            self.query_one(Library).focus()

    async def on_option_list_option_selected(self, event):
        if event.option_list.id == "mb-library":
            slot = components.slot(self.draft.slot)
            await self.open_editor(next(c for c in slot.components if c.id == event.option.id))

    # --- editor -------------------------------------------------------------------------------------------------
    async def open_editor(self, comp):
        """Edit `comp` in the highlighted slot: the installed one starts from its applied settings, another one from
        its defaults (the slot's other keys kept). The tree is not touched."""
        slot = components.slot(self.draft.slot)
        applied = components.part(self.tree.config, slot)
        origin = applied if components.installed(self.tree.config, slot) is comp else \
            {**applied, slot.key: copy.deepcopy(comp.base)}
        self.editing = Editing(slot, comp, origin, copy.deepcopy(origin))
        self.query_one(Library).display = False
        self.query_one("#mb-editor").display = True
        await self.build_fields()
        self.fill_preview(scroll=True)
        self.refresh_bindings()

    async def build_fields(self):
        ed = self.editing
        box = self.query_one("#mb-attrs")
        box.border_title = f"{ed.slot.name} · {ed.comp.name}"
        await box.remove_children()
        rows = [Static(ed.comp.description, classes="mb-desc")]
        for i, f in enumerate(ed.fields):
            v = components.get(ed.draft, components.path_of(ed.slot, f), f.default)
            rows.append(Horizontal(Static(f.label, classes="mb-label"), *self.control(i, f, v), classes="mb-field"))
            rows.append(Static("", id=f"mbe-{i}", classes="mb-error"))
        rows += [Static("", id="mb-help"),
                 Horizontal(Button("Apply", id="mb-apply", compact=True), Button("Cancel", id="mb-cancel", compact=True),
                            id="mb-buttons")]
        await box.mount_all(rows)
        self.show_errors()
        self.query(".mb-control").first().focus()

    def control(self, i, f, v):
        if f.kind == "range":
            pair = v if isinstance(v, list) and len(v) == 2 else ["", ""]
            return [Input(components.show(x), id=f"mbf-{i}-{j}", compact=True, classes="mb-control mb-range",
                          placeholder=("min", "max")[j]) for j, x in enumerate(pair)]
        if f.kind in ("choice", "bool"):
            choices = YES_NO if f.kind == "bool" else f.choices
            values = [c[1] for c in choices]
            known = v in values and (f.kind != "bool" or isinstance(v, bool))
            return [Select([(label, j) for j, (label, _) in enumerate(choices)], id=f"mbf-{i}", compact=True,
                           value=values.index(v) if known else Select.BLANK, allow_blank=not known,
                           classes="mb-control")]
        return [Input(components.show(v), id=f"mbf-{i}", compact=True, classes="mb-control",
                      placeholder=PLACEHOLDER.get(f.path[-1], "required" if f.empty is None else "default"))]

    def field_of(self, widget_id):
        """(index, field, range part or None) of a control id `mbf-<i>[-<part>]`."""
        bits = (widget_id or "").split("-")
        if bits[0] != "mbf" or self.editing is None:
            return None, None, None
        i = int(bits[1])
        return i, self.editing.fields[i], int(bits[2]) if len(bits) > 2 else None

    def on_input_changed(self, event):
        i, f, _ = self.field_of(event.input.id)
        if f is None:
            return
        if f.kind == "range":
            texts = [self.query_one(f"#mbf-{i}-{j}", Input).value for j in (0, 1)]
            parsed = [components.parse(components.Field(f.path, f.label, "int", low=f.low), t) for t in texts]
            err = next((e for _, e in parsed if e), None) or components.check(f, [v for v, _ in parsed])
            new = [v for v, _ in parsed]
        else:
            new, err = components.parse(f, event.value)
        self.edited(i, f, None if err else components.coerce(f, new), err)

    def on_select_changed(self, event):
        i, f, _ = self.field_of(event.select.id)
        if f is None or event.value is Select.BLANK:
            return
        choices = YES_NO if f.kind == "bool" else f.choices
        self.edited(i, f, choices[event.value][1], None)

    def edited(self, i, f, new, err):
        """A control changed: a parsed value goes into the working copy (preview updates); text that does not parse
        keeps the last valid value there and marks the field."""
        ed = self.editing
        if err:
            ed.errors[i] = err
        else:
            ed.errors.pop(i, None)
            components.assign(ed.draft, ed.origin, ed.slot, f, new)
        self.show_errors()
        self.fill_preview()

    def problems(self):
        """{field index: message}: text that does not parse first, then the slot's checks on the working copy."""
        ed = self.editing
        out = dict(ed.errors)
        for f, msg in components.slot_problems(self.proposed(), ed.slot, ed.comp):
            i = ed.fields.index(f) if f in ed.fields else 0
            out.setdefault(i, msg)
        return out

    def show_errors(self):
        if self.editing is None:
            return
        found = self.problems()
        for i, _ in enumerate(self.editing.fields):
            for label in self.query(f"#mbe-{i}"):
                label.update(found.get(i, ""))
                label.display = i in found

    def proposed(self):
        return components.merged(self.tree.config, self.editing.slot, self.editing.draft)

    def dirty(self):
        ed = self.editing
        return ed is not None and (bool(ed.errors) or ed.draft != components.part(self.tree.config, ed.slot))

    def fill_preview(self, scroll=False):
        """Whole proposed configuration; the edited slot's keys (by JSON path) on the accent tint, the rest muted."""
        ed = self.editing
        box = self.query_one(JsonPreview)
        text, first = preview(self.proposed(), ed.slot.keys, box.scrollable_content_region.width or 0)
        self.query_one("#mb-json-text", Static).update(text)
        box.border_title = f"Configuration · {self.tree.name}"
        bad = len(ed.errors)
        box.border_subtitle = (f"{bad} invalid field{'s' * (bad > 1)}: last valid value shown" if bad else
                               "unapplied changes" if self.dirty() else "as applied")
        if scroll and first is not None:
            self.call_after_refresh(lambda: box.scroll_to(y=max(0, first - 1), animate=False))

    def on_descendant_focus(self, event):
        i, f, _ = self.field_of(getattr(event.widget, "id", None))
        if f is not None:
            hint = f.help + (" " if f.help else "") + f"Default: {components.show(f.default) or 'none'}."
            self.query_one("#mb-help", Static).update(Text(hint, theme.style("text-muted")))

    def on_input_submitted(self, event):
        if event.input.id == "mb-rename":
            self.finish_rename(event.value)
        elif event.input.id and event.input.id.startswith("mbf-"):
            self.focus_next()

    def on_button_pressed(self, event):
        if event.button.id == "mb-apply":
            self.action_apply()
        elif event.button.id == "mb-cancel":
            self.close_editor()
            self.to_library()

    def action_apply(self):
        """Validate the working copy; valid: it replaces the tree's configuration (one assignment), the editor
        closes and the grid keeps the slot. Invalid: nothing is committed, the first bad field is focused."""
        ed = self.editing
        if ed is None:
            return
        found = self.problems()
        if found:
            i = min(found)
            self.show_errors()
            self.app.say(f"Not applied: {ed.fields[i].label}: {found[i]}", "error")
            self.query(f"#mbf-{i}, #mbf-{i}-0").first().focus()
            return
        self.draft.apply(self.tree.id, self.proposed())
        name, comp = ed.slot.name, ed.comp.name
        self.close_editor()
        self.fill()
        self.query_one(PipelineGrid).focus()
        self.app.say(f"{name}: {comp} applied to {self.tree.name}", "success")

    def close_editor(self):
        """Drop the working copy; the library for the grid's slot comes back."""
        keep = self.editing.comp.id if self.editing else None
        self.editing = None
        self.query_one("#mb-editor").display = False
        self.query_one("#mb-attrs").remove_children()
        self.query_one(Library).display = True
        self.fill()
        self.fill_library(keep)
        self.refresh_bindings()

    def to_library(self):
        self.query_one(Library).focus()

    # --- unapplied changes ------------------------------------------------------------------------------------------
    def guard(self, then):
        """Run `then` after closing the editor; with unapplied changes the footer asks first (u discards, esc keeps
        editing). Nothing is ever applied on the way."""
        if self.dirty():
            self.pending = then
            if isinstance(self.focused, (Input, Select)):  # u must reach the screen, not the field
                self.refocus = self.focused
                self.query_one("#mb-attrs").focus()
            self.app.ask(f"Discard unapplied changes to {self.editing.slot.name}?", "u", "discard")
            return
        if self.editing is not None:
            self.close_editor()
        if then is not None:
            then()

    def leave_guard(self, then):
        """Called before this screen is closed by navigation (App.go_to): True when it asked instead."""
        if self.dirty():
            self.guard(then)
            return True
        return False

    def confirm_cancelled(self):
        if self.refocus is not None and self.refocus.is_attached:
            self.refocus.focus()
        self.refocus = self.pending = None

    def action_discard(self):
        if self.app.asked("u"):
            then, self.pending, self.refocus = self.pending, None, None
            self.close_editor()
            (then or self.to_library)()
            self.app.say("Unapplied changes discarded.")
        elif self.dirty():
            self.guard(None)
        else:
            self.app.say("No unapplied changes.")

    def check_action(self, action, parameters):
        if action in ("apply", "discard"):
            return self.editing is not None or (action == "discard" and self.app.asking is not None)
        return True

    def action_back(self):
        """Editor -> library (asks with unapplied changes) -> grid -> the previous screen (Dataset Setup)."""
        if self.renaming is not None:
            self.end_rename()
        elif self.editing is not None:
            self.guard(self.to_library)
        elif self.focused is self.query_one(Library):
            self.query_one(PipelineGrid).focus()
        else:
            self.app.pop_screen()

    def action_copy_json(self):
        if self.editing is not None:
            text = components.dumps(self.proposed())
            self.app.copy_to_clipboard(text)
            self.app.say(f"Copied the configuration ({len(text)} characters)")

    # --- panels -----------------------------------------------------------------------------------------------------
    def panels(self):
        out = [self.query_one(TreeTabs), self.query_one(PipelineGrid)]
        if self.editing is None:
            return out + [self.query_one(Library)]
        return out + [self.query_one("#mb-attrs"), self.query_one(JsonPreview)]

    def action_panel(self, step):
        panels = self.panels()
        attrs = self.query_one("#mb-attrs")
        here = next((p for p in panels if self.focused is p or (p is attrs and self.focused in p.walk_children())),
                    None)
        i = panels.index(here) if here is not None else -step
        target = panels[(i + step) % len(panels)]
        if target is attrs:
            target = next(iter(self.query(".mb-control")), attrs)
        target.focus()

    # --- trees ------------------------------------------------------------------------------------------------------
    def switch_to(self, tree_id):
        if tree_id != self.tree.id:
            self.guard(lambda: self.show_tree(tree_id))

    def show_tree(self, tree_id):
        self.draft.tree = tree_id
        self.fill()

    def action_switch_tree(self, step):
        ids = [t.id for t in self.draft.trees]
        j = ids.index(self.tree.id) + step
        if 0 <= j < len(ids):
            self.switch_to(ids[j])

    def action_tree(self, n):
        if n < len(self.draft.trees):
            self.switch_to(self.draft.trees[n].id)

    def action_add_tree(self):
        def add():
            tree = self.draft.add_tree(self.template())
            self.fill()
            self.app.say(f"{tree.name} added (position {self.draft.index(tree.id) + 1})")
        self.guard(add)

    def action_duplicate_tree(self):
        def dup():
            src = self.tree.name
            tree = self.draft.duplicate(self.tree.id)
            self.fill()
            self.app.say(f"{tree.name}: copy of {src}'s applied configuration")
        self.guard(dup)

    def action_remove_tree(self):
        tree = self.tree
        if len(self.draft.trees) == 1:
            self.app.say("A batch needs at least one tree.", "warning")
        elif self.app.asked("delete"):
            if self.editing is not None:
                self.close_editor()
            self.draft.remove(tree.id)
            self.fill()
            self.query_one(TreeTabs).focus()
            self.app.say(f"{tree.name} removed")
        else:
            extra = " and its unapplied changes" if self.dirty() else ""
            self.app.ask(f"Remove {tree.name}? Its configuration{extra} is discarded.", "delete", "remove")

    def action_move_tree(self, step):
        if self.draft.move(self.tree.id, step):
            self.query_one(TreeTabs).show(self.draft.trees, self.tree.id)
            self.app.say(f"{self.tree.name}: position {self.draft.index(self.tree.id) + 1} of {len(self.draft.trees)}")

    def action_rename(self):
        field_ = self.query_one("#mb-rename", Input)
        self.renaming = self.focused or self.query_one(TreeTabs)
        field_.value = self.tree.name
        field_.display = True
        field_.focus()

    def finish_rename(self, name):
        problem = self.draft.rename(self.tree.id, name)
        if problem:
            self.app.say(f"Not renamed: {problem}", "error")
            return
        self.end_rename()
        self.fill()
        if self.editing is not None:
            self.fill_preview()
        self.app.say(f"Renamed to {self.tree.name}")

    def end_rename(self):
        self.renaming = None     # focus to return to after renaming; None: not renaming
        self.query_one("#mb-rename").display = False
        back, self.refocus = self.refocus, None
        (back if back is not None and back.is_attached else self.query_one(TreeTabs)).focus()


def preview(config, keys, width=0):
    """(Rich text of `json.dumps(config, indent=2)`, first highlighted line): lines under the top-level `keys` on the
    section tint (padded to `width` so the tint spans the panel), every other line muted."""
    lines = components.json_lines(config)
    width = max([len(line) for line, _ in lines] + [width])
    text, first = Text(no_wrap=True), None
    for n, (line, key) in enumerate(lines):
        mine = key in keys
        first = n if mine and first is None else first
        text.append(f"{line:<{width}}", f"{theme.color('text')} on {theme.color('section')}" if mine
                    else theme.style("text-muted"))
        if n < len(lines) - 1:
            text.append("\n")
    return text, first
