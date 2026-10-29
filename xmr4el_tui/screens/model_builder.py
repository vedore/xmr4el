"""New Experiment, step 2: Model Builder (tui.md §5.14). Model Builder 45 % (tree tabs, zoom, pipeline grid) over the
Component Library or the Component Editor (Attributes 40 % | JSON preview 60 %) 55 %.

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
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from xmr4el_tui import components, theme
from xmr4el_tui.components import HYBRID
from xmr4el_tui.widgets.context_footer import ContextFooter
from xmr4el_tui.widgets.description import Rows
from xmr4el_tui.widgets.pipeline_grid import BRANCHES, ActionsButton, PipelineGrid, TreeTabs, ZoomBar
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


class AttrSelect(Select):
    """A dropdown in the attribute list: collapsed, up / down move to the previous / next parameter (enter or space
    opens it); expanded, its options take the arrows."""
    BINDINGS = [Binding("up", "screen.field_step(-1)", show=False), Binding("down", "screen.field_step(1)", show=False)]


class TreeMenu(OptionList):
    """Tree Actions: closes (focus left to the screen's choice) when it loses focus."""

    def on_blur(self):
        self.screen.close_menu(restore=False)


TREE_ACTIONS = (("rename", "Rename tree", "F2"), ("duplicate", "Duplicate tree", "d"), ("earlier", "Move earlier", ""),
                ("later", "Move later", ""), ("add", "Add tree", "n"), ("remove", "Remove tree", "delete"))


class JsonPreview(ScrollableContainer):
    """Read-only full configuration of the tree as Apply would commit it; scrolls on its own. y copies it."""
    BINDINGS = [Binding("y", "screen.copy_json", "copy JSON")]


class ModelBuilderScreen(Screen):
    """Levels, each entered with enter and left with esc: Tree Selection (the tab strip) > grid > library > editor
    (attributes, Preview). tab / shift+tab move inside the focused panel, up / down between parameters, 1-9 show
    tree N."""
    crumbs = ("Model Builder",)
    BINDINGS = [Binding("escape", "back", "back"), Binding("ctrl+s", "apply", "apply"),
                Binding("u", "discard", "discard changes"), Binding("f2", "rename", show=False),
                Binding("tab", "local(1)", show=False), Binding("shift+tab", "local(-1)", show=False),
                Binding("up", "field_step(-1)", show=False), Binding("down", "field_step(1)", show=False),
                *(Binding(str(i), f"tree({i - 1})", f"tree {i}", show=False) for i in range(1, 10))]

    def __init__(self):
        super().__init__()
        self.editing = None
        self.pending = None      # what runs after the footer's discard (u)
        self.refocus = None      # control to focus again when that question is cancelled
        self.renaming = None     # focus to return to after renaming; None: not renaming
        self.attr_focus = None   # the attribute control last focused (where esc from the preview returns)
        self.rename_id = None    # the tree being renamed
        self.menu_origin = None  # focus to return to when Tree Actions closes

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Vertical(id="mb"):
            with Vertical(id="mb-top"):
                with Horizontal(id="mb-tabs-row"):
                    yield TreeTabs(id="mb-tabs")
                    yield ActionsButton(id="mb-actions")
                    yield ZoomBar(id="mb-zoom")
                    yield Input(id="mb-rename", placeholder="Tree name", compact=True)
                with ScrollableContainer(id="mb-grid-box"):
                    yield PipelineGrid(id="mb-grid")
            with Vertical(id="mb-bottom"):
                yield Library(id="mb-library")
                with Horizontal(id="mb-editor"):
                    with Vertical(id="mb-attrs-panel"):
                        yield VerticalScroll(id="mb-attrs")
                        with VerticalScroll(id="mb-help"):
                            yield Static(id="mb-help-text")
                        with Horizontal(id="mb-buttons"):
                            yield Button("Apply", id="mb-apply", compact=True)
                            yield Button("Cancel", id="mb-cancel", compact=True)
                            yield Button("Preview", id="mb-preview", compact=True)
                    with JsonPreview(id="mb-json"):
                        yield Static(id="mb-json-text")
        yield TreeMenu(*(Option(Text.assemble(label, ("  " + key if key else "", theme.style("text-muted"))), id=cid)
                         for cid, label, key in TREE_ACTIONS), id="mb-menu")
        yield ContextFooter()

    def on_mount(self):
        self.query_one("#mb-top").border_title = "Model Builder"
        self.query_one("#mb-rename").display = False
        self.query_one("#mb-editor").display = False
        self.query_one("#mb-menu").display = False
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
        if event.option_list.id == "mb-menu":
            self.run_tree_action(event.option.id)
        elif event.option_list.id == "mb-library":
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
        self.query_one("#mb-attrs-panel").border_title = f"{ed.slot.name} · {ed.comp.name}"
        await box.remove_children()
        rows = [Static(ed.comp.description, classes="mb-desc")]
        for i, f in enumerate(ed.fields):
            v = components.get(ed.draft, components.path_of(ed.slot, f), f.default)
            rows.append(Horizontal(Static(f.label, classes="mb-label"), *self.control(i, f, v), classes="mb-field"))
            rows.append(Static("", id=f"mbe-{i}", classes="mb-error"))
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
            return [AttrSelect([(label, j) for j, (label, _) in enumerate(choices)], id=f"mbf-{i}", compact=True,
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
            self.fill_help(f)
        if self.panel_of(event.widget) == "attrs":
            self.attr_focus = event.widget

    def fill_help(self, f):
        """The parameter's name, what it does in XMR4EL, its JSON path and accepted values (fixed-height, scrolls)."""
        muted = theme.style("text-muted")
        text = Text.assemble((f.label, "bold"), "\n", f.help, "\n", ("JSON path: ", muted),
                             components.json_path(self.editing.slot, f), "\n", ("Constraints: ", muted),
                             components.constraints(f))
        self.query_one("#mb-help-text", Static).update(text)
        self.query_one("#mb-help").scroll_home(animate=False)

    def on_input_submitted(self, event):
        if event.input.id == "mb-rename":
            self.finish_rename(event.value)
        elif event.input.id and event.input.id.startswith("mbf-"):
            i, _, _ = self.field_of(event.input.id)
            if i + 1 < len(self.editing.fields):
                self.action_field_step(1)
            else:
                self.query_one("#mb-apply").focus()

    def on_button_pressed(self, event):
        if event.button.id == "mb-preview":
            self.query_one(JsonPreview).focus()
            return
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
        self.attr_focus = None
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
        """Preview -> attributes; editor -> library (asks with unapplied changes) -> grid -> Tree Selection -> the
        previous screen (Dataset Setup)."""
        if self.query_one(TreeMenu).display:
            self.close_menu()
        elif self.renaming is not None:
            self.end_rename()
        elif self.editing is not None:
            if self.focused is self.query_one(JsonPreview):
                focus = self.attr_focus
                (focus if focus is not None and focus.is_attached else self.query_one(".mb-control")).focus()
            else:
                self.guard(self.to_library)
        elif self.focused is self.query_one(Library):
            self.query_one(PipelineGrid).focus()
        elif self.focused is self.query_one(PipelineGrid):
            self.query_one(TreeTabs).cursor = self.tree.id  # Tree Selection
            self.query_one(TreeTabs).focus()
        else:
            self.app.pop_screen()

    def action_copy_json(self):
        if self.editing is not None:
            text = components.dumps(self.proposed())
            self.app.copy_to_clipboard(text)
            self.app.say(f"Copied the configuration ({len(text)} characters)")

    # --- panels -----------------------------------------------------------------------------------------------------
    def containers(self):
        """{panel name: its container}, in ctrl+tab order, for the panels that are visible now."""
        out = {"top": self.query_one("#mb-top")}
        if self.editing is None:
            out["library"] = self.query_one(Library)
        else:
            out.update(attrs=self.query_one("#mb-attrs-panel"), json=self.query_one(JsonPreview))
        return out

    def panel_of(self, widget):
        if widget is None:
            return None
        return next((n for n, box in self.containers().items() if box is widget or box in widget.ancestors), None)

    def members(self, name):
        """The controls Tab visits in a panel, in order: only focusable ones that are shown."""
        if name == "top":
            found = [self.query_one(TreeTabs), self.query_one(PipelineGrid)]
        elif name == "library":
            found = [self.query_one(Library)]
        elif name == "attrs":
            found = [*self.query(".mb-control"), self.query_one("#mb-help"), self.query_one("#mb-apply"),
                     self.query_one("#mb-cancel"), self.query_one("#mb-preview")]
        else:
            found = [self.query_one(JsonPreview)]
        return [w for w in found if w.display and w.visible and not w.disabled]

    def action_local(self, step):
        """Tab / shift+tab: the next / previous control of the focused panel, wrapping inside it."""
        here, name = self.focused, self.panel_of(self.focused)
        members = self.members(name) if name and self.menu_origin is None and self.renaming is None else []
        if here in members:
            members[(members.index(here) + step) % len(members)].focus()

    def action_field_step(self, step):
        """up / down in the attribute list: the previous / next parameter (nothing is applied or edited)."""
        i, _, _ = self.field_of(getattr(self.focused, "id", None))
        if i is not None and 0 <= i + step < len(self.editing.fields):
            self.query(f"#mbf-{i + step}, #mbf-{i + step}-0").first().focus()

    # --- tree actions -----------------------------------------------------------------------------------------------
    def action_tree_menu(self):
        """Tree Actions (a on the tab strip, or the Actions button): rename, duplicate, move, add, remove."""
        menu = self.query_one(TreeMenu)
        if menu.display:
            return
        self.menu_origin = self.focused or self.query_one(TreeTabs)
        if isinstance(self.menu_origin, TreeTabs):  # removal belongs to Tree Selection only
            menu.enable_option("remove")
        else:
            menu.disable_option("remove")
        button = self.query_one(ActionsButton).region
        menu.styles.offset = (max(0, button.right - 30), button.y + 1)
        menu.display = True
        menu.highlighted = 0
        menu.focus()

    def close_menu(self, restore=True):
        menu = self.query_one(TreeMenu)
        if not menu.display:
            return
        back, self.menu_origin = self.menu_origin, None
        menu.display = False
        if restore:
            (back if back is not None and back.is_attached and back.display else self.query_one(TreeTabs)).focus()

    ACTIONS = {"rename": "action_rename", "duplicate": "action_duplicate_tree", "add": "action_add_tree",
               "remove": "action_remove_tree"}

    def run_tree_action(self, cid):
        """The same operations as the keys: the tab strip is focused first (a question in the footer needs its keys)."""
        self.close_menu(restore=False)
        self.query_one(TreeTabs).focus()
        if cid in ("earlier", "later"):
            self.action_move_tree(-1 if cid == "earlier" else 1)
        else:
            getattr(self, self.ACTIONS[cid])()

    # --- trees ------------------------------------------------------------------------------------------------------
    def switch_to(self, tree_id):
        if tree_id != self.tree.id:
            self.guard(lambda: self.show_tree(tree_id))

    def show_tree(self, tree_id):
        self.draft.tree = tree_id
        self.query_one(TreeTabs).cursor = tree_id
        self.fill()

    def activate_tree(self, tree_id):
        """enter in Tree Selection: show the tree (asks first when the editor has unapplied changes) and go to its grid."""
        def go():
            if tree_id != self.tree.id:
                self.show_tree(tree_id)
            self.query_one(PipelineGrid).focus()
        if tree_id == self.tree.id:
            go()
        else:
            self.guard(go)

    def target(self):
        """The tree the tree actions apply to: the Tree Selection highlight (the shown tree outside it)."""
        cursor = self.query_one(TreeTabs).cursor
        if cursor == "add":
            self.app.say("Highlight a tree first.", "warning")
            return None
        return next((t for t in self.draft.trees if t.id == cursor), self.tree)

    def action_tree(self, n):
        if n < len(self.draft.trees):
            self.switch_to(self.draft.trees[n].id)

    def action_add_tree(self):
        def add():
            tree = self.draft.add_tree(self.template())
            self.query_one(TreeTabs).cursor = tree.id
            self.fill()
            self.query_one(PipelineGrid).focus()
            self.app.say(f"{tree.name} added (position {self.draft.index(tree.id) + 1})")
        self.guard(add)

    def action_duplicate_tree(self):
        src = self.target()
        if src is None:
            return

        def dup():
            tree = self.draft.duplicate(src.id)
            self.query_one(TreeTabs).cursor = tree.id
            self.fill()
            self.app.say(f"{tree.name}: copy of {src.name}'s applied configuration")
        self.guard(dup)

    def action_remove_tree(self):
        """x (Tree Selection) or the menu: asks, then removes the highlighted tree; Tree Selection stays."""
        tree, tabs = self.target(), self.query_one(TreeTabs)
        if tree is None:
            return
        if len(self.draft.trees) == 1:
            self.app.say("A batch needs at least one tree.", "warning")
        elif self.app.asked("y"):
            shown = tree.id == self.tree.id
            if self.editing is not None and shown:
                self.close_editor()
            i = self.draft.index(tree.id)
            self.draft.remove(tree.id)
            tabs.cursor = self.draft.trees[min(i, len(self.draft.trees) - 1)].id
            self.fill()
            tabs.focus()
            self.app.say(f"{tree.name} removed")
        else:
            extra = " and its unapplied changes" if self.dirty() and tree.id == self.tree.id else ""
            self.app.ask(f"Remove {tree.name}? Its configuration{extra} is discarded.", "y", "remove")

    def action_confirm_remove(self):
        """y answers the footer's "Remove tree?" (no other effect: it never starts a removal)."""
        if self.app.asking and self.app.asking[1] == "y":
            self.action_remove_tree()

    def action_move_tree(self, step):
        tree = self.target()
        if tree is not None and self.draft.move(tree.id, step):
            self.query_one(TreeTabs).show(self.draft.trees, self.tree.id)
            self.app.say(f"{tree.name}: position {self.draft.index(tree.id) + 1} of {len(self.draft.trees)}")

    def action_rename(self):
        field_, tree = self.query_one("#mb-rename", Input), self.target()
        if tree is None:
            return
        self.rename_id = tree.id
        self.renaming = self.focused or self.query_one(TreeTabs)
        field_.value = tree.name
        field_.display = True
        field_.focus()

    def finish_rename(self, name):
        problem = self.draft.rename(self.rename_id, name)
        if problem:
            self.app.say(f"Not renamed: {problem}", "error")
            return
        renamed = self.draft.trees[self.draft.index(self.rename_id)].name
        self.end_rename()
        self.fill()
        if self.editing is not None:
            self.fill_preview()
        self.app.say(f"Renamed to {renamed}")

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
