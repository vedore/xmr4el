"""New Experiment draft (tui.md §5.4): the dataset being configured and the one confirmed for the batch. Stdlib only.

One draft per TUI session, on the app (`app.draft`, memory only). Dataset Setup edits a `datasets.Selection` per
dataset; nothing reaches `confirmed` until Confirm Dataset validates it. Later New Experiment screens read
`draft.confirmed` (training file, enabled evaluation splits, format): every tree of the batch shares it.

Review Batch (tui.md §5.6) adds the batch's own name (`name`, display only: `id` is the identity runs and paths use),
the execution order edits (`move`) and the configuration reference for Differences (`reference`).

Model Builder (tui.md §5.5) keeps the batch's trees here: `trees` in execution order, each with a stable id, a
display name and its one applied model configuration (a plain dict no other tree shares). Editor drafts never reach
a tree: only Apply replaces its configuration (`apply`).
"""
import copy
import uuid
from dataclasses import dataclass

NAME_MAX = 40
BATCH_NAME_MAX = 60
BATCH_NAME = "Experiment batch"


@dataclass
class Tree:
    id: str      # stable across rename and reorder; the safe identity for later run names and paths
    name: str
    config: dict  # the applied model configuration (components.ordered)
    seq: int = 0  # creation order (never reused): the oldest surviving tree is the configuration reference


class ExperimentDraft:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]  # the batch's identity (stable across renames), safe in run names and paths
        self.name = BATCH_NAME           # display name: any short printable text
        self.created = 0                 # trees ever created: the next Tree.seq
        self.selected = None    # key of the dataset shown in Dataset Setup (being configured)
        self.edits = {}         # dataset key -> Selection that differs from its baseline, not confirmed
        self.confirmed = None   # datasets.Selection shared by every tree of the batch
        self.trees = []         # Tree, in execution order; all use `confirmed`
        self.tree = None        # id of the tree shown in the Model Builder
        self.slot = "transformer"  # Model Builder slot under the cursor, and the last feature slot visited
        self.feature_slot = "char_tfidf"
        self.resume = None      # [(screen kind, constructor kwargs)] of the New Experiment screens closed from a monitor
        self.launched = False   # launched and not resumed since: `n` starts a new experiment, `c` still reopens this one

    def baseline(self, dataset):
        """What `dataset` shows without edits: its confirmed selection, else the recognized default."""
        c = self.confirmed
        return c if c is not None and c.key == dataset.key else dataset.default()

    def current(self, dataset):
        return self.edits.get(dataset.key) or self.baseline(dataset)

    def edit(self, dataset, selection):
        """Keep an edit (never confirms it); an edit back to the baseline is no edit."""
        if selection == self.baseline(dataset):
            self.edits.pop(dataset.key, None)
        else:
            self.edits[dataset.key] = selection

    def changed(self, key):
        return key in self.edits

    def discard(self, key):
        self.edits.pop(key, None)

    def confirm(self, selection, old_key=None):
        """Commit a validated selection (the caller validates); `old_key`: the key it was edited under (a new
        registered dataset gets its key when saved)."""
        self.edits.pop(old_key or selection.key, None)
        self.edits.pop(selection.key, None)
        self.confirmed = selection
        self.selected = selection.key

    # --- trees (Model Builder) ------------------------------------------------------------------------------------
    def current_tree(self):
        return next((t for t in self.trees if t.id == self.tree), self.trees[0] if self.trees else None)

    def index(self, tree_id):
        return next(i for i, t in enumerate(self.trees) if t.id == tree_id)

    def free_name(self, base):
        taken = {t.name.casefold() for t in self.trees}
        if base.casefold() not in taken:
            return base
        return next(n for i in range(2, len(taken) + 3) if (n := f"{base} {i}").casefold() not in taken)

    def add_tree(self, config, after=None, name=None):
        """New tree (own copy of `config`) after tree `after` (None: at the end); it becomes the shown tree."""
        taken = {t.name.casefold() for t in self.trees}
        name = self.free_name(name) if name else next(f"Tree {i}" for i in range(1, len(taken) + 2)
                                                      if f"tree {i}" not in taken)
        tree = Tree(uuid.uuid4().hex[:8], name, copy.deepcopy(config), self.created)
        self.created += 1
        while any(t.id == tree.id for t in self.trees):
            tree.id = uuid.uuid4().hex[:8]
        at = self.index(after) + 1 if after is not None else len(self.trees)
        self.trees.insert(at, tree)
        self.tree = tree.id
        return tree

    def duplicate(self, tree_id):
        """Copy of a tree's applied configuration (never an editor draft) under a new id, right after it."""
        src = self.trees[self.index(tree_id)]
        return self.add_tree(src.config, after=tree_id, name=f"{src.name} copy")

    def remove(self, tree_id):
        """Remove a tree; the last one stays (a batch has at least one). False when refused."""
        if len(self.trees) <= 1:
            return False
        i = self.index(tree_id)
        del self.trees[i]
        if self.tree == tree_id:
            self.tree = self.trees[min(i, len(self.trees) - 1)].id
        return True

    def move(self, tree_id, step):
        """Earlier (-1) / later (+1) in the execution order; False at the end of the queue."""
        i = self.index(tree_id)
        j = i + step
        if not 0 <= j < len(self.trees):
            return False
        self.trees[i], self.trees[j] = self.trees[j], self.trees[i]
        return True

    @property
    def reference(self):
        """The tree Differences compares against: the first-created one still in the batch (by id, so reordering
        never changes it); None for an empty batch."""
        return min(self.trees, key=lambda t: t.seq, default=None)

    @staticmethod
    def batch_name_problem(name):
        name = name.strip()
        if not name:
            return "Name required."
        if len(name) > BATCH_NAME_MAX:
            return f"At most {BATCH_NAME_MAX} characters."
        if not name.isprintable():
            return "Printable characters only."
        return None

    def rename_batch(self, name):
        """Rename the batch (validated); returns the problem or None."""
        problem = self.batch_name_problem(name)
        if problem is None:
            self.name = name.strip()
        return problem

    def name_problem(self, tree_id, name):
        """Why `name` cannot name the tree, or None. Names are display only (ids keep runs apart), but must be one
        printable line, unique (case-insensitive) and short enough for tables."""
        name = name.strip()
        if not name:
            return "Name required."
        if len(name) > NAME_MAX:
            return f"At most {NAME_MAX} characters."
        if not name.isprintable():
            return "Printable characters only."
        if any(t.id != tree_id and t.name.casefold() == name.casefold() for t in self.trees):
            return f"Another tree is named {name!r}."
        return None

    def rename(self, tree_id, name):
        """Rename (validated); returns the problem or None."""
        problem = self.name_problem(tree_id, name)
        if problem is None:
            self.trees[self.index(tree_id)].name = name.strip()
        return problem

    def apply(self, tree_id, config):
        """Replace the tree's applied configuration with a copy of `config` (the caller validated it)."""
        self.trees[self.index(tree_id)].config = copy.deepcopy(config)
