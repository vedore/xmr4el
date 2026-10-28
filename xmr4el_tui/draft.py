"""New Experiment draft (tui.md §5.13): the dataset being configured and the one confirmed for the batch. Stdlib only.

One draft per TUI session, on the app (`app.draft`, memory only). Dataset Setup edits a `datasets.Selection` per
dataset; nothing reaches `confirmed` until Confirm Dataset validates it. Later New Experiment screens read
`draft.confirmed` (training file, enabled evaluation splits, format): every tree of the batch shares it.
"""
import uuid


class ExperimentDraft:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.selected = None    # key of the dataset shown in Dataset Setup (being configured)
        self.edits = {}         # dataset key -> Selection that differs from its baseline, not confirmed
        self.confirmed = None   # datasets.Selection shared by every tree of the batch
        self.trees = []         # tree configurations (Model Builder, next step); all use `confirmed`

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
