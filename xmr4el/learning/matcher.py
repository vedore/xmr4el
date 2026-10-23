"""Train node matchers against local cluster or label targets."""
import numpy as np

from xmr4el.learning.classifiers import ClassifierModel


class Matcher:
    def __init__(self):
        self.model = None

    def save(self, save_dir):
        if not callable(getattr(self.model, "save", None)):
            raise TypeError(f"matcher model ({type(self.model).__name__}) has no save()")
        self.model.save(save_dir)

    @classmethod
    def load(cls, load_dir):
        model = cls()
        model.model = ClassifierModel.load(load_dir)
        return model

    def train(self, X, Y, C, matcher_config):
        """Y columns already follow the node's local label order; C maps them to matcher targets.
        Rows without a local positive are excluded, as in hierarchy child preparation."""
        keep = np.asarray(Y.sum(axis=1)).ravel() > 0
        targets = ((Y[keep] @ C) > 0).astype(int)
        self.model = ClassifierModel.train(X[keep], targets, matcher_config)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def classes(self):
        return self.model.classes()
