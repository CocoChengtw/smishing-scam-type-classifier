"""Flat and hierarchical scam-type classifiers.

The hierarchical model mirrors a common production pattern for fine-grained
abuse taxonomies: first route a message to a coarse *group* of easily
confused classes, then resolve the exact class with a specialist model
trained only on that group. Groups are learned from data rather than hand
picked: we cluster classes by how often an out-of-fold flat model confuses
them.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline


def build_text_pipeline(
    class_weight: str | None = "balanced", C: float = 4.0, max_features: int = 200_000
) -> Pipeline:
    """Word + character n-gram TF-IDF with a multinomial logistic regression.

    Character n-grams carry most of the signal across 50+ languages without a
    tokenizer per language; word n-grams add brand and phrase cues.
    """
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     max_features=max_features)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                     sublinear_tf=True, max_features=max_features)),
        ]
    )
    clf = LogisticRegression(C=C, max_iter=2000, class_weight=class_weight)
    return Pipeline([("features", features), ("clf", clf)])


def confusion_groups(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], n_groups: int
) -> dict[str, int]:
    """Cluster labels into n_groups by symmetric, row-normalized confusion.

    Distance between classes a and b is 1 - (P(pred=b|a) + P(pred=a|b)) / 2,
    so classes the model mixes up end up in the same group.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sim = (cm + cm.T) / 2
    np.fill_diagonal(sim, 0.0)
    dist = 1.0 - sim / max(sim.max(), 1e-12)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    assignment = fcluster(Z, t=n_groups, criterion="maxclust")
    return {label: int(g) for label, g in zip(labels, assignment)}


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """Two-stage classifier: confusion-derived groups, then per-group experts."""

    def __init__(self, n_groups: int = 3, base: Pipeline | None = None,
                 cv_folds: int = 3, random_state: int = 13):
        self.n_groups = n_groups
        self.base = base
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _base(self) -> Pipeline:
        return self.base if self.base is not None else build_text_pipeline()

    def fit(self, X, y, groups_override: dict[str, int] | None = None):
        X, y = np.asarray(X, dtype=object), np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        if groups_override is not None:
            self.groups_ = dict(groups_override)
        else:
            cv = StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.random_state)
            oof = cross_val_predict(clone(self._base()), X, y, cv=cv, n_jobs=1)
            self.groups_ = confusion_groups(y, oof, list(self.classes_), self.n_groups)

        g = np.array([self.groups_[label] for label in y])
        # A single learned group degenerates to a flat model; skip the router then.
        self.router_ = clone(self._base()).fit(X, g) if len(set(g)) > 1 else None
        self.experts_ = {}
        for gid in sorted(set(g)):
            mask = g == gid
            members = sorted(set(y[mask]))
            if len(members) == 1:
                self.experts_[gid] = members[0]  # single-class group: no model needed
            else:
                self.experts_[gid] = clone(self._base()).fit(X[mask], y[mask])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """P(class) = P(group) * P(class | group), so probabilities stay calibrated-ish."""
        X = np.asarray(X, dtype=object)
        idx = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((len(X), len(self.classes_)))
        if self.router_ is None:
            gid = next(iter(self.experts_))
            p_group, group_ids = np.ones((len(X), 1)), [gid]
        else:
            p_group, group_ids = self.router_.predict_proba(X), self.router_.classes_
        for col, gid in enumerate(group_ids):
            expert = self.experts_[gid]
            if isinstance(expert, str):
                out[:, idx[expert]] += p_group[:, col]
                continue
            p_cls = expert.predict_proba(X)
            for j, c in enumerate(expert.classes_):
                out[:, idx[c]] += p_group[:, col] * p_cls[:, j]
        return out

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
