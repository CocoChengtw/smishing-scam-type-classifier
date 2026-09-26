import numpy as np
import pandas as pd
import pytest

from scamtax.data import normalize_text
from scamtax.dedup import template_ids
from scamtax.evaluate import bootstrap_ci, review_routing_curve
from scamtax.models import HierarchicalClassifier, build_text_pipeline, confusion_groups
from scamtax.split import grouped_split


def test_normalize_masks_urls_and_whitespace():
    assert normalize_text("Pay  NOW at https://x.co/a\n") == "pay now at <url>"


def test_near_duplicates_share_template():
    texts = [
        "your parcel is waiting, pay the customs fee at <url> today",
        "your parcel is waiting, pay the customs fee at <url> now",
        "hi mum this is my new number, can you text me back",
        "hi mum this is my new number, can you text me back",
    ]
    ids = template_ids(texts, threshold=0.6)
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]
    assert ids[0] != ids[2]


def test_grouped_split_keeps_templates_on_one_side():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"label": rng.choice(["a", "b"], 400)})
    groups = rng.integers(0, 60, 400)
    tr, te = grouped_split(df, groups, 0.2, seed=0)
    assert set(groups[tr.index]).isdisjoint(set(groups[te.index]))


def test_confusion_groups_merge_confused_classes():
    y_true = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    y_pred = np.array(["b"] * 5 + ["a"] * 5 + ["a"] * 5 + ["b"] * 5 + ["c"] * 10)
    g = confusion_groups(y_true, y_pred, ["a", "b", "c"], n_groups=2)
    assert g["a"] == g["b"] != g["c"]


@pytest.fixture
def toy_data():
    X = (["bank account locked verify login"] * 15 + ["card suspended confirm bank"] * 15
         + ["parcel delivery fee unpaid"] * 15 + ["hi mum new phone text me"] * 15)
    y = ["banking"] * 30 + ["delivery"] * 15 + ["family"] * 15
    return np.array(X, dtype=object), np.array(y)


def test_hierarchical_probabilities_sum_to_one(toy_data):
    X, y = toy_data
    base = build_text_pipeline(max_features=1000)
    base.set_params(features__word__min_df=1, features__char__min_df=1)
    model = HierarchicalClassifier(n_groups=2, base=base).fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (model.predict(X) == y).mean() > 0.9


def test_bootstrap_ci_brackets_point_estimate():
    y = np.array(["a", "b"] * 50)
    pred = y.copy()
    pred[:10] = "a"
    point, lo, hi = bootstrap_ci(y, pred, n_boot=200)
    assert lo <= point <= hi


def test_review_routing_coverage_decreases():
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45], [0.2, 0.8]])
    curve = review_routing_curve(np.array(["a", "a", "b", "b"]), proba, ["a", "b"],
                                 thresholds=[0.0, 0.7])
    assert curve["coverage"].tolist() == [1.0, 0.5]
