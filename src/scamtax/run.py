"""End-to-end experiment: dedup -> split -> train -> evaluate -> report.

Usage: python -m scamtax.run --data data/final_dataset_output.csv --out reports
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from .data import load_dataset
from .dedup import template_ids
from .evaluate import bootstrap_ci, macro_f1, per_class_report, review_routing_curve, slice_report
from .models import HierarchicalClassifier, build_text_pipeline
from .split import grouped_split, random_split


def cap_per_template(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep at most `cap` messages per template so mass campaigns don't dominate."""
    return (df.sample(frac=1.0, random_state=seed)
              .groupby("template", group_keys=False).head(cap))


def one_per_template(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed).groupby("template").head(1)


def evaluate_model(name, model, test, test_unique, n_boot, log):
    t0 = time.time()
    pred = model.predict(test["clean_text"])
    point, lo, hi = bootstrap_ci(test["label"], pred, groups=test["template"], n_boot=n_boot)
    pred_u = model.predict(test_unique["clean_text"])
    res = {
        "model": name,
        "macro_f1": round(point, 3),
        "macro_f1_ci95": [round(lo, 3), round(hi, 3)],
        "macro_f1_unique_templates": round(macro_f1(test_unique["label"], pred_u), 3),
        "accuracy": round(float((pred == test["label"].to_numpy()).mean()), 3),
    }
    log(f"  {name}: {res}  ({time.time() - t0:.0f}s)")
    return res, pred


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/final_dataset_output.csv")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seed, test_size, n_boot = cfg["seed"], cfg["test_size"], cfg["n_boot"]
    lines: list[str] = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    df = load_dataset(args.data)
    labels = sorted(df["label"].unique())
    log(f"messages={len(df)} labels={labels}")

    # 1. Template clustering
    t0 = time.time()
    df["template"] = template_ids(df["clean_text"], **cfg["dedup"])
    n_tpl = df["template"].nunique()
    exact_dups = int(df["clean_text"].duplicated().sum())
    log(f"templates={n_tpl} ({n_tpl / len(df):.1%} of rows) exact_dups={exact_dups} "
        f"({time.time() - t0:.0f}s)")
    label_dist = df["label"].value_counts().to_dict()

    # 2. Leakage check: identical model, row-level vs template-grouped split
    log("Leakage check (flat, class-balanced):")
    leak = {}
    for split_name, (tr, te) in {
        "random_row_split": random_split(df, test_size, seed),
        "template_grouped_split": grouped_split(df, df["template"].to_numpy(), test_size, seed),
    }.items():
        m = build_text_pipeline(**cfg["model"]).fit(tr["clean_text"], tr["label"])
        leak[split_name] = round(macro_f1(te["label"], m.predict(te["clean_text"])), 3)
        log(f"  {split_name}: macro-F1={leak[split_name]} (test n={len(te)})")

    # 3. Main comparison on the grouped split
    train, test = grouped_split(df, df["template"].to_numpy(), test_size, seed)
    test = test.reset_index(drop=True)
    test_unique = one_per_template(test, seed)
    train_capped = cap_per_template(train, cfg["template_cap"], seed)
    log(f"train={len(train)} (capped={len(train_capped)}) test={len(test)} "
        f"test_unique_templates={len(test_unique)}")

    base_kwargs = cfg["model"]
    runs = {
        "flat_unweighted": (build_text_pipeline(**{**base_kwargs, "class_weight": None}), train),
        "flat_balanced": (build_text_pipeline(**base_kwargs), train),
        "flat_balanced_template_cap": (build_text_pipeline(**base_kwargs), train_capped),
        "hierarchical_template_cap": (
            HierarchicalClassifier(n_groups=cfg["hierarchical"]["n_groups"],
                                   base=build_text_pipeline(**base_kwargs),
                                   random_state=seed),
            train_capped),
    }
    results, preds, fitted = [], {}, {}
    for name, (model, tr) in runs.items():
        t0 = time.time()
        model.fit(tr["clean_text"].to_numpy(), tr["label"].to_numpy())
        log(f"  fit {name} in {time.time() - t0:.0f}s")
        res, pred = evaluate_model(name, model, test, test_unique, n_boot, log)
        results.append(res)
        preds[name], fitted[name] = pred, model

    hier = fitted["hierarchical_template_cap"]
    groups = {}
    for label, g in hier.groups_.items():
        groups.setdefault(int(g), []).append(str(label))
    log(f"learned groups: {list(groups.values())}")

    best = max(results, key=lambda r: r["macro_f1"])["model"]
    log(f"best model: {best}")
    best_pred, best_model = preds[best], fitted[best]
    per_class = per_class_report(test["label"], best_pred, labels)
    per_lang = slice_report(test, best_pred, "language", min_support=cfg["min_slice_support"])
    proba = best_model.predict_proba(test["clean_text"].to_numpy())
    classes = best_model.classes_
    routing = review_routing_curve(test["label"], proba, classes)

    _plot_confusion(test["label"], best_pred, labels, out / "figures" / "confusion_matrix.png")
    joblib.dump(best_model, out / "model.joblib")

    metrics = {
        "dataset": {"messages": len(df), "templates": int(n_tpl), "exact_duplicates": exact_dups,
                    "label_distribution": label_dist},
        "leakage_check": leak,
        "models": results,
        "learned_groups": list(groups.values()),
        "best_model": best,
        "per_class": per_class.to_dict(orient="index"),
        "per_language": per_lang.to_dict(orient="records"),
        "review_routing": routing.to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "run_log.txt").write_text("\n".join(lines) + "\n")
    _write_markdown(out / "results.md", metrics, per_class, per_lang, routing)
    log("done")


def _plot_confusion(y_true, y_pred, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Row-normalized confusion matrix (template-grouped test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_markdown(path, metrics, per_class, per_lang, routing):
    rows = ["| Model | Macro-F1 (95% CI) | Macro-F1, one msg per template | Accuracy |",
            "|---|---|---|---|"]
    for r in metrics["models"]:
        lo, hi = r["macro_f1_ci95"]
        rows.append(f"| {r['model']} | {r['macro_f1']:.3f} ({lo:.3f}–{hi:.3f}) | "
                    f"{r['macro_f1_unique_templates']:.3f} | {r['accuracy']:.3f} |")
    leak = metrics["leakage_check"]
    text = [
        "# Results", "",
        "## Leakage check", "",
        f"- Random row split: macro-F1 **{leak['random_row_split']:.3f}**",
        f"- Template-grouped split: macro-F1 **{leak['template_grouped_split']:.3f}**", "",
        "## Model comparison (template-grouped test set)", "", *rows, "",
        f"Learned confusion groups: {metrics['learned_groups']}", "",
        f"## Per-class ({metrics['best_model']})", "", per_class.to_markdown(), "",
        "## Per-language", "", per_lang.to_markdown(index=False), "",
        "## Review routing", "", routing.to_markdown(index=False), "",
    ]
    Path(path).write_text("\n".join(text))


if __name__ == "__main__":
    main()
