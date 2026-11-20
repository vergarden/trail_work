"""Evaluate the selected baseline on five scaffold folds of the original train set.

Run with: python scaffold_cv.py
Writes scaffold_cv_metrics.json after all folds pass the coverage checks.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from train_baseline import (
    BASE_DIR, build_models, load_features, load_training_frame, regression_metrics,
)


def main() -> None:
    split_path = BASE_DIR / "split.csv"
    baseline = json.loads((BASE_DIR / "baseline_metrics.json").read_text("utf-8"))
    model_name = baseline["best"]["name"]
    seed = baseline["seed"]
    X, ids, _ = load_features(BASE_DIR / "features.npz")
    frame, rows = load_training_frame(BASE_DIR / "Lipophilicity_clean.csv", split_path, ids)
    split_info = pd.read_csv(
        split_path, usecols=["CMPD_CHEMBLID", "scaffold"], index_col="CMPD_CHEMBLID",
        keep_default_na=False,
    )
    scaffolds = split_info.loc[frame["CMPD_CHEMBLID"], "scaffold"]
    # Match split.py: an empty scaffold denotes one singleton per acyclic molecule.
    keys = np.array([
        scaffold or f"__ACYCLIC__::{compound_id}::{i}"
        for i, (compound_id, scaffold) in enumerate(scaffolds.items())
    ], dtype=object)
    train = frame["split"].eq("train").to_numpy()
    groups = keys[train]
    outside_overlap = len(set(groups) & set(keys[~train]))
    if outside_overlap:
        raise ValueError("Original train and held-out sets share a scaffold group")
    X = X[rows[train]]
    y = frame.loc[train, "exp"].to_numpy(dtype=np.float32)
    coverage = np.zeros(len(y), dtype=int)
    folds = []
    splitter = GroupKFold(n_splits=5)
    with threadpool_limits(limits=2):
        for fold, (fit, held) in enumerate(splitter.split(X, groups=groups), start=1):
            fit_groups, held_groups = set(groups[fit]), set(groups[held])
            overlap = len(fit_groups & held_groups)
            if overlap:
                raise ValueError(f"Fold {fold}: scaffold group crosses fold boundary")
            coverage[held] += 1
            model = build_models(seed, len(fit))[model_name]
            model.fit(X[fit], y[fit])
            metrics = regression_metrics(y[held], model.predict(X[held]))
            folds.append(dict(
                fold=fold, fit_rows=len(fit), held_rows=len(held),
                fit_groups=len(fit_groups), held_groups=len(held_groups),
                group_overlap=overlap, **metrics,
            ))
            print(f"Fold {fold}: {metrics}", flush=True)
    if not np.all(coverage == 1):
        raise ValueError("Each original train row must be held out exactly once")
    summary = {}
    for metric in ("rmse", "mae", "r2"):
        scores = [fold[metric] for fold in folds]
        summary[metric] = {"mean": float(np.mean(scores)), "sample_sd": float(np.std(scores, ddof=1))}
    report = {
        "model": model_name, "pipeline_source": "train_baseline.build_models",
        "model_seed": seed, "sklearn_version": sklearn.__version__,
        "splitter": "GroupKFold(n_splits=5, shuffle=False)", "fold_seed": None,
        "population": "original train split only", "rows": len(y),
        "groups": len(set(groups)), "acyclic_singletons": int((scaffolds.to_numpy()[train] == "").sum()),
        "excluded_split_counts": frame.loc[~train, "split"].value_counts().to_dict(),
        "overlap_with_excluded_groups": outside_overlap,
        "rows_held_out_exactly_once": int(np.sum(coverage == 1)),
        "folds": folds, "unweighted_fold_summary": summary,
    }
    output = BASE_DIR / "scaffold_cv_metrics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{output.name}: {summary}", flush=True)


if __name__ == "__main__":
    main()
