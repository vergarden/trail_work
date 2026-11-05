"""RandomForest baseline：训练、预测、特征重要性与可视化。

职责边界：
- 读取 features.npz / Lipophilicity_clean.csv / split.csv；
- 只训练 RandomForestRegressor；
- valid 只用于报告/后续选择，test 只评估一次；
- 不重新划分、不清洗、不删除样本。

复用 train_baseline.py 中的数据对齐和指标函数，避免两套逻辑。

用法：
    python train_rf.py
    python train_rf.py --n-estimators 800 --min-samples-leaf 1

输出：
    rf_metrics.json
    rf_model.joblib
    rf_predictions.csv
    rf_feature_importance.csv
    rf_parity_test.png
    rf_residuals_test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from train_baseline import load_features, load_training_frame, regression_metrics

BASE_DIR = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "valid", "test")


def build_model(
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
    max_depth: int | None,
) -> object:
    """RandomForest 回归 pipeline；预处理只在 train 上 fit。"""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed,
            oob_score=True,
        ),
    )


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    """真实值 vs 预测值；对角线为理想结果。"""
    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.55, edgecolors="none")
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=1)
    ax.set_xlabel("True logD")
    ax.set_ylabel("Predicted logD")
    ax.set_title(title)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    """残差图：检查是否存在系统性偏差。"""
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, s=12, alpha=0.55, edgecolors="none")
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted logD")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="RandomForest baseline")
    parser.add_argument("--features", type=Path, default=BASE_DIR / "features.npz")
    parser.add_argument(
        "--clean", type=Path, default=BASE_DIR / "Lipophilicity_clean.csv"
    )
    parser.add_argument("--split", type=Path, default=BASE_DIR / "split.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument(
        "--max-features",
        default="0.1",
        help="RandomForest 的 max_features：sqrt / log2 / 0.3 等",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="树的最大深度；0 表示不限制",
    )
    args = parser.parse_args()

    # max_features 允许传数字字符串，例如 0.3
    try:
        max_features: str | float = float(args.max_features)
    except ValueError:
        max_features = args.max_features

    X, ids, names = load_features(args.features)
    frame, rows = load_training_frame(args.clean, args.split, ids)
    X = X[rows]

    y = frame["exp"].to_numpy(dtype=np.float32)
    masks = {
        name: (frame["split"] == name).to_numpy() for name in SPLIT_NAMES
    }

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_valid, y_valid = X[masks["valid"]], y[masks["valid"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]

    print(f"rows={len(frame)}, features={X.shape[1]}")
    print(
        "split: "
        + ", ".join(
            f"{name}={len(y[masks[name]])}" for name in SPLIT_NAMES
        )
    )

    max_depth = None if args.max_depth == 0 else args.max_depth

    model = build_model(
        seed=args.seed,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)

    predictions = {
        "train": model.predict(X_train),
        "valid": model.predict(X_valid),
        "test": model.predict(X_test),
    }
    metrics = {
        "train": regression_metrics(y_train, predictions["train"]),
        "valid": regression_metrics(y_valid, predictions["valid"]),
        "test": regression_metrics(y_test, predictions["test"]),
    }

    rf = model.named_steps["randomforestregressor"]
    oob_r2 = float(rf.oob_score_) if hasattr(rf, "oob_score_") else None

    print("\nRandomForest")
    for name in SPLIT_NAMES:
        print(f"  {name:5s}: {metrics[name]}")
    print(f"  oob R2: {oob_r2}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "rf_model.joblib"
    metrics_path = args.output_dir / "rf_metrics.json"
    pred_path = args.output_dir / "rf_predictions.csv"
    importance_path = args.output_dir / "rf_feature_importance.csv"

    joblib.dump(model, model_path, compress=3)

    # 全量预测，便于和 baseline_predictions.csv 对照
    all_pred = model.predict(X)
    pred_df = pd.DataFrame(
        {
            "CMPD_CHEMBLID": frame["CMPD_CHEMBLID"],
            "split": frame["split"],
            "y_true": y,
            "y_pred": all_pred,
        }
    )
    pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "feature": names,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.head(50).to_csv(importance_path, index=False, encoding="utf-8-sig")

    plot_parity(
        y_test,
        predictions["test"],
        args.output_dir / "rf_parity_test.png",
        title="RandomForest: test parity",
    )
    plot_residuals(
        y_test,
        predictions["test"],
        args.output_dir / "rf_residuals_test.png",
        title="RandomForest: test residuals",
    )

    payload = {
        "seed": args.seed,
        "n_rows": int(len(frame)),
        "n_features": int(X.shape[1]),
        "params": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": max_features,
            "max_depth": max_depth,
        },
        "split_counts": {
            name: int(len(y[masks[name]])) for name in SPLIT_NAMES
        },
        "metrics": metrics,
        "oob_r2": oob_r2,
        "outputs": {
            "model": str(model_path),
            "predictions": str(pred_path),
            "feature_importance": str(importance_path),
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n已写出：{metrics_path}")
    print(f"已写出：{model_path}")
    print(f"已写出：{pred_path}")
    print(f"已写出：{importance_path}")
    print(f"已写出：{args.output_dir / 'rf_parity_test.png'}")
    print(f"已写出：{args.output_dir / 'rf_residuals_test.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
