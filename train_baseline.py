"""Lipophilicity baseline：RDKit 描述符 + Morgan 指纹 + 线性/树模型。

职责边界：
- dataclean.py：数据清洗；
- split.py：scaffold 划分；
- features.py：生成特征缓存 features.npz；
- train_baseline.py（本文件）：只读取以上产物，训练和评估 baseline。

本脚本不会重新划分数据、不会删除样本、不会用 test 调参。
模型选择只看 valid RMSE，test 只用于最终报告。

用法：
    python train_baseline.py
    python train_baseline.py --features features.npz --split split.csv

输出：
    baseline_metrics.json       三个模型的 valid/test 指标与最佳模型
    baseline_best.joblib        按 valid RMSE 选出的最佳模型
    baseline_predictions.csv    最佳模型在 valid/test 上的预测
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

BASE_DIR = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# 数据读取与对齐
# ---------------------------------------------------------------------------


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 features.py 生成的特征缓存。

    本地缓存里的 ids 可能是 object 数组（pandas 字符串列的默认行为），
    所以这里允许 pickle；不要用它加载不可信的第三方文件。
    """
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    ids = data["ids"].astype(str)
    names = data["names"]

    if X.ndim != 2:
        raise ValueError(f"特征矩阵必须是二维，实际 shape={X.shape}")
    if len(ids) != X.shape[0]:
        raise ValueError("features.npz 中 ids 与 X 行数不一致")
    if len(names) != X.shape[1]:
        raise ValueError("features.npz 中 names 与 X 列数不一致")
    if len(set(ids)) != len(ids):
        raise ValueError("features.npz 中 ids 有重复")
    if not np.isfinite(X).all():
        raise ValueError("特征矩阵包含 NaN/Inf，请检查 features.py")

    return X, ids, names


def load_training_frame(
    clean_path: Path,
    split_path: Path,
    ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """对齐 clean / split / features，返回带 split 的表和按行对齐的 X。"""
    clean = pd.read_csv(clean_path)
    split = pd.read_csv(split_path, keep_default_na=False)

    required_clean = {"CMPD_CHEMBLID", "exp"}
    required_split = {"CMPD_CHEMBLID", "split"}
    if not required_clean.issubset(clean.columns):
        raise ValueError(f"clean 缺少列：{required_clean - set(clean.columns)}")
    if not required_split.issubset(split.columns):
        raise ValueError(f"split 缺少列：{required_split - set(split.columns)}")

    frame = clean[["CMPD_CHEMBLID", "exp"]].merge(
        split[["CMPD_CHEMBLID", "split"]],
        on="CMPD_CHEMBLID",
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != len(clean):
        raise ValueError(
            f"clean 与 split 行数不一致：clean={len(clean)}, merged={len(frame)}；"
            "请检查 split.csv 是否覆盖全部样本"
        )
    if not frame["split"].isin(SPLIT_NAMES).all():
        raise ValueError("split 列存在 train/valid/test 以外的值")
    if not np.isfinite(frame["exp"].to_numpy(dtype=np.float64)).all():
        raise ValueError("exp 列包含 NaN/Inf，请先运行 dataclean.py")

    # 按 features.npz 的 ID 顺序取行，保证 X 与 frame 一一对应
    id_to_row = {compound_id: index for index, compound_id in enumerate(ids)}
    missing = set(frame["CMPD_CHEMBLID"].astype(str)) - set(id_to_row)
    if missing:
        raise ValueError(f"features.npz 缺少 {len(missing)} 个分子，示例：{list(missing)[:5]}")

    rows = np.array(
        [id_to_row[str(compound_id)] for compound_id in frame["CMPD_CHEMBLID"]],
        dtype=np.int64,
    )
    return frame.reset_index(drop=True), rows


# ---------------------------------------------------------------------------
# 模型与指标
# ---------------------------------------------------------------------------


def build_models(seed: int, n_train: int) -> dict[str, object]:
    """三个 baseline；预处理全部包在 Pipeline 里，只在 train 上 fit。

    RDKit 描述符里有 Ipc 这类重尾特征，直接用 StandardScaler + Ridge
    会让 test 的极端特征外推爆炸。这里用 QuantileTransformer(uniform)
    把每个特征压到 [0, 1]：它不是数据清洗，只是只在 train 上拟合的
    预处理，能显著提高 Ridge 的数值稳定性。
    """
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            QuantileTransformer(
                n_quantiles=min(1000, n_train),
                output_distribution="uniform",
                random_state=seed,
            ),
            Ridge(alpha=100.0),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "hist_gbr": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=seed,
            ),
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE 为主指标，同时给 MAE / R²。"""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Lipophilicity baseline")
    parser.add_argument(
        "--features", type=Path, default=BASE_DIR / "features.npz"
    )
    parser.add_argument(
        "--clean", type=Path, default=BASE_DIR / "Lipophilicity_clean.csv"
    )
    parser.add_argument(
        "--split", type=Path, default=BASE_DIR / "split.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, ids, names = load_features(args.features)
    frame, rows = load_training_frame(args.clean, args.split, ids)
    X = X[rows]

    y = frame["exp"].to_numpy(dtype=np.float32)
    train_mask = (frame["split"] == "train").to_numpy()
    valid_mask = (frame["split"] == "valid").to_numpy()
    test_mask = (frame["split"] == "test").to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"rows={len(frame)}, features={X.shape[1]}")
    print(
        "split: "
        + ", ".join(
            f"{name}={int((frame['split'] == name).sum())}" for name in SPLIT_NAMES
        )
    )

    models = build_models(args.seed, len(X_train))
    results: dict[str, dict[str, dict[str, float]]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

        results[name] = {
            "valid": regression_metrics(y_valid, valid_pred),
            "test": regression_metrics(y_test, test_pred),
        }

        print(f"\n{name}")
        print(f"  valid: {results[name]['valid']}")
        print(f"  test : {results[name]['test']}")

    # 模型选择只看 valid RMSE
    best_name = min(results, key=lambda key: results[key]["valid"]["rmse"])
    best_model = models[best_name]
    best_valid = results[best_name]["valid"]
    best_test = results[best_name]["test"]

    print(f"\nbest model: {best_name}")
    print(f"  valid: {best_valid}")
    print(f"  test : {best_test}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "baseline_best.joblib"
    metrics_path = args.output_dir / "baseline_metrics.json"
    pred_path = args.output_dir / "baseline_predictions.csv"

    joblib.dump(best_model, best_model_path, compress=3)

    pred_all = best_model.predict(X)
    predictions = pd.DataFrame(
        {
            "CMPD_CHEMBLID": frame["CMPD_CHEMBLID"],
            "split": frame["split"],
            "y_true": y,
            "y_pred": pred_all,
        }
    )
    predictions["abs_error"] = (predictions["y_true"] - predictions["y_pred"]).abs()
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")

    payload = {
        "seed": args.seed,
        "n_rows": int(len(frame)),
        "n_features": int(X.shape[1]),
        "feature_blocks": {
            "rdkit_descriptors": int(sum(not name.startswith("morgan_") for name in names)),
            "morgan_bits": int(sum(name.startswith("morgan_") for name in names)),
        },
        "split_counts": {
            name: int((frame["split"] == name).sum()) for name in SPLIT_NAMES
        },
        "models": results,
        "best": {
            "name": best_name,
            "valid": best_valid,
            "test": best_test,
            "model_path": str(best_model_path),
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n已写出：{metrics_path}")
    print(f"已写出：{best_model_path}")
    print(f"已写出：{pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
