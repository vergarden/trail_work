"""Summarize the saved baseline predictions on the original scaffold test split."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from train_baseline import load_features, regression_metrics

BASE_DIR = Path(__file__).resolve().parent
ID = "CMPD_CHEMBLID"


def load_test(directory):
    split = pd.read_csv(directory / "split.csv", keep_default_na=False)
    pred = pd.read_csv(directory / "baseline_predictions.csv")
    test = split.loc[split["split"] == "test", [ID, "scaffold", "exp"]]
    saved = pred.loc[pred["split"] == "test", [ID, "y_true", "y_pred"]]
    if test.empty or set(test[ID]) != set(saved[ID]):
        raise ValueError("Test IDs differ between split.csv and baseline_predictions.csv")
    frame = test.merge(saved, on=ID, validate="one_to_one")
    if not np.allclose(frame["exp"], frame["y_true"], rtol=0, atol=1e-6):
        raise ValueError("Saved targets differ from split.csv")

    features, ids, names = load_features(directory / "features.npz")
    weights = pd.Series(features[:, list(names).index("MolWt")], index=ids)
    frame["molwt"] = frame[ID].map(weights)
    if not np.isfinite(frame[["molwt", "y_true", "y_pred"]]).all().all():
        raise ValueError("Test molecular weights, targets and predictions must be finite")
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["abs_error"] = frame["residual"].abs()
    frame["squared_error"] = frame["residual"] ** 2
    # The saved split gives each acyclic compound its own group.
    frame["scaffold_group"] = frame["scaffold"].where(
        frame["scaffold"] != "", "acyclic:" + frame[ID]
    )
    return frame


def summarize(frame, groups):
    table = frame.groupby(groups, observed=False).agg(
        n=(ID, "size"), bias=("residual", "mean"),
        mae=("abs_error", "mean"), mse=("squared_error", "mean"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    if table["n"].sum() != len(frame):
        raise ValueError("Group counts do not cover the test split")
    return table


def range_table(frame, column, cuts):
    bins = pd.cut(frame[column], [-np.inf, *cuts, np.inf], right=False)
    rows = ["| 区间 | 分子数 | 平均残差 | MAE | RMSE |", "|---|---:|---:|---:|---:|"]
    for group, row in summarize(frame, bins).iterrows():
        values = [f"{v:.4f}" if pd.notna(v) else "—" for v in row[["bias", "mae", "rmse"]]]
        rows.append(f"| `{group}` | {int(row.n)} | " + " | ".join(values) + " |")
    return "\n".join(rows)


def plot_errors(frame, model, path):
    fig, (parity, hist) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    actual, predicted = frame["y_true"], frame["y_pred"]
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    parity.plot([low, high], [low, high], "--", color="#6c7075", linewidth=1)
    parity.scatter(actual, predicted, s=14, alpha=0.55, color="#166785", edgecolors="none")
    parity.set(xlabel="Measured logD", ylabel="Predicted logD", aspect="equal")
    hist.hist(frame["residual"], bins="fd", color="#166785", edgecolor="white")
    hist.axvline(0, color="#6c7075", linestyle="--", linewidth=1)
    hist.set(xlabel="Residual (predicted − measured logD)", ylabel="Molecules")
    fig.suptitle(f"{model} • scaffold test (n = {len(frame)})")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "reports" / "baseline_errors")
    args = parser.parse_args()
    frame = load_test(args.input_dir)
    saved = json.loads((args.input_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    best = saved["best"]
    scores = regression_metrics(frame["y_true"], frame["y_pred"])
    if len(frame) != saved["split_counts"]["test"] or any(
        not np.isclose(v, best["test"][k], rtol=1e-5, atol=1e-7) for k, v in scores.items()
    ):
        raise ValueError("Recomputed test count or metrics disagree with baseline_metrics.json")

    residual = frame["residual"]
    within_one = (residual.abs() <= 1).sum()
    scaffolds = summarize(frame, "scaffold_group")
    n = len(frame)
    report = f"""# 最佳 baseline 的 scaffold 测试集误差

模型 `{best['name']}` 由验证集 RMSE 选定。按化合物 ID 对齐已保存的测试集划分、预测和分子量，覆盖全部 {n} 个测试分子。每个分子权重相同。下列分析仅描述这一次留出集。

残差 = 预测 logD − 实测 logD，正值表示高估。RMSE 为 {scores['rmse']:.4f}，MAE 为 {scores['mae']:.4f}，R² 为 {scores['r2']:.4f}。平均残差为 {residual.mean():+.4f}，中位数为 {residual.median():+.4f}，第 10 至第 90 百分位为 {residual.quantile(.10):+.4f} 至 {residual.quantile(.90):+.4f}。绝对误差 ≤ 1 的有 {within_one}/{n} 个分子（{within_one/n:.1%}）。

![左图为实测与预测 logD，虚线表示完全一致。右图为残差直方图，虚线表示零误差。](parity_and_residuals.png)

## 按实测 logD 分组

各表中 `[a, b)` 表示 a ≤ x < b。每组指标的分母是该组分子数。

{range_table(frame, 'y_true', [0, 1, 2, 3, 4])}

## 按分子量分组

分子量取自训练时缓存的 RDKit `MolWt` 描述符，单位为 Da。

{range_table(frame, 'molwt', [300, 400, 500])}

## 按骨架分组

测试集含 {frame.loc[frame['scaffold'] != '', 'scaffold'].nunique()} 个非空 Bemis–Murcko 骨架。另有 {(frame['scaffold'] == '').sum()} 个无环分子，沿用划分规则，逐个成组。共 {len(scaffolds)} 组，其中 {(scaffolds['n'] == 1).sum()} 组只有一个分子。单分子组的 MAE 和 RMSE 都等于这一分子的绝对误差，无法估计骨架内的波动或判断哪类骨架更难预测。

[全部骨架组的样本数、平均残差、MAE 和 RMSE](by_scaffold.csv)

在项目目录运行 `python analyze_errors.py` 可从已保存的预测重新生成此报告。
"""
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plot_errors(frame, best["name"], out / "parity_and_residuals.png")
    scaffolds.to_csv(out / "by_scaffold.csv", float_format="%.8f")
    (out / "report.md").write_text(report, encoding="utf-8")
    print(f"{n} test molecules, RMSE {scores['rmse']:.4f}. Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
