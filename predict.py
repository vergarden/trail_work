"""用训练好的 logD 模型逐行预测，并对有有效标签的行生成检验报告。

有有效标签时，报告目录会额外输出 residuals.png（预测值 vs 残差）；
没有标签时跳过残差图，不影响预测和指标。
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from dataclean import REQUIRED_COLUMNS, check_smiles
from features import BASE_DIR, DESC_NAMES, MORGAN, featurize

EXP_MIN = -10.0
EXP_MAX = 10.0


def load_model(model_path, feature_path):
    names = DESC_NAMES + [f"morgan_{i}" for i in range(MORGAN.GetOptions().fpSize)]
    with np.load(feature_path, allow_pickle=False) as cache:
        if cache["names"].tolist() != names:
            raise ValueError("当前特征名称或顺序与训练缓存不符，请使用训练环境和对应的缓存。")
    model = joblib.load(model_path)
    if model.n_features_in_ != len(names):
        raise ValueError("模型输入维度与训练特征不符。")
    return model


def evaluate(input_path, model):
    data = pd.read_csv(input_path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    data.columns = data.columns.str.strip()
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"输入缺少列：{', '.join(sorted(missing))}")
    rows = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    rows.insert(0, "source_row", np.arange(len(rows)) + 2)
    features, positions, issues = [], [], []
    for pos, smiles in enumerate(rows["smiles"]):
        canonical, issue = check_smiles(smiles.strip())
        if canonical is not None:
            try:
                vector = featurize(canonical)
                if not np.isfinite(vector).all():
                    raise ValueError("nonfinite features")
            except (ValueError, RuntimeError, OverflowError) as exc:
                issue = f"feature_error: {exc}"
            else:
                features.append(vector)
                positions.append(pos)
        issues.append(issue)

    predictions = np.full(len(rows), np.nan)
    if features:
        values = np.asarray(model.predict(np.stack(features)), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("模型返回了非有限预测值，请检查模型与特征。")
        predictions[positions] = values
    rows["prediction_issue"] = issues
    rows["y_pred"] = predictions
    labels = pd.to_numeric(rows["exp"], errors="coerce").to_numpy(dtype=float)
    valid_labels = (
        np.isfinite(labels)
        & (labels >= EXP_MIN)
        & (labels <= EXP_MAX)
    )
    rows["label_issue"] = np.where(
        valid_labels,
        "",
        "missing_or_out_of_range_exp",
    )
    scored = np.isfinite(predictions) & valid_labels
    rows["abs_error"] = np.nan
    actual, predicted = labels[scored], predictions[scored]
    rows.loc[scored, "abs_error"] = np.abs(predicted - actual)
    metrics = {
        "rows": len(rows),
        "predicted": len(positions),
        "prediction_failures": len(rows) - len(positions),
        "scored": len(actual),
        "invalid_labels": int((~valid_labels).sum()),
        "rmse": float(root_mean_squared_error(actual, predicted)) if len(actual) else None,
        "mae": float(mean_absolute_error(actual, predicted)) if len(actual) else None,
        "r2": float(r2_score(actual, predicted)) if np.unique(actual).size >= 2 else None,
    }
    return rows, metrics


def plot_residuals(rows, path, metrics):
    """有同时有效的预测和标签时，画出 预测值 vs 残差 图。

    用 Pillow 直接绘制，不依赖 matplotlib，避免 Windows 上
    matplotlib 与 libglib/libintl 的 DLL 冲突。
    残差 = prediction - true；横向虚线为 0，点越贴近虚线越好。
    没有有效标签时直接返回 False，不生成图片。
    """
    labels = pd.to_numeric(rows["exp"], errors="coerce").to_numpy(dtype=float)
    predictions = pd.to_numeric(rows["y_pred"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(labels) & np.isfinite(predictions)
    if not mask.any():
        return False

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    y_true = labels[mask]
    y_pred = predictions[mask]
    residuals = y_pred - y_true

    width, height = 900, 600
    left, right, top, bottom = 85, 35, 55, 75
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = float(y_pred.min()), float(y_pred.max())
    if x_max == x_min:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    y_min = min(float(residuals.min()), -0.5)
    y_max = max(float(residuals.max()), 0.5)
    if y_max == y_min:
        y_min, y_max = y_min - 0.5, y_max + 0.5

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, width - right, height - bottom], outline="black")

    # 网格与坐标刻度
    for i in range(5):
        x_value = x_min + (x_max - x_min) * i / 4
        px = sx(x_value)
        draw.line([px, top, px, height - bottom], fill="#e5e5e5")
        draw.text((px - 18, height - bottom + 10), f"{x_value:.1f}", fill="black", font=font)

        y_value = y_min + (y_max - y_min) * i / 4
        py = sy(y_value)
        draw.line([left, py, width - right, py], fill="#e5e5e5")
        draw.text((8, py - 7), f"{y_value:.1f}", fill="black", font=font)

    # 零残差虚线
    zero_y = sy(0.0)
    for x0 in range(left, width - right, 12):
        draw.line([x0, zero_y, min(x0 + 6, width - right), zero_y], fill="red", width=1)

    # 散点
    for x_value, residual in zip(y_pred, residuals):
        px, py = sx(float(x_value)), sy(float(residual))
        draw.ellipse([px - 2.2, py - 2.2, px + 2.2, py + 2.2], fill="#166785")

    rmse = metrics.get("rmse")
    title = "Residuals" if rmse is None else f"Residuals (RMSE={rmse:.4f})"
    draw.text((width // 2 - 95, 15), title, fill="black", font=title_font)
    draw.text((width // 2 - 55, height - 30), "Predicted logD", fill="black", font=font)
    draw.text((8, 18), "Residual (pred - true)", fill="black", font=font)

    image.save(str(path), format="PNG")
    return True


def write_reports(out_dir, rows, metrics):
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir.parent) as tmp:
        report = Path(tmp) / "report"
        report.mkdir()
        rows.to_csv(report / "predictions.csv", index=False)
        issues = rows.loc[rows["prediction_issue"].ne("") | rows["label_issue"].ne("")]
        issues.to_csv(report / "issues.csv", index=False)
        ranked = rows.dropna(subset=["abs_error"]).nlargest(10, "abs_error")
        ranked.to_csv(report / "largest_errors.csv", index=False)
        plot_residuals(rows, report / "residuals.png", metrics)
        (report / "metrics.json").write_text(
            json.dumps(metrics, allow_nan=False, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report.rename(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=BASE_DIR / "baseline_best.joblib")
    parser.add_argument("--features", type=Path, default=BASE_DIR / "features.npz",
                        help="训练时保存的特征缓存，用于核对列名与顺序")
    parser.add_argument("--out", type=Path, required=True, help="新的报告目录")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("报告目录已存在，请为本次检验指定新目录。")
    model = load_model(args.model, args.features)
    rows, metrics = evaluate(args.input, model)
    write_reports(args.out, rows, metrics)
    print(f"已写入 {args.out}：预测 {metrics['predicted']}/{len(rows)} 行，计分 {metrics['scored']} 行。")


if __name__ == "__main__":
    main()
