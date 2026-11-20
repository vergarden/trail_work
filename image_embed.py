"""冻结 ResNet18 图像嵌入 + Ridge/RandomForest 的 logD 回归实验。

职责边界：
- render_images.py 已负责渲染；
- 本脚本只读取 images + split.csv + clean CSV；
- 不重新划分、不清洗、不微调 CNN。

用法：
    python image_embed.py --images-dir images
    python image_embed.py --images-dir images --output-dir image_results

输出：
    image_embeddings.npz   X (n, 512), ids, split
    image_metrics.json     Ridge / RandomForest 的 valid/test 指标
    image_predictions.csv  最佳图像模型在全部样本上的预测
    image_best.joblib      按 valid RMSE 选出的模型
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

BASE_DIR = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "valid", "test")
SPLIT_COLUMNS = {"CMPD_CHEMBLID", "split", "image_path"}


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def load_manifest(images_dir: Path, manifest_path: Path, clean_path: Path, split_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"找不到 {manifest_path}；请先运行 python render_images.py"
        )
    manifest = pd.read_csv(manifest_path)
    missing = SPLIT_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest 缺少列：{sorted(missing)}")

    clean = pd.read_csv(clean_path)
    split = pd.read_csv(split_path, keep_default_na=False)
    frame = manifest[["CMPD_CHEMBLID", "split", "image_path", "scaffold"]].merge(
        clean[["CMPD_CHEMBLID", "exp"]], on="CMPD_CHEMBLID", how="inner", validate="one_to_one"
    )
    frame = frame.merge(
        split[["CMPD_CHEMBLID", "split"]],
        on="CMPD_CHEMBLID",
        how="inner",
        validate="one_to_one",
        suffixes=("_manifest", "_split"),
    )
    if len(frame) != len(manifest):
        raise ValueError("manifest 与 clean/split 的 ID 集合不一致")
    if not (frame["split_manifest"] == frame["split_split"]).all():
        raise ValueError("manifest 的 split 与 split.csv 不一致；请重新渲染")
    frame = frame.rename(columns={"split_manifest": "split"}).drop(columns=["split_split"])
    frame["image_path"] = frame["image_path"].map(lambda p: images_dir / str(p))
    missing_files = [str(p) for p in frame["image_path"] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"缺少 {len(missing_files)} 张图片，示例：{missing_files[:3]}")
    if not np.isfinite(frame["exp"].to_numpy(dtype=float)).all():
        raise ValueError("exp 包含 NaN/Inf，请先运行 dataclean.py")
    return frame


def extract_embeddings(
    frame: pd.DataFrame,
    *,
    size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = ImagePathDataset(frame["image_path"].tolist(), transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            embeddings.append(model(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ResNet18 image embedding baseline")
    parser.add_argument("--images-dir", type=Path, default=BASE_DIR / "images")
    parser.add_argument(
        "--manifest", type=Path, default=BASE_DIR / "images" / "manifest.csv"
    )
    parser.add_argument("--clean", type=Path, default=BASE_DIR / "Lipophilicity_clean.csv")
    parser.add_argument("--split", type=Path, default=BASE_DIR / "split.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.size < 64:
        parser.error("--size 至少为 64")
    if args.seed < 0:
        parser.error("--seed 必须非负")

    frame = load_manifest(args.images_dir, args.manifest, args.clean, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"images={len(frame)}, device={device}, backbone=resnet18(ImageNet)")

    X = extract_embeddings(
        frame,
        size=args.size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    y = frame["exp"].to_numpy(dtype=np.float32)
    split = frame["split"].to_numpy()
    ids = frame["CMPD_CHEMBLID"].astype(str).to_numpy()

    train_mask = split == "train"
    valid_mask = split == "valid"
    test_mask = split == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(
        "split: "
        + ", ".join(
            f"{name}={int((split == name).sum())}" for name in SPLIT_NAMES
        )
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    models: dict[str, object] = {}

    # Ridge：只在 valid 上选 alpha
    ridge_scores = {}
    best_alpha = None
    best_valid = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, y_train)
        score = metrics(y_valid, model.predict(X_valid))
        ridge_scores[str(alpha)] = score
        print(f"ridge alpha={alpha:<5} valid RMSE={score['rmse']:.4f}")
        if best_valid is None or score["rmse"] < best_valid:
            best_valid = score["rmse"]
            best_alpha = alpha
            models["ridge"] = model

    results["ridge"] = {
        "valid": ridge_scores[str(best_alpha)],
        "test": metrics(y_test, models["ridge"].predict(X_test)),
        "alpha": best_alpha,
        "valid_grid": ridge_scores,
    }

    # RandomForest：固定参数
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf
    results["random_forest"] = {
        "valid": metrics(y_valid, rf.predict(X_valid)),
        "test": metrics(y_test, rf.predict(X_test)),
    }

    best_name = min(results, key=lambda name: results[name]["valid"]["rmse"])
    best_model = models[best_name]
    print(f"\nbest image model: {best_name}")
    for name, value in results.items():
        print(f"  {name:14s} valid={value['valid']} test={value['test']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "image_embeddings.npz",
        X=X,
        ids=np.array(ids, dtype=str),
        split=np.array(split, dtype=str),
    )
    joblib.dump(best_model, args.output_dir / "image_best.joblib", compress=3)

    predictions = pd.DataFrame(
        {
            "CMPD_CHEMBLID": ids,
            "split": split,
            "y_true": y,
            "y_pred": best_model.predict(X),
        }
    )
    predictions["abs_error"] = (predictions["y_true"] - predictions["y_pred"]).abs()
    predictions.to_csv(
        args.output_dir / "image_predictions.csv", index=False, encoding="utf-8-sig"
    )

    payload = {
        "backbone": "resnet18(ImageNet, frozen)",
        "image_size": args.size,
        "n_rows": int(len(frame)),
        "n_features": int(X.shape[1]),
        "split_counts": {
            name: int((split == name).sum()) for name in SPLIT_NAMES
        },
        "seed": args.seed,
        "models": results,
        "best": {"name": best_name, "valid": results[best_name]["valid"], "test": results[best_name]["test"]},
    }
    (args.output_dir / "image_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n已写出：{args.output_dir / 'image_embeddings.npz'}")
    print(f"已写出：{args.output_dir / 'image_metrics.json'}")
    print(f"已写出：{args.output_dir / 'image_predictions.csv'}")
    print(f"已写出：{args.output_dir / 'image_best.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
