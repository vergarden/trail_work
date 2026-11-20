"""按 split.csv 把清洗后的分子渲染成固定尺寸 PNG。

职责边界：
- dataclean.py 负责数据清洗；
- split.py 负责 scaffold 划分；
- 本脚本只渲染图像，不删除、不划分、不训练。

默认输出：
    images/
      train/{CMPD_CHEMBLID}.png
      valid/{CMPD_CHEMBLID}.png
      test/{CMPD_CHEMBLID}.png
      manifest.csv

用法：
    python render_images.py
    python render_images.py --size 256 --output-dir images_256
    python render_images.py --limit 20 --output-dir images_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

BASE_DIR = Path(__file__).resolve().parent
REQUIRED_CLEAN_COLUMNS = ("CMPD_CHEMBLID", "canonical_smiles")
REQUIRED_SPLIT_COLUMNS = ("CMPD_CHEMBLID", "split")
SPLIT_NAMES = ("train", "valid", "test")


def safe_filename(value: str) -> str:
    """把 ID 转成安全文件名；当前数据都是 CHEMBLxxx，通常不会变化。"""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name or "molecule"


def render_png(smiles: str, size: int) -> bytes:
    """把单个 SMILES 渲染成 PNG bytes。

    固定参数，保证同一分子在 train/valid/test 中画法一致。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"无法解析 SMILES：{smiles[:80]}")

    # 固定 2D 朝向，避免同一分子每次画出来角度不同
    rdDepictor.Compute2DCoords(mol, canonOrient=True)

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.minFontSize = 12
    opts.maxFontSize = 16
    opts.padding = 0.05
    opts.clearBackground = True

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def load_data(clean_path: Path, split_path: Path) -> pd.DataFrame:
    """读取 clean + split，按 ID 对齐；不做任何清洗或重新划分。"""
    clean = pd.read_csv(clean_path)
    split = pd.read_csv(split_path, keep_default_na=False)

    missing_clean = [c for c in REQUIRED_CLEAN_COLUMNS if c not in clean.columns]
    if missing_clean:
        raise ValueError(f"clean 缺少列：{missing_clean}")
    missing_split = [c for c in REQUIRED_SPLIT_COLUMNS if c not in split.columns]
    if missing_split:
        raise ValueError(f"split 缺少列：{missing_split}")

    frame = clean[["CMPD_CHEMBLID", "canonical_smiles"]].merge(
        split[["CMPD_CHEMBLID", "split"] + (["scaffold"] if "scaffold" in split.columns else [])],
        on="CMPD_CHEMBLID",
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != len(clean) or len(frame) != len(split):
        raise ValueError(
            f"clean/split 行数不一致：clean={len(clean)}, split={len(split)}, "
            f"merged={len(frame)}"
        )
    if not frame["split"].isin(SPLIT_NAMES).all():
        raise ValueError("split 列存在 train/valid/test 以外的值")
    if frame["canonical_smiles"].isna().any():
        raise ValueError("存在空 canonical_smiles，请先运行 dataclean.py")

    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="把分子渲染成固定尺寸 PNG")
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "Lipophilicity_clean.csv",
        help="清洗后的 CSV",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=BASE_DIR / "split.csv",
        help="split.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "images",
        help="输出目录",
    )
    parser.add_argument("--size", type=int, default=224, help="正方形边长")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只渲染前 N 行，0 表示全部（用于 smoke test）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 PNG；默认遇到同名文件会报错",
    )
    args = parser.parse_args()

    if args.size < 64:
        parser.error("--size 至少为 64")
    if args.limit < 0:
        parser.error("--limit 不能为负数")

    RDLogger.DisableLog("rdApp.*")
    frame = load_data(args.input, args.split)
    if args.limit:
        frame = frame.head(args.limit).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        (args.output_dir / split_name).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in frame.itertuples(index=False):
        compound_id = str(row.CMPD_CHEMBLID)
        split_name = str(row.split)
        try:
            png = render_png(row.canonical_smiles, args.size)
        except ValueError as exc:
            raise SystemExit(f"{compound_id}: {exc}；请先运行 dataclean.py") from exc

        # 用 PIL 再检查一次实际尺寸，防止渲染器静默改变大小
        with Image.open(io.BytesIO(png)) as image:
            width, height = image.size
            image.verify()
        if (width, height) != (args.size, args.size):
            raise RuntimeError(
                f"{compound_id}: 渲染尺寸为 {width}x{height}，期望 {args.size}x{args.size}"
            )

        filename = safe_filename(compound_id) + ".png"
        path = args.output_dir / split_name / filename
        if path.exists() and not args.overwrite:
            # 同一 ID 只应出现一次；如果文件名冲突，显式报错而不是覆盖
            raise RuntimeError(
                f"输出文件已存在：{path}；如需覆盖请加 --overwrite"
            )
        path.write_bytes(png)

        manifest_rows.append(
            {
                "CMPD_CHEMBLID": compound_id,
                "split": split_name,
                "scaffold": getattr(row, "scaffold", ""),
                "image_path": path.relative_to(args.output_dir).as_posix(),
                "width": width,
                "height": height,
                "bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"渲染完成：{len(manifest)} 张，尺寸 {args.size}x{args.size}")
    print(manifest["split"].value_counts().reindex(list(SPLIT_NAMES), fill_value=0).to_string())
    print(f"输出目录：{args.output_dir}")
    print(f"manifest：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
