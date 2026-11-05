"""Lipophilicity 数据集清洗模块（重构版）。

原脚本已备份为 ``dataclean.py.bak``，本文件保持相同的清洗顺序、移出原因和
输出列（4200 行 -> 保留 3778 行，移出 422 行），但拆分为可复用函数，方便
后续 ``train_*.py`` / ``predict.py`` 直接 import，避免训练与推理预处理不一致。

命令行用法::

    python dataclean.py
    python dataclean.py --input Lipophilicity.csv \\
        --clean-out Lipophilicity_clean.csv \\
        --review-out Lipophilicity_review.csv

作为库使用::

    from dataclean import clean_file, load_raw, clean_dataframe

    result = clean_file("Lipophilicity.csv")
    clean_df = result.clean        # 清洗后的数据
    review_df = result.review      # 被移出的行及原因
    print(result.summary())        # 可直接写进 README 的统计

清洗规则（顺序与原脚本一致）：

1. 完全重复的记录（按 CMPD_CHEMBLID/exp/smiles 三列判断）；
2. 关键字段缺失；
3. exp 非数字、NaN、无穷；
4. exp 超出配置范围（默认 -10 ~ 10，仅为本项目的核查范围）；
5. SMILES 结构问题：超长、非法、重原子过多、多组分；
6. CMPD_CHEMBLID 格式不合法；
7. 同一 canonical SMILES 对应不同 exp（整组隔离，不取平均）；
8. 同一 canonical SMILES 且 exp 相同（保留第一条）。

与备份脚本相比，清洗结果 CSV 完全一致；review CSV 的行内容和行数一致，
只是列顺序调整为 ``..., canonical_smiles, structure_issue, reason``。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

SCRIPT_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS: tuple[str, ...] = ("CMPD_CHEMBLID", "exp", "smiles")
ID_PATTERN = r"CHEMBL[0-9]+"

DEFAULT_MIN_EXP = -10.0
DEFAULT_MAX_EXP = 10.0
DEFAULT_MAX_SMILES_LEN = 1000
DEFAULT_MAX_HEAVY_ATOMS = 200

# 移出原因（保持与原脚本一致的字符串，便于对照历史 report）
REASON_DUPLICATE_RECORD = "duplicate_record"
REASON_MISSING_VALUE = "missing_value"
REASON_INVALID_EXP = "invalid_exp"
REASON_EXP_NEEDS_REVIEW = "exp_needs_review"
REASON_STRUCTURE_NEEDS_REVIEW = "structure_needs_review"
REASON_INVALID_ID = "invalid_id"
REASON_CONFLICTING_LABELS = "conflicting_labels"
REASON_DUPLICATE_MOLECULE = "duplicate_molecule"

# 结构问题的细分类，写在 review 的 structure_issue 列
ISSUE_SMILES_TOO_LONG = "smiles_too_long"
ISSUE_INVALID_SMILES = "invalid_smiles"
ISSUE_MOLECULE_TOO_LARGE = "molecule_too_large"
ISSUE_MULTIPLE_FRAGMENTS = "multiple_fragments"
ISSUE_CANONICALIZATION_FAILED = "canonicalization_failed"

REVIEW_COLUMNS: tuple[str, ...] = (
    *REQUIRED_COLUMNS,
    "source_row",
    "exp_raw",
    "canonical_smiles",
    "structure_issue",
    "reason",
)

__all__ = [
    "CleanResult",
    "check_smiles",
    "clean_dataframe",
    "clean_file",
    "load_raw",
    "main",
    "REQUIRED_COLUMNS",
    "REVIEW_COLUMNS",
]


@dataclass
class CleanResult:
    """一次清洗的结果，``clean``/``review`` 可直接写出为 CSV。"""

    clean: pd.DataFrame
    review: pd.DataFrame
    n_raw: int

    @property
    def n_clean(self) -> int:
        return len(self.clean)

    @property
    def n_removed(self) -> int:
        return len(self.review)

    def reason_counts(self) -> pd.Series:
        """按原因统计移出数量（降序）。"""
        if self.review.empty:
            return pd.Series(dtype="int64")
        return self.review["reason"].value_counts()

    def summary(self) -> dict[str, object]:
        """返回可直接序列化为 JSON / 写进 README 的统计信息。"""
        exp = self.clean["exp"]
        exp_stats: dict[str, float] = {}
        if len(exp):
            exp_stats = {
                "min": float(exp.min()),
                "max": float(exp.max()),
                "mean": float(exp.mean()),
                "std": float(exp.std()) if len(exp) > 1 else 0.0,
            }
        return {
            "n_raw": self.n_raw,
            "n_clean": self.n_clean,
            "n_removed": self.n_removed,
            "removed_by_reason": {
                str(reason): int(count) for reason, count in self.reason_counts().items()
            },
            "exp_stats": exp_stats,
        }


def load_raw(path: str | Path) -> pd.DataFrame:
    """读取原始 CSV；只去掉列名两端空白，不做任何过滤。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def check_smiles(
    smiles: object,
    *,
    max_smiles_len: int = DEFAULT_MAX_SMILES_LEN,
    max_heavy_atoms: int = DEFAULT_MAX_HEAVY_ATOMS,
) -> tuple[str | None, str]:
    """校验单个 SMILES，返回 ``(canonical_smiles, issue)``。

    成功时 ``issue`` 为空字符串；失败时 canonical 为 None。
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, ISSUE_INVALID_SMILES

    # 关键顺序：超长 SMILES 必须在 MolFromSmiles/MolToSmiles 之前拦截。
    # RDKit 对数千个原子的分子做规范化时会递归过深，进程直接栈溢出
    # （Windows 下表现为 0xC00000FD，Python 无法用 try/except 捕获）。
    if len(smiles) > max_smiles_len:
        return None, ISSUE_SMILES_TOO_LONG

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, ISSUE_INVALID_SMILES
    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        return None, ISSUE_MOLECULE_TOO_LARGE

    # 多组分结构先隔离，留待决定如何处理水合物、盐等
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, ISSUE_MULTIPLE_FRAGMENTS

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), ""
    except Exception:  # noqa: BLE001 - 规范化失败属于数据问题，记录而不是中断整批
        return None, ISSUE_CANONICALIZATION_FAILED


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """校验列名、记录原始行号、清理字符串空白并把空串转为缺失值。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必须是 pandas.DataFrame")

    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 必须包含这些列：{list(REQUIRED_COLUMNS)}，缺少：{missing}")

    # 原始行号：CSV 第 1 行是表头，所以数据第一行记为 2。
    # 如果输入本身已带 source_row（例如重新清洗清洗结果），保留原值。
    if "source_row" not in df.columns:
        df["source_row"] = np.arange(len(df), dtype="int64") + 2

    # 保留转换前的标签，方便对照原始 CSV
    df["exp_raw"] = df["exp"]

    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    return df


def _append_rejected(
    rejected: list[pd.DataFrame],
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
) -> None:
    """把被移出的行追加到 rejected，并标注原因。"""
    if not mask.any():
        return
    bad = df.loc[mask].copy()
    bad["reason"] = reason
    rejected.append(bad)


def clean_dataframe(
    df: pd.DataFrame,
    *,
    min_exp: float = DEFAULT_MIN_EXP,
    max_exp: float = DEFAULT_MAX_EXP,
    max_smiles_len: int = DEFAULT_MAX_SMILES_LEN,
    max_heavy_atoms: int = DEFAULT_MAX_HEAVY_ATOMS,
) -> CleanResult:
    """清洗一个已经读入内存的数据框，返回 ``CleanResult``。

    参数
    ----
    df:
        至少包含 ``CMPD_CHEMBLID``、``exp``、``smiles`` 三列。
    min_exp, max_exp:
        exp 的核查范围（闭区间），默认 -10 ~ 10。
    max_smiles_len:
        SMILES 字符串长度上限，默认 1000，用于避免 RDKit 栈溢出。
    max_heavy_atoms:
        重原子数上限，默认 200。
    """
    if min_exp >= max_exp:
        raise ValueError(f"min_exp 必须小于 max_exp，当前为 {min_exp} >= {max_exp}")
    if max_smiles_len <= 0:
        raise ValueError("max_smiles_len 必须为正整数")
    if max_heavy_atoms <= 0:
        raise ValueError("max_heavy_atoms 必须为正整数")

    n_raw = len(df)
    work = _normalize_frame(df)
    rejected: list[pd.DataFrame] = []
    required = list(REQUIRED_COLUMNS)

    # 1) 完全重复的记录；去重时不能比较 source_row，否则每行都会被认为不同
    mask = work.duplicated(subset=required)
    _append_rejected(rejected, work, mask, REASON_DUPLICATE_RECORD)
    work = work.loc[~mask].copy()

    # 2) 关键字段缺失
    mask = work[required].isna().any(axis=1)
    _append_rejected(rejected, work, mask, REASON_MISSING_VALUE)
    work = work.loc[~mask].copy()

    # 3) 标签转数值，隔离非数字、NaN 和无穷值
    work["exp"] = pd.to_numeric(work["exp"], errors="coerce").astype("float64")
    mask = ~np.isfinite(work["exp"])
    _append_rejected(rejected, work, mask, REASON_INVALID_EXP)
    work = work.loc[~mask].copy()

    # 4) 本项目设定的核查范围，不是 logD 的通用合法范围
    mask = ~work["exp"].between(min_exp, max_exp)
    _append_rejected(rejected, work, mask, REASON_EXP_NEEDS_REVIEW)
    work = work.loc[~mask].copy()

    # 5) SMILES 结构检查，并生成保留立体信息的规范表达
    checked = work["smiles"].map(
        lambda smiles: check_smiles(
            smiles,
            max_smiles_len=max_smiles_len,
            max_heavy_atoms=max_heavy_atoms,
        )
    )
    work["canonical_smiles"] = checked.map(lambda item: item[0])
    work["structure_issue"] = checked.map(lambda item: item[1])
    mask = work["structure_issue"].ne("")
    _append_rejected(rejected, work, mask, REASON_STRUCTURE_NEEDS_REVIEW)
    work = work.loc[~mask].copy()

    # 6) 检查本数据集的 ID 格式
    mask = ~work["CMPD_CHEMBLID"].str.fullmatch(ID_PATTERN, na=False)
    _append_rejected(rejected, work, mask, REASON_INVALID_ID)
    work = work.loc[~mask].copy()

    # 7) 同一分子对应不同标签：整组隔离，不直接取平均
    label_counts = work.groupby("canonical_smiles")["exp"].transform("nunique")
    mask = label_counts.gt(1)
    _append_rejected(rejected, work, mask, REASON_CONFLICTING_LABELS)
    work = work.loc[~mask].copy()

    # 8) 同一分子且标签相同：保留一条
    mask = work.duplicated(subset=["canonical_smiles"])
    _append_rejected(rejected, work, mask, REASON_DUPLICATE_MOLECULE)
    work = work.loc[~mask].copy()

    clean = work.loc[:, [*required, "canonical_smiles", "source_row"]].copy()
    if rejected:
        review = pd.concat(rejected, ignore_index=True)
        review = review.reindex(columns=list(REVIEW_COLUMNS))
    else:
        review = pd.DataFrame(columns=list(REVIEW_COLUMNS))

    return CleanResult(clean=clean, review=review, n_raw=n_raw)


def clean_file(
    input_path: str | Path,
    clean_out: str | Path | None = None,
    review_out: str | Path | None = None,
    *,
    min_exp: float = DEFAULT_MIN_EXP,
    max_exp: float = DEFAULT_MAX_EXP,
    max_smiles_len: int = DEFAULT_MAX_SMILES_LEN,
    max_heavy_atoms: int = DEFAULT_MAX_HEAVY_ATOMS,
) -> CleanResult:
    """读取 CSV、清洗，并按需写出 clean/review 两个 CSV。"""
    df = load_raw(input_path)
    result = clean_dataframe(
        df,
        min_exp=min_exp,
        max_exp=max_exp,
        max_smiles_len=max_smiles_len,
        max_heavy_atoms=max_heavy_atoms,
    )

    if clean_out is not None:
        clean_path = Path(clean_out)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        result.clean.to_csv(clean_path, index=False, encoding="utf-8-sig")

    if review_out is not None:
        review_path = Path(review_out)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        result.review.to_csv(review_path, index=False, encoding="utf-8-sig")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="清洗 Lipophilicity 数据集（保持原有规则与移出原因）",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "Lipophilicity.csv",
        help="原始 CSV 路径（默认：脚本目录下的 Lipophilicity.csv）",
    )
    parser.add_argument(
        "--clean-out",
        type=Path,
        default=SCRIPT_DIR / "Lipophilicity_clean.csv",
        help="清洗结果输出路径",
    )
    parser.add_argument(
        "--review-out",
        type=Path,
        default=SCRIPT_DIR / "Lipophilicity_review.csv",
        help="移出记录输出路径",
    )
    parser.add_argument("--min-exp", type=float, default=DEFAULT_MIN_EXP, help="exp 下限（含）")
    parser.add_argument("--max-exp", type=float, default=DEFAULT_MAX_EXP, help="exp 上限（含）")
    parser.add_argument(
        "--max-smiles-len",
        type=int,
        default=DEFAULT_MAX_SMILES_LEN,
        help="SMILES 长度上限，超长分子会在解析前被移出（防止 RDKit 栈溢出）",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=DEFAULT_MAX_HEAVY_ATOMS,
        help="重原子数上限",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    parser.add_argument("--quiet", action="store_true", help="只输出结果，不打印明细")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.min_exp >= args.max_exp:
        parser.error("--min-exp 必须小于 --max-exp")
    if args.max_smiles_len <= 0 or args.max_heavy_atoms <= 0:
        parser.error("--max-smiles-len 和 --max-heavy-atoms 必须为正整数")

    # 无效 SMILES 的解析错误由 review 记录，不需要刷屏
    RDLogger.DisableLog("rdApp.*")

    try:
        result = clean_file(
            args.input,
            None if args.dry_run else args.clean_out,
            None if args.dry_run else args.review_out,
            min_exp=args.min_exp,
            max_exp=args.max_exp,
            max_smiles_len=args.max_smiles_len,
            max_heavy_atoms=args.max_heavy_atoms,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"原始 {result.n_raw} 条，保留 {result.n_clean} 条，移出 {result.n_removed} 条")
        for reason, count in result.reason_counts().items():
            print(f"  {reason:<24} {int(count)}")
        if not args.dry_run:
            print(f"清洗结果：{args.clean_out}")
            print(f"移出记录及原因：{args.review_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
