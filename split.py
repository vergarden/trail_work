"""Bemis-Murcko 骨架划分（只负责划分，不负责数据清洗）。

职责边界：
- dataclean.py：处理缺失值、非法 SMILES、重复分子、标签异常；
- split.py：读取已清洗的数据，计算骨架，按骨架分组并划分，
  校验无泄漏后写出。

因此本脚本不会删除、修复或去重任何一行；输入不合法时直接报错，
并提示先运行 dataclean.py。

默认用法：
    python split.py

其它数据集（列名不同）：
    python split.py --input other_clean.csv \
        --id-col compound_id --smiles-col canonical_smiles --target-col logD \
        --output-csv other_split.csv --output-json other_split.json

库调用：
    from split import SplitConfig, run_split

    result = run_split(SplitConfig(seed=42))
    print(result.summary.counts)
    print(result.frames()["train"].shape)
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

SCRIPT_DIR = Path(__file__).resolve().parent

SplitName = Literal["train", "valid", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "valid", "test")

# 自动识别列名时使用的候选列表，按优先级从高到低
ID_CANDIDATES = ("CMPD_CHEMBLID", "id", "ID", "compound_id", "mol_id", "name")
SMILES_CANDIDATES = ("canonical_smiles", "smiles", "SMILES", "smi", "structure")
TARGET_CANDIDATES = ("exp", "target", "y", "label", "logD", "logd", "value")


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


@dataclass
class SplitConfig:
    """一次骨架划分的全部配置。"""

    input_path: Path = SCRIPT_DIR / "Lipophilicity_clean.csv"
    output_csv: Path = SCRIPT_DIR / "split.csv"
    output_json: Path = SCRIPT_DIR / "split.json"

    # 列名；None 表示自动识别
    id_col: str | None = None
    smiles_col: str | None = None
    target_col: str | None = None

    seed: int = 42
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    include_chirality: bool = False
    max_smiles_len: int = 1000

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_csv = Path(self.output_csv)
        self.output_json = Path(self.output_json)

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio 必须在 (0, 1) 之间")
        if not 0 < self.valid_ratio < 1:
            raise ValueError("valid_ratio 必须在 (0, 1) 之间")
        if self.train_ratio + self.valid_ratio >= 1:
            raise ValueError("train_ratio + valid_ratio 必须小于 1")
        if self.seed < 0:
            raise ValueError("seed 必须是非负整数")
        if self.max_smiles_len <= 0:
            raise ValueError("max_smiles_len 必须为正整数")


@dataclass(frozen=True)
class ResolvedColumns:
    """实际使用的列名；输入没有 ID 时 id_col 为 __row_id__。"""

    id_col: str
    smiles_col: str
    target_col: str | None


@dataclass(frozen=True)
class ScaffoldGroup:
    """一个骨架组；组内所有分子必须进入同一个集合。"""

    key: str
    scaffold: str
    indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class SplitAssignment:
    """三个集合的行索引。"""

    train: tuple[int, ...]
    valid: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[SplitName, tuple[int, ...]]:
        return {"train": self.train, "valid": self.valid, "test": self.test}

    def counts(self) -> dict[SplitName, int]:
        return {name: len(indices) for name, indices in self.as_dict().items()}


@dataclass(frozen=True)
class SplitChecks:
    """三类泄漏检查；所有值都应为 0。"""

    smiles_overlap: dict[str, int]
    scaffold_overlap: dict[str, int]
    group_overlap: dict[str, int]

    @property
    def is_clean(self) -> bool:
        return not (
            any(self.smiles_overlap.values())
            or any(self.scaffold_overlap.values())
            or any(self.group_overlap.values())
        )


@dataclass(frozen=True)
class SplitSummary:
    """划分结果统计。"""

    n_rows: int
    counts: dict[SplitName, int]
    scaffold_counts: dict[SplitName, int]
    acyclic_molecules: int
    checks: SplitChecks
    label_stats: dict[SplitName, dict[str, float]]


@dataclass(frozen=True)
class SplitResult:
    """run_split 的返回值；dataframe 已包含 split 列。"""

    dataframe: pd.DataFrame
    assignment: SplitAssignment
    summary: SplitSummary
    resolved: ResolvedColumns
    split_csv: Path
    split_json: Path

    def frames(self) -> dict[SplitName, pd.DataFrame]:
        """按 split 列拆成三个子表，方便直接交给训练脚本。"""
        if "split" not in self.dataframe.columns:
            raise ValueError("dataframe 缺少 split 列")
        return {
            name: self.dataframe.loc[
                self.dataframe["split"] == name
            ].reset_index(drop=True)
            for name in SPLIT_NAMES
        }


# ---------------------------------------------------------------------------
# 读取与列名解析
# ---------------------------------------------------------------------------


def load_raw(path: Path) -> pd.DataFrame:
    """读取已清洗的 CSV；只去掉列名两端空白，不做任何行级清洗。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        raise ValueError(f"输入数据为空：{path}")
    return df


def resolve_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    label: str,
    *,
    required: bool,
) -> str | None:
    """优先使用显式列名，否则按候选列表自动识别。"""
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"指定的 {label} 列不存在：{explicit}")
        return explicit

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            f"找不到 {label} 列，请用参数显式指定；当前列：{list(df.columns)}"
        )
    return None


def resolve_columns(df: pd.DataFrame, config: SplitConfig) -> ResolvedColumns:
    """确定 ID / SMILES / target 列；没有 ID 时生成 __row_id__。"""
    id_col = resolve_column(
        df, config.id_col, ID_CANDIDATES, "ID", required=False
    )
    if id_col is None:
        id_col = "__row_id__"
        # 仅仅生成可追溯的行 ID，不修改或删除任何化学数据
        df[id_col] = [f"row_{index:06d}" for index in range(1, len(df) + 1)]

    smiles_col = resolve_column(
        df, config.smiles_col, SMILES_CANDIDATES, "SMILES", required=True
    )
    assert smiles_col is not None

    target_col = resolve_column(
        df, config.target_col, TARGET_CANDIDATES, "target", required=False
    )

    return ResolvedColumns(
        id_col=id_col,
        smiles_col=smiles_col,
        target_col=target_col,
    )


# ---------------------------------------------------------------------------
# 输入校验：清洗由 dataclean.py 负责，这里只检查和报错
# ---------------------------------------------------------------------------


def validate_input(
    df: pd.DataFrame,
    resolved: ResolvedColumns,
    config: SplitConfig,
) -> None:
    """检查 split.py 依赖的前提；发现脏数据直接报错，不自动修复。"""
    smiles = df[resolved.smiles_col]

    # 缺失 / 空字符串：dataclean.py 应该已经处理
    missing_mask = smiles.isna() | smiles.astype("string").str.strip().eq("")
    if missing_mask.any():
        examples = (
            df.loc[missing_mask, resolved.id_col].astype(str).head(5).tolist()
        )
        raise ValueError(
            f"{int(missing_mask.sum())} 行缺少 {resolved.smiles_col}"
            f"（示例 ID：{examples}）；请先运行 dataclean.py"
        )

    # 重复分子：同一个 canonical SMILES 只能保留一条
    duplicate_mask = smiles.duplicated(keep=False)
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, resolved.id_col].astype(str).head(5).tolist()
        )
        raise ValueError(
            f"{resolved.smiles_col} 存在重复分子"
            f"（示例 ID：{examples}）；请先运行 dataclean.py 去重"
        )

    # 超长 SMILES 必须在 RDKit 解析前拦截，否则可能栈溢出
    too_long = smiles.astype("string").str.len() > config.max_smiles_len
    if too_long.any():
        examples = df.loc[too_long, resolved.id_col].astype(str).head(5).tolist()
        raise ValueError(
            f"{int(too_long.sum())} 行 SMILES 超过 {config.max_smiles_len} 字符"
            f"（示例 ID：{examples}）；请在 dataclean.py 中过滤"
        )


# ---------------------------------------------------------------------------
# 骨架计算与分组
# ---------------------------------------------------------------------------


def compute_scaffold(
    smiles: str,
    *,
    include_chirality: bool = False,
    max_smiles_len: int = 1000,
) -> str:
    """计算 Bemis-Murcko 骨架；无环分子返回空字符串。

    这里不负责 canonical 化、去重或过滤；这些是 dataclean.py 的职责。
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("空 SMILES")

    smiles = smiles.strip()
    if len(smiles) > max_smiles_len:
        raise ValueError(f"SMILES 超过 {max_smiles_len} 字符")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("RDKit 无法解析")

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality,
    )


def annotate_scaffolds(
    df: pd.DataFrame,
    resolved: ResolvedColumns,
    config: SplitConfig,
) -> pd.DataFrame:
    """给数据框增加 scaffold 和 group_key 两列，不增删行。"""
    out = df.copy()
    scaffold_values: list[str] = []

    for position, (index, value) in enumerate(out[resolved.smiles_col].items()):
        try:
            scaffold = compute_scaffold(
                value,
                include_chirality=config.include_chirality,
                max_smiles_len=config.max_smiles_len,
            )
        except (ValueError, RuntimeError) as exc:
            source_row = (
                out.at[index, "source_row"]
                if "source_row" in out.columns
                else position + 2
            )
            compound_id = out.at[index, resolved.id_col]
            raise ValueError(
                f"source_row={source_row}, ID={compound_id}: {exc}；"
                "请先运行 dataclean.py 处理该分子"
            ) from exc
        scaffold_values.append(scaffold)

    out["scaffold"] = scaffold_values

    # 无环分子的骨架为空：用 ID + 行号保证它们各自成组，
    # 否则所有无环分子会被当成同一个“骨架”
    out["group_key"] = [
        scaffold if scaffold else f"__ACYCLIC__::{compound_id}::{position}"
        for position, (scaffold, compound_id) in enumerate(
            zip(out["scaffold"], out[resolved.id_col])
        )
    ]
    return out


def build_groups(df: pd.DataFrame) -> list[ScaffoldGroup]:
    """按 group_key 聚合为骨架组。"""
    if "group_key" not in df.columns:
        raise ValueError("请先调用 annotate_scaffolds")

    groups: list[ScaffoldGroup] = []
    for key, part in df.groupby("group_key", sort=False):
        groups.append(
            ScaffoldGroup(
                key=str(key),
                scaffold=str(part["scaffold"].iloc[0]),
                indices=tuple(int(index) for index in part.index),
            )
        )
    return groups


# ---------------------------------------------------------------------------
# 划分、检查与统计
# ---------------------------------------------------------------------------


def assign_splits(
    groups: list[ScaffoldGroup],
    n_rows: int,
    config: SplitConfig,
) -> SplitAssignment:
    """按组大小降序贪心分配；保证同一骨架只进一个集合。"""
    shuffled = list(groups)

    # seed 只影响同样大小的组之间的顺序；排序后大组优先
    random.Random(config.seed).shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda group: group.size, reverse=True)

    n_train = int(round(n_rows * config.train_ratio))
    n_valid = int(round(n_rows * config.valid_ratio))

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []

    for group in ordered:
        if len(train) + group.size <= n_train:
            train.extend(group.indices)
        elif len(valid) + group.size <= n_valid:
            valid.extend(group.indices)
        else:
            test.extend(group.indices)

    # 覆盖 + 互斥检查：防止漏行或重复分配
    all_indices = train + valid + test
    if len(all_indices) != n_rows or len(set(all_indices)) != n_rows:
        raise RuntimeError(
            "划分索引不完整或重复："
            f"expected={n_rows}, actual={len(all_indices)}, "
            f"unique={len(set(all_indices))}"
        )
    if not train or not valid or not test:
        raise RuntimeError(
            f"划分出现空集合：train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    return SplitAssignment(
        train=tuple(train),
        valid=tuple(valid),
        test=tuple(test),
    )


def validate_splits(
    df: pd.DataFrame,
    assignment: SplitAssignment,
    smiles_col: str,
) -> SplitChecks:
    """检查结构列、骨架、group_key 是否跨集合。"""
    split_map = assignment.as_dict()
    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))

    smiles_sets = {
        name: set(df.loc[list(indices), smiles_col])
        for name, indices in split_map.items()
    }
    scaffold_sets = {
        name: set(df.loc[list(indices), "scaffold"]) - {""}
        for name, indices in split_map.items()
    }
    group_sets = {
        name: set(df.loc[list(indices), "group_key"])
        for name, indices in split_map.items()
    }

    return SplitChecks(
        smiles_overlap={
            f"{a}-{b}": len(smiles_sets[a] & smiles_sets[b]) for a, b in pairs
        },
        scaffold_overlap={
            f"{a}-{b}": len(scaffold_sets[a] & scaffold_sets[b]) for a, b in pairs
        },
        group_overlap={
            f"{a}-{b}": len(group_sets[a] & group_sets[b]) for a, b in pairs
        },
    )


def summarize(
    df: pd.DataFrame,
    assignment: SplitAssignment,
    resolved: ResolvedColumns,
) -> SplitSummary:
    """统计数量、骨架数、标签分布和泄漏检查。"""
    split_map = assignment.as_dict()

    counts: dict[SplitName, int] = {
        name: len(indices) for name, indices in split_map.items()
    }
    scaffold_counts: dict[SplitName, int] = {
        name: len(set(df.loc[list(indices), "scaffold"]) - {""})
        for name, indices in split_map.items()
    }

    label_stats: dict[SplitName, dict[str, float]] = {}
    if resolved.target_col is not None:
        target = pd.to_numeric(df[resolved.target_col], errors="coerce")
        for name, indices in split_map.items():
            values = target.loc[list(indices)].dropna()
            if len(values):
                label_stats[name] = {
                    "count": int(len(values)),
                    "mean": round(float(values.mean()), 4),
                    "std": round(float(values.std()), 4) if len(values) > 1 else 0.0,
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                }

    return SplitSummary(
        n_rows=len(df),
        counts=counts,
        scaffold_counts=scaffold_counts,
        acyclic_molecules=int((df["scaffold"] == "").sum()),
        checks=validate_splits(df, assignment, resolved.smiles_col),
        label_stats=label_stats,
    )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def build_split_dataframe(
    df: pd.DataFrame,
    assignment: SplitAssignment,
    resolved: ResolvedColumns,
) -> pd.DataFrame:
    """生成带 split 列的最终表；只保留划分需要的列。"""
    out = df.copy()
    out["split"] = pd.NA

    for name, indices in assignment.as_dict().items():
        out.loc[list(indices), "split"] = name

    if out["split"].isna().any():
        raise RuntimeError("存在没有 split 标签的行")

    columns = [resolved.id_col, resolved.smiles_col, "scaffold"]
    if resolved.target_col is not None:
        columns.append(resolved.target_col)
    if "source_row" in out.columns:
        columns.append("source_row")
    columns.append("split")

    out = out[columns].copy()
    out["split"] = pd.Categorical(
        out["split"],
        categories=list(SPLIT_NAMES),
        ordered=True,
    )
    # 按 train -> valid -> test 排序，每个集合内部保持原始顺序
    out = out.sort_values("split", kind="stable").reset_index(drop=True)
    out["split"] = out["split"].astype(str)
    return out


def write_summary(
    summary: SplitSummary,
    config: SplitConfig,
    resolved: ResolvedColumns,
    path: Path,
) -> dict[str, object]:
    """写出可审计的 split.json。"""
    payload: dict[str, object] = {
        "input": str(config.input_path),
        "seed": config.seed,
        "include_chirality": config.include_chirality,
        "max_smiles_len": config.max_smiles_len,
        "ratios": {
            "train": config.train_ratio,
            "valid": config.valid_ratio,
            "test": round(1 - config.train_ratio - config.valid_ratio, 4),
        },
        "columns": {
            "id": resolved.id_col,
            "smiles": resolved.smiles_col,
            "target": resolved.target_col,
        },
        "n_rows": summary.n_rows,
        "counts": summary.counts,
        "scaffold_counts": summary.scaffold_counts,
        "acyclic_molecules": summary.acyclic_molecules,
        "checks": {
            "smiles_overlap": summary.checks.smiles_overlap,
            "scaffold_overlap": summary.checks.scaffold_overlap,
            "group_overlap": summary.checks.group_overlap,
        },
        "label_stats": summary.label_stats,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


# ---------------------------------------------------------------------------
# 高层入口
# ---------------------------------------------------------------------------


def run_split(config: SplitConfig, *, write: bool = True) -> SplitResult:
    """读取 -> 校验 -> 算骨架 -> 分组 -> 划分 -> 校验 -> 写出。"""
    RDLogger.DisableLog("rdApp.*")

    raw = load_raw(config.input_path)

    # 保留原始 CSV 行号，方便回溯；已有 source_row 时不覆盖
    if "source_row" not in raw.columns:
        raw["source_row"] = range(2, len(raw) + 2)

    resolved = resolve_columns(raw, config)

    # 清洗是 dataclean.py 的职责；split.py 只校验并报错
    validate_input(raw, resolved, config)

    annotated = annotate_scaffolds(raw, resolved, config)
    groups = build_groups(annotated)
    assignment = assign_splits(groups, len(annotated), config)
    summary = summarize(annotated, assignment, resolved)

    if not summary.checks.is_clean:
        raise RuntimeError(
            "划分出现泄漏，已停止写出："
            f"scaffold={summary.checks.scaffold_overlap}, "
            f"smiles={summary.checks.smiles_overlap}, "
            f"group={summary.checks.group_overlap}"
        )

    split_df = build_split_dataframe(annotated, assignment, resolved)

    if write:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(config.output_csv, index=False, encoding="utf-8-sig")
        write_summary(summary, config, resolved, config.output_json)

    return SplitResult(
        dataframe=split_df,
        assignment=assignment,
        summary=summary,
        resolved=resolved,
        split_csv=config.output_csv,
        split_json=config.output_json,
    )


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 Bemis-Murcko 骨架划分已清洗的数据集（默认 Lipophilicity）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "Lipophilicity_clean.csv",
        help="输入 clean CSV（必须已经由 dataclean.py 清洗过）",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "split.csv",
        help="输出 split.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "split.json",
        help="输出 split.json",
    )
    parser.add_argument("--id-col", default=None, help="ID 列名（默认自动识别）")
    parser.add_argument(
        "--smiles-col", default=None, help="SMILES 列名（默认自动识别）"
    )
    parser.add_argument(
        "--target-col", default=None, help="标签列名（默认自动识别，可无）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--valid", type=float, default=0.1, help="验证集比例")
    parser.add_argument(
        "--include-chirality",
        action="store_true",
        help="骨架计算保留立体化学（默认忽略）",
    )
    parser.add_argument(
        "--max-smiles-len",
        type=int,
        default=1000,
        help="SMILES 长度上限；超过会报错，不会自动剔除（请在 dataclean.py 中处理）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = SplitConfig(
            input_path=args.input,
            output_csv=args.output_csv,
            output_json=args.output_json,
            id_col=args.id_col,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            seed=args.seed,
            train_ratio=args.train,
            valid_ratio=args.valid,
            include_chirality=args.include_chirality,
            max_smiles_len=args.max_smiles_len,
        )
        result = run_split(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"错误：{exc}") from exc

    summary = result.summary
    total = summary.n_rows

    print(f"总分子数 {total}，无环分子 {summary.acyclic_molecules}")
    for name in SPLIT_NAMES:
        count = summary.counts[name]
        print(
            f"  {name:5s} {count:5d} ({count / total:5.1%})  "
            f"scaffolds={summary.scaffold_counts[name]}"
        )

    print(
        "重叠检查：",
        {
            "smiles": summary.checks.smiles_overlap,
            "scaffold": summary.checks.scaffold_overlap,
            "group": summary.checks.group_overlap,
        },
    )
    print(f"已写出：{result.split_csv}")
    print(f"已写出：{result.split_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
