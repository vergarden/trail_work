import pandas as pd
from pathlib import Path
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'Lipophilicity.csv'
OUTPUT = ROOT / 'Lipophilicity_cleaned.csv'


def parse_exp(value):
    """将 exp 转换为数值类型；无法转换的值返回 None；过滤极端异常值。"""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == '':
        return None

    try:
        val = float(text)
    except ValueError:
        return None

    # 过滤明显异常的哨兵值 (如 999999) 和极端离群值
    if abs(val) > 100:
        return None

    return val


def normalize_smiles(value):
    """清理 SMILES 字段，空值或空白字符串统一转为 None。"""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == '' or text.lower() == 'nan':
        return None

    return text


def is_valid_smiles(value):
    """检查 SMILES 是否能被 RDKit 成功解析。"""
    if value is None:
        return False

    try:
        return Chem.MolFromSmiles(value) is not None
    except Exception:
        return False


def main():
    """执行数据清洗，生成可用于建模的干净数据集。"""
    df = pd.read_csv(INPUT)
    print(f'原始数据形状: {df.shape}')

    df = df.copy()
    df['CMPD_CHEMBLID'] = df['CMPD_CHEMBLID'].astype(str).str.strip()
    df['smiles'] = df['smiles'].apply(normalize_smiles)
    df['exp'] = df['exp'].apply(parse_exp)

    initial_rows = len(df)

    # 1. 删除标签或输入缺失的样本
    df = df.dropna(subset=['exp', 'smiles']).copy()
    missing_rows_removed = initial_rows - len(df)

    # 2. 删除无效 SMILES
    invalid_smiles_mask = ~df['smiles'].apply(is_valid_smiles)
    df = df.loc[~invalid_smiles_mask].copy()
    invalid_smiles_removed = int(invalid_smiles_mask.sum())

    # 3. 删除完全重复样本
    full_row_duplicates_mask = df.duplicated(keep='first')
    df = df.loc[~full_row_duplicates_mask].copy()
    full_row_duplicates_removed = int(full_row_duplicates_mask.sum())

    # 4. 删除相同 CHEMBL ID 的重复条目，只保留第一条
    compound_id_duplicates_mask = df['CMPD_CHEMBLID'].duplicated(keep='first')
    df = df.loc[~compound_id_duplicates_mask].copy()
    compound_id_duplicates_removed = int(compound_id_duplicates_mask.sum())

    df.to_csv(OUTPUT, index=False)

    print(f'清洗后数据形状: {df.shape}')
    print(f'共删除行数: {initial_rows - len(df)}')
    print(f'  - 缺失 exp / smiles 的样本: {missing_rows_removed}')
    print(f'  - 无效 SMILES 的样本: {invalid_smiles_removed}')
    print(f'  - 完全重复样本: {full_row_duplicates_removed}')
    print(f'  - 重复 CHEMBL ID 的样本: {compound_id_duplicates_removed}')
    print(f'已保存到: {OUTPUT}')


if __name__ == '__main__':
    main()
