import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'Lipophilicity_cleaned.csv'
OUTPUT = ROOT / 'Lipophilicity_features.csv'

# 选择一组常用的分子描述符，覆盖分子大小、极性、疏水性、环结构等信息
DESCRIPTORS = {
    'MolWt': Descriptors.MolWt,
    'ExactMolWt': Descriptors.ExactMolWt,
    'LogP': Descriptors.MolLogP,
    'TPSA': Descriptors.TPSA,
    'NumHeavyAtoms': Descriptors.HeavyAtomCount,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'RingCount': Descriptors.RingCount,
    'FractionCSP3': Descriptors.FractionCSP3,
}


def compute_descriptor_features(mol):
    """为单个 RDKit 分子计算一组结构描述符。"""
    feature_dict = {}
    for name, func in DESCRIPTORS.items():
        try:
            feature_dict[name] = float(func(mol))
        except Exception:
            feature_dict[name] = np.nan
    return feature_dict


def compute_morgan_features(mol, generator):
    """为已解析的 RDKit 分子生成 Morgan 指纹位向量。"""
    fp = generator.GetFingerprint(mol)
    return [int(bit) for bit in fp.ToBitString()]


def main():
    df = pd.read_csv(INPUT)
    print(f'读取清洗后数据，形状: {df.shape}')

    generator = GetMorganGenerator(radius=2, fpSize=256)
    descriptor_rows = []
    morgan_rows = []

    for i, smiles in enumerate(df['smiles']):
        if i % 500 == 0:
            print(f'进度: {i}/{len(df)}', flush=True)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            descriptor_rows.append({name: np.nan for name in DESCRIPTORS})
            morgan_rows.append([0] * 256)
            continue

        descriptor_rows.append(compute_descriptor_features(mol))
        morgan_rows.append(compute_morgan_features(mol, generator))

    print(f'进度: {len(df)}/{len(df)}', flush=True)

    desc_df = pd.DataFrame(descriptor_rows)
    morgan_df = pd.DataFrame(morgan_rows, columns=[f'morgan_{i}' for i in range(256)])

    desc_df = desc_df.fillna(desc_df.median(numeric_only=True))
    morgan_df = morgan_df.fillna(0)

    feature_df = pd.concat([df.reset_index(drop=True), desc_df, morgan_df], axis=1)
    feature_df.to_csv(OUTPUT, index=False)

    print(f'特征维度: {feature_df.shape[1]} 列')
    print(f'已保存到: {OUTPUT}')
    print(f'目标统计: mean={feature_df["exp"].mean():.2f}, std={feature_df["exp"].std():.2f}, '
          f'min={feature_df["exp"].min():.2f}, max={feature_df["exp"].max():.2f}')
    print(f'描述符列: {list(DESCRIPTORS.keys())}')
    print(f'Morgan 指纹列: 256 位')


if __name__ == '__main__':
    main()
