"""
测试推理接口：接收 test.csv → 清洗 → 特征提取 → 模型推理 → 对比报告
用法: python predict.py [test_csv_path]
默认读取项目根目录下的 test.csv
"""
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model_rf.pkl'
FEATURE_LIST_PATH = ROOT / 'feature_list.json'
DEFAULT_TEST = ROOT / 'test.csv'

# 描述符注册表（与 feature_engineering.py 保持一致）
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


def parse_exp(value):
    """将 exp 转为数值；不能转换或极端值返回 None。"""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == '':
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if abs(val) > 100:
        return None
    return val


def normalize_smiles(value):
    """清理 SMILES 字段。"""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == '' or text.lower() == 'nan':
        return None
    return text


def clean_dataframe(df):
    """对 test.csv 执行与训练时一致的清洗流水线。"""
    df = df.copy()
    df['CMPD_CHEMBLID'] = df['CMPD_CHEMBLID'].astype(str).str.strip()
    df['smiles'] = df['smiles'].apply(normalize_smiles)
    df['exp'] = df['exp'].apply(parse_exp)

    initial = len(df)
    df = df.dropna(subset=['exp', 'smiles']).copy()
    df = df[df['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)].copy()
    print(f'测试集清洗: {initial} → {len(df)} 行 (移除 {initial - len(df)} 条无效样本)')
    return df


def extract_features(df, feature_cols):
    """对清洗后的测试数据提取与训练时相同的特征。"""
    generator = GetMorganGenerator(radius=2, fpSize=256)
    desc_rows, morg_rows = [], []

    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            desc_rows.append({name: np.nan for name in DESCRIPTORS})
            morg_rows.append([0] * 256)
            continue
        desc = {name: float(func(mol)) if func(mol) is not None else np.nan
                for name, func in DESCRIPTORS.items()}
        fp = generator.GetFingerprint(mol)
        bits = [int(b) for b in fp.ToBitString()]
        desc_rows.append(desc)
        morg_rows.append(bits)

    desc_df = pd.DataFrame(desc_rows).fillna(0)
    morg_df = pd.DataFrame(morg_rows, columns=[f'morgan_{i}' for i in range(256)])
    feat_df = pd.concat([df.reset_index(drop=True), desc_df, morg_df], axis=1)

    # 确保列顺序与训练时的 feature_cols 一致
    available = [c for c in feature_cols if c in feat_df.columns]
    X = feat_df[available].values.astype(np.float32)
    return X, feat_df


def main():
    test_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEST
    if not test_path.exists():
        print(f'错误: 找不到测试文件 {test_path}')
        print(f'请将 test.csv 放到 {ROOT} 目录下，或指定路径: python predict.py <path>')
        sys.exit(1)

    # 1. 加载模型和特征列表
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    print(f'已加载模型: {MODEL_PATH}')
    print(f'特征数量: {len(feature_cols)}')

    # 2. 读取并清洗测试数据
    df = pd.read_csv(test_path)
    print(f'\n原始测试集: {df.shape[0]} 条样本')
    df_clean = clean_dataframe(df)

    # 3. 特征提取
    X_test, feat_df = extract_features(df_clean, feature_cols)
    y_true = df_clean['exp'].values.astype(np.float32)
    print(f'特征矩阵: {X_test.shape}')

    # 4. 推理
    y_pred = model.predict(X_test)

    # 5. 评估报告
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'\n{"="*50}')
    print(f'  测试推理报告')
    print(f'{"="*50}')
    print(f'  样本数:      {len(y_true)}')
    print(f'  RMSE:        {rmse:.4f}')
    print(f'  MAE:         {mae:.4f}')
    print(f'  R²:          {r2:.4f}')
    print(f'  目标均值:     {y_true.mean():.2f}')
    print(f'  预测均值:     {y_pred.mean():.2f}')
    print(f'{"="*50}')

    # 6. 保存详细预测结果
    output_df = df_clean[['CMPD_CHEMBLID', 'exp', 'smiles']].copy()
    output_df['predicted_logD'] = y_pred
    output_df['abs_error'] = np.abs(output_df['exp'] - output_df['predicted_logD'])
    output_path = ROOT / 'test_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f'\n详细预测结果已保存: {output_path}')


if __name__ == '__main__':
    main()
