"""
数据预处理分析脚本
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# 1. 读取数据
df = pd.read_csv(r'c:\Users\Lenovo\Desktop\Reasearch\trail_work\Lipophilicity.csv')
print('=' * 60)
print('1. 基本信息')
print('=' * 60)
print(f'形状: {df.shape}')
print(f'列名: {list(df.columns)}')
print(f'数据类型:\n{df.dtypes}')

# 2. 缺失值
print('\n' + '=' * 60)
print('2. 缺失值检查')
print('=' * 60)
print(df.isnull().sum())

# 3. 重复值
print('\n' + '=' * 60)
print('3. 重复值检查')
print('=' * 60)
print(f'完全重复行: {df.duplicated().sum()}')
print(f'CMPD_CHEMBLID 重复: {df["CMPD_CHEMBLID"].duplicated().sum()}')
dup_ids = df[df['CMPD_CHEMBLID'].duplicated(keep=False)]['CMPD_CHEMBLID'].unique()
if len(dup_ids) > 0:
    print(f'重复的ID列表: {list(dup_ids)[:10]}...' if len(dup_ids) > 10 else f'重复的ID列表: {list(dup_ids)}')

# 4. exp (logD) 统计
print('\n' + '=' * 60)
print('4. exp (logD) 统计')
print('=' * 60)
print(df['exp'].describe())
print(f'\nexp <= 0: {(df["exp"] <= 0).sum()} 条')
print(f'exp < 0:  {(df["exp"] < 0).sum()} 条')
print(f'exp == 0: {(df["exp"] == 0).sum()} 条')

# 离群值检查 (IQR法)
Q1 = df['exp'].quantile(0.25)
Q3 = df['exp'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['exp'] < lower) | (df['exp'] > upper)]
print(f'\nIQR 离群值范围: [{lower:.2f}, {upper:.2f}]')
print(f'离群值数量: {len(outliers)}')

# 5. SMILES 合法性检查
print('\n' + '=' * 60)
print('5. SMILES 合法性检查')
print('=' * 60)

def check_smiles(smi):
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == '':
        return False, '空值或非字符串'
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, 'RDKit无法解析'
    return True, '有效'

validity = df['smiles'].apply(check_smiles)
invalid_mask = ~validity.apply(lambda x: x[0])
print(f'无效 SMILES 数量: {invalid_mask.sum()}')
if invalid_mask.sum() > 0:
    print('\n无效 SMILES 详情:')
    invalid_df = df[invalid_mask].copy()
    invalid_df['reason'] = validity[invalid_mask].apply(lambda x: x[1])
    print(invalid_df[['CMPD_CHEMBLID', 'smiles', 'reason']].to_string())

# 6. SMILES 字符串统计
print('\n' + '=' * 60)
print('6. SMILES 字符串长度统计')
print('=' * 60)
df['smiles_len'] = df['smiles'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
print(df['smiles_len'].describe())

# 7. 检查 SMILES 中是否有异常字符
print('\n' + '=' * 60)
print('7. SMILES 特殊字符检查')
print('=' * 60)
# 检查包含非标准SMILES字符的行
import re
valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]=#-+/.@\\%')
for i, smi in enumerate(df['smiles']):
    unusual = set(str(smi)) - valid_chars
    if unusual:
        print(f'行{i}: ID={df.iloc[i]["CMPD_CHEMBLID"]}, 异常字符: {unusual}, SMILES前80字: {str(smi)[:80]}')
        if i > 20:  # 只显示前20个
            print('...(更多省略)')
            break
