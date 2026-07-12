"""
数据预处理分析脚本 v2 - 深入检查
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\Lenovo\Desktop\Reasearch\trail_work\Lipophilicity.csv')

print('=' * 60)
print('1. exp 列类型问题 — 为什么是 object？')
print('=' * 60)

# 检查非数值的 exp 值
def is_numeric(val):
    if pd.isna(val):
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

exp_numeric_mask = df['exp'].apply(is_numeric)
print(f'非数值 exp 数量: {(~exp_numeric_mask).sum()}')
non_numeric = df[~exp_numeric_mask]
if len(non_numeric) > 0:
    print('\n非数值 exp 详情:')
    print(non_numeric.to_string())

# 看看哪些值导致类型为object
print('\n前20个 exp 值及其类型:')
for i in range(min(20, len(df))):
    val = df.iloc[i]['exp']
    print(f'  行{i}: "{val}" (type={type(val).__name__})')

print('\n' + '=' * 60)
print('2. 缺失值定位')
print('=' * 60)
print(f'exp 缺失行: {df[df["exp"].isna()].index.tolist()}')
print(f'smiles 缺失行: {df[df["smiles"].isna()].index.tolist()}')

# 显示缺失行
missing = df[df['exp'].isna() | df['smiles'].isna()]
print('\n缺失行详情:')
print(missing.to_string())

print('\n' + '=' * 60)
print('3. 重复行详情')
print('=' * 60)
print(f'完全重复行: {df.duplicated().sum()}')
dup_ids = df[df['CMPD_CHEMBLID'].duplicated(keep=False)]
dup_count = dup_ids.groupby('CMPD_CHEMBLID').size().value_counts().sort_index()
print(f'重复ID的出现次数分布:')
for cnt, num in dup_count.items():
    print(f'  出现{cnt}次的ID: {num}个')

# 展示几个重复的
print('\n重复示例 (CHEMBL17):')
print(dup_ids[dup_ids['CMPD_CHEMBLID'] == 'CHEMBL17'].to_string())

print('\n' + '=' * 60)
print('4. 现在将 exp 转为数值再统计')
print('=' * 60)
df_num = df[exp_numeric_mask].copy()
df_num['exp'] = df_num['exp'].astype(float)
print(df_num['exp'].describe())

Q1 = df_num['exp'].quantile(0.25)
Q3 = df_num['exp'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df_num[(df_num['exp'] < lower) | (df_num['exp'] > upper)]
print(f'\nIQR范围: [{lower:.2f}, {upper:.2f}]')
print(f'IQR离群值: {len(outliers)} 条')
print(f'极端值 (exp > 10 或 exp < -5): {((df_num["exp"] > 10) | (df_num["exp"] < -5)).sum()}')

print('\n' + '=' * 60)
print('5. exp 分布概览')
print('=' * 60)
bins = [-999, -5, -2, 0, 2, 5, 10, 999]
labels = ['< -5', '-5 ~ -2', '-2 ~ 0', '0 ~ 2', '2 ~ 5', '5 ~ 10', '> 10']
df_num['exp_bin'] = pd.cut(df_num['exp'], bins=bins, labels=labels)
print(df_num['exp_bin'].value_counts().sort_index())
