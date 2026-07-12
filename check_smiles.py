"""SMILES合法性检查 - 带进度"""
import pandas as pd
import sys
from rdkit import Chem

df = pd.read_csv(r'c:\Users\Lenovo\Desktop\Reasearch\trail_work\Lipophilicity.csv')
n = len(df)

invalid = []
for i, row in df.iterrows():
    if i % 500 == 0:
        print(f'进度: {i}/{n}', flush=True)
    smi = row['smiles']
    if pd.isna(smi):
        invalid.append((i, row['CMPD_CHEMBLID'], 'NaN'))
    elif Chem.MolFromSmiles(str(smi)) is None:
        invalid.append((i, row['CMPD_CHEMBLID'], str(smi)[:100]))

print(f'\n总行数: {n}', flush=True)
print(f'无效SMILES: {len(invalid)}', flush=True)
for item in invalid:
    print(f'  行{item[0]}: {item[1]}', flush=True)
