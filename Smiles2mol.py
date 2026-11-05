

#将smiles式列转成分子对象
def smiles2mol(csv_path = None, smiles = "canonical_smiles"):
    from rdkit import Chem
    import pandas as pd
    import pathlib as pl


    ##获取smiles式位置
    #如果没有额外输入指定的csv,就自动读取该目录下由dataclean.py生成的清洗文件
    script_dir = pl.Path(__file__).parent
    if csv_path is None:
        csv_path = pl.Path(__file__).resolve().parent / "Lipophilicity_clean.csv"


    df = pd.read_csv(csv_path)
    #尝试拉取smiles列
    if smiles not in df.columns:
        raise ValueError(f"缺失smiles列:{smiles},请检查输入")   
    
    ####下列代码为gpt补足  
    def parse_smiles(value):
        # 空值或空字符串无法转成有效分子
        if not isinstance(value, str) or not value.strip():
            return None

        mol = Chem.MolFromSmiles(value.strip())

        if mol is None or mol.GetNumAtoms() == 0:
            return None

        return mol

    # 保留原来的信息和行顺序，只增加分子对象列
    df["ROMol"] = df[smiles].map(parse_smiles)

    return df


if __name__ == "__main__":
    df = smiles2mol()

    success = int(df["ROMol"].notna().sum())
    print(
        f"共 {len(df)} 条，转换成功 {success} 条，"
        f"失败 {len(df) - success} 条"
    )
   ####gpt补足到此位置


    #转化为分子对象，为什么这玩意会报错？？？？？
    #Chem.PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles',molCol="ROMol")
    #df["molwt"] = df["ROMol"].apply(lambda x: Chem.Descriptors.MolWt(x) if x else None)


    







