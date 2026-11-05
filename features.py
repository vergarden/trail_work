import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")

BASE_DIR = Path(__file__).resolve().parent

# 固定描述符顺序，保证训练/推理特征列一致
DESC_NAMES = [name for name, _ in Descriptors._descList]
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurize(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES：{smiles[:80]}")

    desc = Descriptors.CalcMolDescriptors(mol)
    d = np.array(
        [desc.get(name, np.nan) for name in DESC_NAMES],
        dtype=np.float64,
    )
    # 描述符里可能有 inf/nan；先替换，避免后续模型报错
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    fp = np.zeros((2048,), dtype=np.uint8)
    ConvertToNumpyArray(MORGAN.GetFingerprint(mol), fp)

    return np.concatenate([d.astype(np.float32), fp.astype(np.float32)])


def main() -> None:
    df = pd.read_csv(BASE_DIR / "Lipophilicity_clean.csv")

    X = np.stack([featurize(s) for s in df["canonical_smiles"]])
    ids = df["CMPD_CHEMBLID"].to_numpy()

    names = DESC_NAMES + [f"morgan_{i}" for i in range(2048)]

    assert X.shape == (len(df), len(names))
    assert np.isfinite(X).all()

    np.savez_compressed(
        BASE_DIR / "features.npz",
        X=X,
        ids=ids,
        names=np.array(names),
    )
    print("features:", X.shape)
    print("ids:", len(ids))


if __name__ == "__main__":
    main()