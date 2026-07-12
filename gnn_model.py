"""
GNN 模型：纯 PyTorch 手写消息传递 → 分子图 → logD 预测
无需 PyG / DGL，零额外安装
"""
import sys
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'Lipophilicity_cleaned.csv'
MODEL_PATH = ROOT / 'model_gnn.pkl'             # 保存 state_dict
FEATURE_SAVE = ROOT / 'gnn_features.json'       # 保存原子特征维度信息

# ==================== 原子特征编码 ====================
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Other']
HYBRID_TYPES = ['SP', 'SP2', 'SP3', 'Other']
DEGREE_BINS = [0, 1, 2, 3, 4]  # 度 ≤4


def atom_features(atom):
    """将单个 RDKit 原子编码为固定长度向量。"""
    feats = []

    # 原子类型 one-hot
    sym = atom.GetSymbol()
    for t in ATOM_TYPES:
        feats.append(1.0 if sym == t else 0.0)

    # 杂化 one-hot
    hyb = str(atom.GetHybridization())
    for h in HYBRID_TYPES:
        feats.append(1.0 if h in hyb else 0.0)

    # 度 (one-hot)
    deg = atom.GetDegree()
    for d in DEGREE_BINS:
        feats.append(1.0 if deg == d else 0.0)

    # 连续特征
    feats.append(float(atom.GetFormalCharge()))
    feats.append(float(atom.GetNumRadicalElectrons()))
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    feats.append(float(atom.GetAtomicNum()) / 100.0)  # 归一化

    return np.array(feats, dtype=np.float32)


def smiles_to_graph(smi, max_atoms=60):
    """SMILES → 邻接矩阵 + 节点特征矩阵。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None, None

    atoms = mol.GetAtoms()
    n = len(atoms)
    if n > max_atoms:
        return None, None, None, None  # 跳过大分子

    # 节点特征
    node_feats = np.zeros((max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(atoms):
        node_feats[i] = atom_features(atom)

    # 邻接矩阵 (含自环 + 边类型)
    # 自环=1, 单键=2, 双键=3, 芳香键=4, 三键=5
    adj = np.eye(max_atoms, dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE:
                w = 2.0
            elif bt == Chem.BondType.DOUBLE:
                w = 3.0
            elif bt == Chem.BondType.AROMATIC:
                w = 4.0
            elif bt == Chem.BondType.TRIPLE:
                w = 5.0
            else:
                w = 1.0
            adj[i, j] = w
            adj[j, i] = w

    # mask: 哪些位置是真正有原子的
    mask = np.zeros(max_atoms, dtype=np.float32)
    mask[:n] = 1.0

    return node_feats, adj, mask, n


# 原子特征维度 = 10(类型) + 4(杂化) + 5(度) + 5(连续) = 24
ATOM_FEAT_DIM = len(ATOM_TYPES) + len(HYBRID_TYPES) + len(DEGREE_BINS) + 5


# ==================== GNN 层 ====================
class GCNLayer(nn.Module):
    """单层简化图卷积：H' = ReLU(A @ H @ W)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, adj):
        """H: (B, N, d_in), adj: (B, N, N)"""
        H = torch.bmm(adj, H)         # 邻居聚合
        H = self.W(H)                 # 线性变换
        return F.relu(H)


class MolecularGNN(nn.Module):
    """两层图卷积 + 全局平均池化 → MLP → logD"""
    def __init__(self, atom_dim=ATOM_FEAT_DIM, hidden=64, out_dim=1):
        super().__init__()
        self.conv1 = GCNLayer(atom_dim, hidden)
        self.conv2 = GCNLayer(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, out_dim),
        )

    def forward(self, node_feats, adj, mask=None):
        """node_feats: (B, N, d), adj: (B, N, N), mask: (B, N)"""
        H = self.conv1(node_feats, adj)
        H = self.conv2(H, adj)

        # 全局平均池化（只对有原子的位置取平均）
        if mask is not None:
            H = H * mask.unsqueeze(-1)
            graph_vec = H.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            graph_vec = H.mean(dim=1)

        return self.mlp(graph_vec).squeeze(-1)


# ==================== 数据集 ====================
class MolDataset(Dataset):
    def __init__(self, df, max_atoms=100):
        self.graphs = []
        self.y = []
        for _, row in df.iterrows():
            result = smiles_to_graph(row['smiles'], max_atoms)
            if result[0] is None:
                continue
            feats, adj, mask, n = result
            self.graphs.append((feats, adj, mask))
            self.y.append(float(row['exp']))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feats, adj, mask = self.graphs[idx]
        return (
            torch.tensor(feats),
            torch.tensor(adj),
            torch.tensor(mask),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def collate_fn(batch):
    fs, ads, ms, ys = zip(*batch)
    return torch.stack(fs), torch.stack(ads), torch.stack(ms), torch.stack(ys)


# ==================== 训练工具 ====================
def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for feats, adj, mask, y in loader:
        feats, adj, mask, y = feats.to(device), adj.to(device), mask.to(device), y.to(device)
        pred = model(feats, adj, mask)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
    return np.sqrt(total_loss / len(loader.dataset))


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    model.eval()
    preds, truths = [], []
    for feats, adj, mask, y in loader:
        feats, adj, mask = feats.to(device), adj.to(device), mask.to(device)
        p = model(feats, adj, mask)
        preds.append(p.cpu().numpy())
        truths.append(y.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    rmse = np.sqrt(mean_squared_error(truths, preds))
    mae = mean_absolute_error(truths, preds)
    r2 = r2_score(truths, preds)
    return rmse, mae, r2


# ==================== 主流程 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    df = pd.read_csv(INPUT)
    print(f'读取清洗数据: {df.shape[0]} 条')

    # 构建图数据集（可能因为 max_atoms 过滤少量大分子）
    dataset = MolDataset(df, max_atoms=60)
    print(f'构建图数据集: {len(dataset)} 条 (过滤 {len(df) - len(dataset)} 条超长分子)')

    # 获取目标值序列用于划分
    y_all = np.array([dataset[i][3].item() for i in range(len(dataset))])
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn)

    print(f'训练: {len(train_ds)} | 验证: {len(val_ds)} | 测试: {len(test_ds)}')

    # 模型
    model = MolecularGNN(atom_dim=ATOM_FEAT_DIM, hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    best_val_rmse = float('inf')
    best_state = None

    for epoch in range(1, 81):
        train_rmse = train_epoch(model, train_loader, opt, device)
        val_rmse, val_mae, val_r2 = evaluate_gnn(model, val_loader, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}, Val R²={val_r2:.4f}')

    # 加载最佳模型，测试集评估
    model.load_state_dict(best_state)
    test_rmse, test_mae, test_r2 = evaluate_gnn(model, test_loader, device)

    sep = "=" * 10
    print(f'\n{sep} GNN 测试集结果 {sep}')
    print(f'  RMSE: {test_rmse:.4f}')
    print(f'  MAE:  {test_mae:.4f}')
    print(f'  R²:   {test_r2:.4f}')

    # 保存模型和配置
    torch.save(best_state, MODEL_PATH)
    with open(FEATURE_SAVE, 'w') as f:
        json.dump({'atom_feat_dim': ATOM_FEAT_DIM, 'hidden': 64}, f)
    print(f'\n模型已保存: {MODEL_PATH}')

    # 与 GBR 对比
    print(f'\n{sep} GNN vs GBR 对比 {sep}')
    print(f'  GBR (之前最优): RMSE=0.7492  R²=0.6161')
    print(f'  GNN (本次):      RMSE={test_rmse:.4f}  R²={test_r2:.4f}')


if __name__ == '__main__':
    main()
