"""Train a small bond-aware graph regressor on the existing scaffold split.

Run python train_graph.py to write graph_results/best.pt and metrics.json.
"""

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import rdkit
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
SEED = 42
WIDTH = 32
DEPTH = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 40
PATIENCE = 8
CPU_THREADS = 1

# Fixed feature vocabulary. Unknown elements/bonds share a fallback category.
ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14)
BONDS = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
         Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
ATOM_DIM = len(ELEMENTS) + 1 + 6


def molecular_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid or empty SMILES: {smiles}")
    atoms = []
    for atom in mol.GetAtoms():
        element = atom.GetAtomicNum()
        one_hot = [element == number for number in ELEMENTS] + [element not in ELEMENTS]
        atoms.append(one_hot + [
            atom.GetDegree() / 4, atom.GetTotalNumHs() / 4, atom.GetFormalCharge(),
            atom.GetIsAromatic(), atom.IsInRing(), atom.GetMass() / 100,
        ])
    edges, types = [], []
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        kind = bond.GetBondType()
        bond_id = BONDS.index(kind) if kind in BONDS else len(BONDS)
        edges.extend([(left, right), (right, left)])
        types.extend([bond_id, bond_id])
    return (torch.tensor(atoms, dtype=torch.float32),
            torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T,
            torch.tensor(types, dtype=torch.long))


def load_split(path):
    groups = {"train": [], "valid": [], "test": []}
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = row["CMPD_CHEMBLID"]
            target = float(row["exp"])
            if not identifier or identifier in ids or not math.isfinite(target):
                raise ValueError(f"Invalid ID or target for {identifier!r}")
            ids.add(identifier)
            graph = molecular_graph(row["canonical_smiles"])
            groups[row["split"]].append((graph, target))
    if not all(groups.values()):
        raise ValueError("Each split must have at least one molecule")
    return groups


def collate(samples):
    atoms, edges, bonds, sizes, targets = [], [], [], [], []
    offset = 0
    for (features, index, types), target in samples:
        atoms.append(features)
        edges.append(index + offset)
        bonds.append(types)
        sizes.append(len(features))
        targets.append(target)
        offset += len(features)
    sizes = torch.tensor(sizes)
    molecule = torch.arange(len(samples)).repeat_interleave(sizes)
    graph = (torch.cat(atoms), torch.cat(edges, dim=1), torch.cat(bonds),
             molecule, sizes.unsqueeze(1))
    return graph, torch.tensor(targets, dtype=torch.float32)


class GraphRegressor(nn.Module):
    def __init__(self, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.encode = nn.Linear(ATOM_DIM, width)
        self.bond = nn.Embedding(len(BONDS) + 1, width)
        self.updates = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width),
                          nn.LayerNorm(width), nn.ReLU()) for _ in range(depth)
        )
        self.readout = nn.Sequential(nn.Linear(width + 1, width), nn.ReLU(),
                                     nn.Linear(width, 1))

    def forward(self, atoms, edges, bonds, molecule, sizes):
        hidden = self.encode(atoms)
        source, dest = edges
        bond_features = self.bond(bonds)
        for update in self.updates:
            messages = (hidden[source] + bond_features).relu()
            neighbors = torch.zeros_like(hidden).index_add_(0, dest, messages)
            hidden = hidden + update(hidden + neighbors)
        pooled = hidden.new_zeros((len(sizes), hidden.shape[1]))
        pooled.index_add_(0, molecule, hidden)
        summary = torch.cat((pooled / sizes, sizes.float().log()), dim=1)
        return self.readout(summary).squeeze(1)


@torch.no_grad()
def evaluate(model, loader, mean, std):
    model.eval()
    predictions, targets = [], []
    for graph, target in loader:
        predictions.append(model(*graph) * std + mean)
        targets.append(target)
    actual = torch.cat(targets).double()
    error = torch.cat(predictions).double() - actual
    return {"rmse": error.square().mean().sqrt().item(), "mae": error.abs().mean().item(),
            "r2": (1 - error.square().sum() / (actual - actual.mean()).square().sum()).item()}


def main():
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    split_path = ROOT / "split.csv"
    groups = load_split(split_path)
    loaders = {key: DataLoader(group, batch_size=BATCH_SIZE, shuffle=key == "train",
                               collate_fn=collate) for key, group in groups.items()}
    train_y = torch.tensor([target for _, target in groups["train"]])
    mean, std = train_y.mean().item(), train_y.std(unbiased=False).item()
    if std == 0:
        raise ValueError("Training targets have zero variance")
    model = GraphRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_rmse = math.inf
    best_epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for graph, target in loaders["train"]:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(*graph), (target - mean) / std)
            loss.backward()
            optimizer.step()
        valid = evaluate(model, loaders["valid"], mean, std)
        if not math.isfinite(valid["rmse"]):
            raise ValueError(f"Non-finite validation RMSE at epoch {epoch}")
        if valid["rmse"] < best_rmse:
            best_rmse = valid["rmse"]
            best_epoch, best_valid = epoch, valid
            best_state = deepcopy(model.state_dict())
        print(f"Epoch {epoch:02d}  valid RMSE {valid['rmse']:.4f}", flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], mean, std)
    results = ROOT / "graph_results"
    results.mkdir(exist_ok=True)
    checkpoint = {"model": best_state, "width": WIDTH, "depth": DEPTH,
                  "target_mean": mean, "target_std": std, "epoch": best_epoch}
    torch.save(checkpoint, results / "best.pt")
    report = {
        "model": "bond-aware message-passing regressor", "seed": SEED,
        "input": "split.csv", "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "split_counts": {key: len(group) for key, group in groups.items()},
        "torch": str(torch.__version__), "rdkit": rdkit.__version__, "cpu_threads": CPU_THREADS,
        "width": WIDTH, "depth": DEPTH, "batch_size": BATCH_SIZE,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "loss": "standardized target MSE",
        "target_train_mean": mean, "target_train_std": std,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "epochs_trained": epoch,
        "selection": "lowest validation RMSE", "best_epoch": best_epoch,
        "valid": best_valid, "test": test, "checkpoint": "best.pt",
    }
    (results / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
