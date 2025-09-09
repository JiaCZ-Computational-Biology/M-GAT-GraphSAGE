from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from torch_geometric.nn import GATConv, global_max_pool
import torch.nn.functional as F

def one_of_k_encoding_unk(x, valid_entries):
    """Perform one-hot encoding with an 'unknown' category."""
    if x not in valid_entries:
        x = 'Unknown'
    return [1 if entry == x else 0 for entry in valid_entries]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()
    atom_features = []

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']) + \
                  one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]

        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)
        atom_features.append(atom_feats)

    atom_features = torch.tensor(atom_features, dtype=torch.float)
    adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_matrix[start, end] = 1.0
        adj_matrix[end, start] = 1.0

    edge_index = adj_matrix.nonzero(as_tuple=False).t().long()
    return atom_features, edge_index

class GATNet(nn.Module):
    def __init__(self, num_features_xd=35, n_output=1, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        self.out = nn.Linear(output_dim, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)

        out = self.out(x)
        return out

test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv'
test_df = pd.read_csv(test_csv_file)

test_data_list = []
for index, row in test_df.iterrows():
    smiles = str(row['Smiles'])
    try:
        atom_features, edge_index = smiles_to_graph(smiles)
        data = Data(x=atom_features, edge_index=edge_index)
        data.y = torch.tensor(row['pchembl'], dtype=torch.float)
        test_data_list.append(data)
    except ValueError as e:
        print(e)

test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

model = GATNet(n_output=1)
model.load_state_dict(torch.load('D:\\pycharm\\gutingle\\pythonProject2\\best_gat_model.pth'))
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        all_predictions.append(output.view(-1).cpu().numpy())
        all_targets.append(batch.y.view(-1).cpu().numpy())

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

mse = mean_squared_error(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)
pearson_corr, _ = pearsonr(all_targets, all_predictions)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")