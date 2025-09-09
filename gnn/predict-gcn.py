import pandas as pd
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from scipy.stats import pearsonr
from rdkit import Chem
import numpy as np
import torch

def one_of_k_encoding_unk(x, valid_entries):
    """Perform one-hot encoding with 'unknown' category."""
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
        results = one_of_k_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']) + \
                  one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3,
                      Chem.rdchem.HybridizationType.SP3D,
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

class GCNNet(nn.Module):
    def __init__(self, n_output=1, num_features_xd=35, dropout=0.1):
        super(GCNNet, self).__init__()
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, self.n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, data.batch)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        out = self.fc_g2(x)
        return out

test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv'
test_df = pd.read_csv(test_csv_file)
smiles_column = 'Smiles'
target_column = 'pchembl'

test_data_list = []
for index, row in test_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index = smiles_to_graph(smiles)
        data = Data(x=atom_features, edge_index=edge_index)
        data.y = torch.tensor(row[target_column], dtype=torch.float)
        test_data_list.append(data)
    except ValueError as e:
        print(e)

test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

model = GCNNet(n_output=1)
model.load_state_dict(torch.load('D:\\pycharm\\gutingle\\pythonProject2\\GraphDTA\\best_gcn_model.pth'))
model.eval()

y_true = []
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        y_true.extend(batch.y.numpy())
        y_pred.extend(output.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))

pearson_corr, p_value = pearsonr(y_true.flatten(), y_pred.flatten())

print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'Pearson Correlation Coefficient: {pearson_corr:.4f}')
print(f'P-value: {p_value:.4f}')