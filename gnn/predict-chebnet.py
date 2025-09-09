import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from torch_geometric.nn import MessagePassing, global_max_pool
import torch.nn.functional as F

def one_of_k_encoding_unk(x, valid_entries):
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

class ChebConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(ChebConv, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        L = self.get_laplacian(edge_index, x.size(0))
        T_k = [torch.eye(L.size(0)).to(x.device), L]
        for k in range(2, self.K):
            T_k.append(2 * L @ T_k[k - 1] - T_k[k - 2])

        out = T_k[0] @ x
        for k in range(1, self.K):
            out += T_k[k] @ x

        return self.lin(out)

    def get_laplacian(self, edge_index, num_nodes):
        row, col = edge_index
        laplacian = torch.zeros(num_nodes, num_nodes).to(edge_index.device)
        laplacian[row, col] = -1
        laplacian = laplacian + torch.diag(torch.sum(laplacian, dim=1))
        return laplacian

class ChebNet(nn.Module):
    def __init__(self, num_features_xd=35, n_output=1, output_dim=128, K=3, dropout=0.2):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(num_features_xd, 16, K)
        self.conv2 = ChebConv(16, output_dim, K)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.out = nn.Linear(output_dim, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)
        out = self.out(x)
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

model_file_name = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\best_cheb_model.pth'
model = ChebNet(n_output=1)
model.load_state_dict(torch.load(model_file_name))
model.eval()

def evaluate_model(test_loader):
    all_predictions = []
    all_targets = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            batch_targets = batch.y.view(-1, 1)
            all_predictions.append(output.numpy())
            all_targets.append(batch_targets.numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    mse = criterion(torch.tensor(all_predictions), torch.tensor(all_targets)).item()
    mae = np.mean(np.abs(all_predictions - all_targets))
    pearson_corr = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]

    return mse, mae, pearson_corr

mse, mae, pearson_corr = evaluate_model(test_loader)

print(f"Model evaluation results:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nPearson correlation coefficient: {pearson_corr:.4f}")