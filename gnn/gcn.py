import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from torch_geometric.nn import GCNConv, global_max_pool

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic())
        ]
        atom_features.append(features)

    atom_features = torch.tensor(atom_features, dtype=torch.float)
    adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_matrix[start, end] = 1.0
        adj_matrix[end, start] = 1.0

    edge_index = adj_matrix.nonzero(as_tuple=False).t().long()
    return atom_features, edge_index

class GCNNet(nn.Module):
    def __init__(self, n_output=1, num_features_xd=5, dropout=0.1):
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

train_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\train_data.csv'
test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\validation_data.csv'

train_df = pd.read_csv(train_csv_file)
smiles_column = 'Smiles'
target_column = 'pchembl'

train_data_list = []
for index, row in train_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index = smiles_to_graph(smiles)
        data = Data(x=atom_features, edge_index=edge_index)
        data.y = torch.tensor(row[target_column], dtype=torch.float)
        train_data_list.append(data)
    except ValueError as e:
        print(e)

test_df = pd.read_csv(test_csv_file)
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

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

model = GCNNet(n_output=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00059)
criterion = nn.MSELoss()

best_mse = float('inf')
best_epoch = -1
model_file_name = 'best_gcn_model.pth'

for epoch in range(10):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        batch_targets = batch.y.view(-1, 1)
        loss = criterion(output, batch_targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()
    test_loss_mse = 0.0
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            batch_targets = batch.y.view(-1, 1)
            mse_loss = criterion(output, batch_targets)
            test_loss_mse += mse_loss.item()

    test_loss_mse /= len(test_loader)

    if test_loss_mse < best_mse:
        best_mse = test_loss_mse
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name)
        print(f"Improved model found at epoch {best_epoch}, MSE: {best_mse:.4f}")

print(f"Training complete. Best model saved at epoch {best_epoch} with MSE: {best_mse:.4f}")