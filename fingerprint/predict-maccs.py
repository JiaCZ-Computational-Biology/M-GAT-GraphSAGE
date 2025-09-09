from scipy import stats
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from torch_geometric.nn import GATConv, SAGEConv, global_max_pool
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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

def get_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot generate molecule from SMILES: {smiles}")

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = list(maccs_fp)
    maccs_tensor = torch.tensor(maccs_array, dtype=torch.float).view(1, -1)

    return maccs_tensor

class ModifiedGATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModifiedGATLayer, self).__init__()
        self.query_transform = nn.Linear(in_features, out_features)
        self.key_transform = nn.Linear(in_features, out_features)
        self.value_transform = nn.Linear(in_features, out_features)

        self.conv3 = nn.Conv1d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=out_features, out_channels=out_features, kernel_size=5, padding=2)
        self.linear_transform = nn.Linear(out_features * 2 + out_features, out_features)

    def forward(self, x):
        Q = self.query_transform(x)
        K = self.key_transform(x)
        V = self.value_transform(x)

        K = K.unsqueeze(2)
        K_conv3 = self.conv3(K)
        K_conv5 = self.conv5(K)

        K_concat = torch.cat((K_conv3, K_conv5, K), dim=1)
        K_new = self.linear_transform(K_concat.transpose(1, 2))

        attention_scores = torch.matmul(Q, K_new.transpose(1, 2)) / (K_new.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=-1)
        output = torch.matmul(attention_weights, V) + V
        return output

class GAT_GraphSAGE(nn.Module):
    def __init__(self, n_output=1, num_features_xd=35, output_dim=128, dropout=0.2):
        super(GAT_GraphSAGE, self).__init__()
        self.n_output = n_output

        self.conv1 = ModifiedGATLayer(in_features=num_features_xd, out_features=num_features_xd)
        self.conv2 = SAGEConv(num_features_xd, num_features_xd)

        self.fc_g1 = nn.Linear(num_features_xd, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(output_dim, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = global_max_pool(x, batch)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        out = self.out(x)
        return out

class CNNNet(nn.Module):
    def __init__(self, input_dim=167, output_dim=1024, dropout=0.2):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(128 * input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, maccs):
        maccs = maccs.squeeze(1)
        maccs = maccs.unsqueeze(1)
        x = self.conv1(maccs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CombinedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CombinedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate_model(test_loader, gat_graphsage_model, cnn_model, combined_model):
    gat_graphsage_model.eval()
    cnn_model.eval()
    combined_model.eval()

    true_values = []
    pred_values = []

    with torch.no_grad():
        for batch_data, batch_maccs in test_loader:
            gat_graphsage_output = gat_graphsage_model(batch_data)
            cnn_output = cnn_model(batch_maccs)
            combined_output = torch.cat((gat_graphsage_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            true_values.extend(batch_data.y.cpu().numpy())
            pred_values.extend(final_output.cpu().numpy())

    true_values = np.array(true_values)
    pred_values = np.array(pred_values).flatten()

    mse = np.mean((true_values - pred_values) ** 2)
    mae = np.mean(np.abs(true_values - pred_values))
    pearson_corr, p_value = stats.pearsonr(true_values, pred_values)

    return {
        'MSE': mse,
        'MAE': mae,
        'Pearson_Correlation': pearson_corr,
        'P_Value': p_value,
        'true_values': true_values,
        'pred_values': pred_values
    }

test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv'
test_df = pd.read_csv(test_csv_file)
smiles_column = 'Smiles'
target_column = 'pchembl'

test_data_list = []
for index, row in test_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index = smiles_to_graph(smiles)
        maccs = get_maccs(smiles)
        data = Data(x=atom_features, edge_index=edge_index)
        data.y = torch.tensor(row[target_column], dtype=torch.float)
        test_data_list.append((data, maccs))
    except ValueError as e:
        print(e)

test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

gat_graphsage_model = GAT_GraphSAGE(n_output=1, num_features_xd=35)
cnn_model = CNNNet(input_dim=167, output_dim=1024)
combined_model = CombinedNet(input_dim=1025, hidden_dim=512, output_dim=1)

checkpoint = torch.load('D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\best_gat-graphsage-model.pth')
gat_graphsage_model.load_state_dict(checkpoint['gat_graphsage_model_state_dict'])
cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

evaluation_results = evaluate_model(test_loader, gat_graphsage_model, cnn_model, combined_model)

print("Model Performance Evaluation Results:")
for metric, value in evaluation_results.items():
    if metric not in ['true_values', 'pred_values']:
        print(f"{metric}: {value}")

plt.figure(figsize=(10, 6))
plt.scatter(evaluation_results['true_values'], evaluation_results['pred_values'], alpha=0.5)
plt.plot([min(evaluation_results['true_values']), max(evaluation_results['true_values'])],
         [min(evaluation_results['true_values']), max(evaluation_results['true_values'])],
         'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values')
plt.tight_layout()
plt.show()