import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

class GINConvNet(nn.Module):
    def __init__(self, n_output=1, num_features_xd=35, dropout=0.2):
        super(GINConvNet, self).__init__()
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features_xd, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn1 = nn.BatchNorm1d(dim)

        self.conv2 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn2 = nn.BatchNorm1d(dim)

        self.conv3 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn3 = nn.BatchNorm1d(dim)

        self.conv4 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn4 = nn.BatchNorm1d(dim)

        self.conv5 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn5 = nn.BatchNorm1d(dim)

        self.fc1_xd = nn.Linear(dim, 128)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = self.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = self.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        x = global_add_pool(x, batch)
        x = self.relu(self.fc1_xd(x))
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        out = self.out(x)

        return out

independent_test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\independent_test_data.csv'

independent_test_df = pd.read_csv(independent_test_csv_file)
smiles_column = 'Smiles'
target_column = 'pchembl'

independent_test_data_list = []
for index, row in independent_test_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index = smiles_to_graph(smiles)
        data = Data(x=atom_features, edge_index=edge_index)
        data.y = torch.tensor(row[target_column], dtype=torch.float)
        independent_test_data_list.append(data)
    except ValueError as e:
        print(e)

independent_test_loader = DataLoader(independent_test_data_list, batch_size=64, shuffle=False)

model = GINConvNet(n_output=1)
model_file_name = 'best_gin_model.pth'
model.load_state_dict(torch.load(model_file_name))
model.eval()

predictions = []
true_values = []

with torch.no_grad():
    for batch in independent_test_loader:
        output = model(batch)
        predictions.extend(output.cpu().numpy().flatten())
        true_values.extend(batch.y.cpu().numpy())

predictions = np.array(predictions)
true_values = np.array(true_values)

mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
pearson_corr, pearson_p_value = pearsonr(true_values, predictions)

print("=== Independent Test Set Evaluation Results ===")
print(f"MSE (Mean Squared Error): {mse:.6f}")
print(f"MAE (Mean Absolute Error): {mae:.6f}")
print(f"Pearson Correlation Coefficient: {pearson_corr:.6f}")
print(f"Pearson Correlation p-value: {pearson_p_value:.6f}")
print(f"Number of test samples: {len(true_values)}")

results_df = pd.DataFrame({
    'True_Values': true_values,
    'Predictions': predictions,
    'Absolute_Error': np.abs(true_values - predictions)
})
results_df.to_csv('independent_test_results.csv', index=False)
print("\nPrediction results saved to 'independent_test_results.csv'")

rmse = np.sqrt(mse)
print(f"\nAdditional Statistics:")
print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"True value range: [{true_values.min():.4f}, {true_values.max():.4f}]")