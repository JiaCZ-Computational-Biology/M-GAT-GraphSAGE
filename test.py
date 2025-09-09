import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler


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
                  ]) + [atom.GetIsAromatic()] + \
                  one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
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


def get_ecfp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot generate molecule from SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return torch.tensor(np.array(fp), dtype=torch.float).view(1, -1)


class ModifiedGATLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ModifiedGATLayer, self).__init__()
        self.query_transform = torch.nn.Linear(in_features, out_features)
        self.key_transform = torch.nn.Linear(in_features, out_features)
        self.value_transform = torch.nn.Linear(in_features, out_features)
        self.conv3 = torch.nn.Conv1d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv1d(in_channels=out_features, out_channels=out_features, kernel_size=5, padding=2)
        self.linear_transform = torch.nn.Linear(out_features * 2 + out_features, out_features)

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


class GAT_GraphSAGE(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=35, output_dim=128, dropout=0.3):
        super(GAT_GraphSAGE, self).__init__()
        from torch_geometric.nn import SAGEConv, global_max_pool
        self.conv1 = ModifiedGATLayer(in_features=num_features_xd, out_features=num_features_xd)
        self.conv2 = SAGEConv(num_features_xd, num_features_xd)
        self.fc_g1 = torch.nn.Linear(num_features_xd, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(output_dim, n_output)
        self.global_max_pool = global_max_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.global_max_pool(x, batch)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        out = self.out(x)
        return out


class CNNNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(CNNNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.fc1 = torch.nn.Linear(128 * input_dim, 256)
        self.fc2 = torch.nn.Linear(256, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, ecfp):
        ecfp = ecfp.squeeze(1).unsqueeze(1)
        x = self.relu(self.conv1(ecfp))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CombinedNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CombinedNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def main():
    test_csv_file = 'test_data.csv'
    test_df = pd.read_csv(test_csv_file)
    smiles_column = 'Smiles'
    target_column = 'pchembl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gat_graphsage_model = GAT_GraphSAGE(n_output=1, num_features_xd=35).to(device)
    cnn_model = CNNNet(input_dim=1024, output_dim=1024).to(device)
    combined_model = CombinedNet(input_dim=1025, hidden_dim=512, output_dim=1).to(device)

    checkpoint = torch.load('best_model.pth', map_location=device)
    gat_graphsage_model.load_state_dict(checkpoint['gat_graphsage_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])
    scaler = checkpoint['scaler']

    gat_graphsage_model.eval()
    cnn_model.eval()
    combined_model.eval()

    true_values = []
    predicted_values = []

    print("Starting model evaluation on independent test set...")

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        y_true = row[target_column]

        try:
            atom_features, edge_index = smiles_to_graph(smiles)
            ecfp = get_ecfp(smiles)

            data = Data(x=atom_features, edge_index=edge_index)

            batch = torch.zeros(atom_features.size(0), dtype=torch.long).to(device)
            data.batch = batch

            data = data.to(device)
            ecfp = ecfp.to(device)

            with torch.no_grad():
                gat_graphsage_output = gat_graphsage_model(data)
                cnn_output = cnn_model(ecfp)
                combined_output = torch.cat((gat_graphsage_output, cnn_output), dim=1)
                final_output = combined_model(combined_output)

                normalized_pred = final_output.cpu().numpy()
                original_pred = scaler.inverse_transform(normalized_pred)[0][0]

                true_values.append(y_true)
                predicted_values.append(original_pred)

                if (index + 1) % 100 == 0:
                    print(f"Processed {index + 1} samples...")

        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            continue

    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    pearson_corr, p_value = pearsonr(true_values, predicted_values)

    print("\nModel evaluation results:")
    print(f"Number of test samples: {len(true_values)}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {p_value:.4e})")

    results_df = pd.DataFrame({
        'SMILES': test_df[smiles_column][:len(predicted_values)],
        'True_Value': true_values,
        'Predicted_Value': predicted_values,
        'Absolute_Error': np.abs(true_values - predicted_values)
    })

    results_df.to_csv('model_prediction_results.csv', index=False)
    print("Prediction results saved to model_prediction_results.csv")


if __name__ == "__main__":
    main()