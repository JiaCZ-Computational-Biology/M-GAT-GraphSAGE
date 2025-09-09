import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from torch_geometric.nn import GATConv, SAGEConv, global_max_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import pickle

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

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

def get_smifp(smiles, nbits=1024):
    charset = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H',
               '(', ')', '[', ']', '=', '#', '@', '+', '-', '\\', '/',
               '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'c', 'n', 'o', 's', 'p']

    fingerprint = np.zeros(nbits)

    if not smiles or len(smiles) == 0:
        return torch.tensor(fingerprint, dtype=torch.float).view(1, -1)

    for n in range(1, 4):
        for i in range(len(smiles) - n + 1):
            fragment = smiles[i:i + n]
            hash_val = hash(fragment) % nbits
            fingerprint[hash_val] = 1

    char_counts = {}
    for char in smiles:
        if char in charset:
            char_counts[char] = char_counts.get(char, 0) + 1

    for char, count in char_counts.items():
        if char in charset:
            char_idx = charset.index(char)
            for j in range(min(count, 10)):
                hash_val = hash(f"{char}_{j}") % nbits
                fingerprint[hash_val] = 1

    length_feature = len(smiles) % 100
    for i in range(length_feature):
        hash_val = hash(f"length_{i}") % nbits
        fingerprint[hash_val] = 1

    return torch.tensor(fingerprint, dtype=torch.float).view(1, -1)

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
    def __init__(self, n_output=1, num_features_xd=35, output_dim=128, dropout=0.3):
        super(GAT_GraphSAGE, self).__init__()
        self.conv1 = ModifiedGATLayer(in_features=num_features_xd, out_features=num_features_xd)
        self.conv2 = SAGEConv(num_features_xd, num_features_xd)
        self.fc_g1 = nn.Linear(num_features_xd, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(output_dim, n_output)

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
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(128 * input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, smifp):
        smifp = smifp.squeeze(1).unsqueeze(1)
        x = self.relu(self.conv1(smifp))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
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
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model_and_evaluate(model_path, test_csv_file, smiles_column='Smiles', target_column='pchembl'):
    print("Loading saved model...")

    checkpoint = torch.load(model_path, map_location='cpu')

    scaler = checkpoint['scaler']

    gat_graphsage_model = GAT_GraphSAGE(n_output=1, num_features_xd=35)
    cnn_model = CNNNet(input_dim=1024, output_dim=1024)
    combined_model = CombinedNet(input_dim=1025, hidden_dim=512, output_dim=1)

    gat_graphsage_model.load_state_dict(checkpoint['gat_graphsage_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

    gat_graphsage_model.eval()
    cnn_model.eval()
    combined_model.eval()

    print("Model loaded successfully!")

    print(f"Reading independent test set: {test_csv_file}")
    test_df = pd.read_csv(test_csv_file)
    print(f"Test set size: {len(test_df)}")

    test_data_list = []
    valid_indices = []

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        try:
            atom_features, edge_index = smiles_to_graph(smiles)
            smifp = get_smifp(smiles)
            data = Data(x=atom_features, edge_index=edge_index)
            data.y_original = torch.tensor(row[target_column], dtype=torch.float)
            test_data_list.append((data, smifp))
            valid_indices.append(index)
        except ValueError as e:
            print(f"Skipping invalid SMILES: {smiles}, Error: {e}")

    print(f"Valid test sample count: {len(test_data_list)}")

    if len(test_data_list) == 0:
        print("No valid test samples!")
        return

    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

    print("Making predictions...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data, batch_smifp in test_loader:
            gat_graphsage_output = gat_graphsage_model(batch_data)
            cnn_output = cnn_model(batch_smifp)
            combined_output = torch.cat((gat_graphsage_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            denormalized_preds = scaler.inverse_transform(final_output.cpu().numpy())

            original_targets = batch_data.y_original.cpu().numpy()

            all_predictions.extend(denormalized_preds.flatten())
            all_targets.extend(original_targets.flatten())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    pearson_corr, p_value = pearsonr(targets, predictions)

    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))

    print("\n" + "=" * 60)
    print("Independent Test Set Evaluation Results")
    print("=" * 60)
    print(f"Test sample count: {len(predictions)}")
    print(f"MSE (Mean Squared Error):  {mse:.6f}")
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"Pearson correlation coefficient: {pearson_corr:.6f}")
    print(f"Pearson p-value: {p_value:.6e}")
    print(f"R² coefficient of determination: {r2:.6f}")
    print("=" * 60)

    results_df = test_df.loc[valid_indices].copy()
    results_df['predicted_pchembl'] = predictions
    results_df['actual_pchembl'] = targets
    results_df['prediction_error'] = targets - predictions
    results_df['absolute_error'] = np.abs(targets - predictions)

    output_file = 'independent_test_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Prediction results saved to: {output_file}")

    print("\nPrediction statistics:")
    print(f"True value range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"Predicted value range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Mean absolute error: {mae:.6f}")
    print(f"Maximum absolute error: {np.max(np.abs(targets - predictions)):.6f}")
    print(f"Minimum absolute error: {np.min(np.abs(targets - predictions)):.6f}")

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson_corr': pearson_corr,
        'pearson_p_value': p_value,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }

if __name__ == "__main__":
    model_path = 'best_model.pth'
    test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv'

    import os

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist!")
        print("Please ensure you have run the training script and saved the model.")
        exit(1)

    if not os.path.exists(test_csv_file):
        print(f"Error: Test set file {test_csv_file} does not exist!")
        print("Please provide an independent test set CSV file.")
        exit(1)

    try:
        results = load_model_and_evaluate(
            model_path=model_path,
            test_csv_file=test_csv_file,
            smiles_column='Smiles',
            target_column='pchembl'
        )

        print("\nEvaluation completed!")

    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()