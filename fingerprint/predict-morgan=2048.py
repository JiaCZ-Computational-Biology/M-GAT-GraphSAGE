import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import GATConv, SAGEConv, global_max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

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

def get_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot generate molecule from SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return torch.tensor(np.array(fp), dtype=torch.float).view(1, -1)

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

    def forward(self, morgan_fp):
        morgan_fp = morgan_fp.squeeze(1).unsqueeze(1)
        x = self.relu(self.conv1(morgan_fp))
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

def calculate_metrics(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    pearson_corr, pearson_p = pearsonr(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Pearson_r': pearson_corr,
        'Pearson_p': pearson_p,
        'R2': r2
    }

def load_model_and_evaluate(model_path, test_csv_file, smiles_column='Smiles', target_column='pchembl'):
    print("Loading model...")

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    gat_graphsage_model = GAT_GraphSAGE(n_output=1, num_features_xd=35)
    cnn_model = CNNNet(input_dim=2048, output_dim=2048)
    combined_model = CombinedNet(input_dim=2049, hidden_dim=512, output_dim=1)

    gat_graphsage_model.load_state_dict(checkpoint['gat_graphsage_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

    scaler = checkpoint['scaler']

    print(f"Model loaded successfully!")
    print(f"Training best normalized MSE: {checkpoint['normalized_mse']:.4f}")
    print(f"Training best original MSE: {checkpoint['original_mse']:.4f}")

    gat_graphsage_model.eval()
    cnn_model.eval()
    combined_model.eval()

    print(f"\nReading independent test set: {test_csv_file}")
    test_df = pd.read_csv(test_csv_file)
    print(f"Independent test set sample count: {len(test_df)}")

    test_data_list = []
    failed_smiles = []

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        try:
            atom_features, edge_index = smiles_to_graph(smiles)
            morgan_fp = get_morgan_fingerprint(smiles)
            data = Data(x=atom_features, edge_index=edge_index)
            data.y_original = torch.tensor(row[target_column], dtype=torch.float)
            test_data_list.append((data, morgan_fp))
        except ValueError as e:
            failed_smiles.append(smiles)
            print(f"Failed to process SMILES: {smiles} - {e}")

    print(f"Successfully processed samples: {len(test_data_list)}")
    if failed_smiles:
        print(f"Failed samples: {len(failed_smiles)}")

    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

    print("\nMaking predictions...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data, batch_morgan_fp in test_loader:
            gat_graphsage_output = gat_graphsage_model(batch_data)
            cnn_output = cnn_model(batch_morgan_fp)
            combined_output = torch.cat((gat_graphsage_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            denormalized_preds = torch.tensor(
                scaler.inverse_transform(final_output.cpu()),
                dtype=torch.float
            )

            all_predictions.extend(denormalized_preds.flatten().tolist())
            all_targets.extend(batch_data.y_original.flatten().tolist())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(all_targets, all_predictions)

    print("\n" + "=" * 60)
    print("Independent Test Set Evaluation Results")
    print("=" * 60)
    print(f"Sample count: {len(all_targets)}")
    print(f"MSE (Mean Squared Error): {metrics['MSE']:.6f}")
    print(f"MAE (Mean Absolute Error): {metrics['MAE']:.6f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.6f}")
    print(f"Pearson correlation coefficient: {metrics['Pearson_r']:.6f}")
    print(f"Pearson correlation p-value: {metrics['Pearson_p']:.6e}")
    print(f"R² (Coefficient of determination): {metrics['R2']:.6f}")
    print("=" * 60)

    results_df = pd.DataFrame({
        'True_Value': all_targets,
        'Predicted_Value': all_predictions,
        'Absolute_Error': np.abs(all_targets - all_predictions),
        'Squared_Error': (all_targets - all_predictions) ** 2
    })

    output_file = 'independent_test_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nPrediction results saved to: {output_file}")

    return metrics, results_df

if __name__ == "__main__":
    model_path = 'best_model.pth'
    test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv'

    try:
        metrics, predictions_df = load_model_and_evaluate(
            model_path=model_path,
            test_csv_file=test_csv_file,
            smiles_column='Smiles',
            target_column='pchembl'
        )

        print("\nEvaluation completed!")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure the following files exist:")
        print(f"1. Model file: {model_path}")
        print(f"2. Test data file: {test_csv_file}")

    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()