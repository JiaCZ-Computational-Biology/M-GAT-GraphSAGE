import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
from torch_geometric.nn import GATConv, SAGEConv, global_max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

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

def get_bci_fingerprint(smiles, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot generate molecule from SMILES: {smiles}")

    base_fp = Chem.LayeredFingerprint(mol, fpSize=512)
    base_fp_array = np.array(base_fp)

    descriptors = []
    try:
        descriptors.extend([
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumHeteroatoms(mol)
        ])

        descriptors.extend([
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            Descriptors.Chi0n(mol),
            Descriptors.Chi1n(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol)
        ])

        descriptors.extend([
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol)
        ])

        descriptors.extend([
            Descriptors.EState_VSA1(mol),
            Descriptors.EState_VSA2(mol),
            Descriptors.EState_VSA3(mol),
            Descriptors.EState_VSA4(mol),
            Descriptors.EState_VSA5(mol),
            Descriptors.EState_VSA6(mol),
            Descriptors.EState_VSA7(mol),
            Descriptors.EState_VSA8(mol),
            Descriptors.EState_VSA9(mol),
            Descriptors.EState_VSA10(mol),
            Descriptors.EState_VSA11(mol)
        ])

        descriptors.extend([
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            Descriptors.MolMR(mol),
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol)
        ])

        descriptors.extend([
            rdMolDescriptors.CalcNumAtomStereoCenters(mol),
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol)
        ])

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        descriptors.extend([
            num_atoms,
            num_bonds,
            num_bonds / max(num_atoms, 1),
            len(Chem.GetMolFrags(mol)),
        ])

    except Exception as e:
        print(f"Error calculating descriptors: {e}")
        descriptors = [0.0] * 50

    descriptors = np.array(descriptors, dtype=np.float32)
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=1.0, neginf=-1.0)

    if len(descriptors) < 512:
        descriptors = np.pad(descriptors, (0, 512 - len(descriptors)), 'constant')
    else:
        descriptors = descriptors[:512]

    bci_fp = np.concatenate([base_fp_array.astype(np.float32), descriptors])

    if len(bci_fp) > nBits:
        bci_fp = bci_fp[:nBits]
    elif len(bci_fp) < nBits:
        bci_fp = np.pad(bci_fp, (0, nBits - len(bci_fp)), 'constant')

    return torch.tensor(bci_fp, dtype=torch.float).view(1, -1)

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

    def forward(self, bci_fp):
        bci_fp = bci_fp.squeeze(1).unsqueeze(1)
        x = self.relu(self.conv1(bci_fp))
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

def load_model_and_evaluate(model_path, test_csv_path, smiles_column='Smiles', target_column='pchembl'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    gat_graphsage_model = GAT_GraphSAGE(n_output=1, num_features_xd=35).to(device)
    cnn_model = CNNNet(input_dim=1024, output_dim=1024).to(device)
    combined_model = CombinedNet(input_dim=1025, hidden_dim=512, output_dim=1).to(device)

    gat_graphsage_model.load_state_dict(checkpoint['gat_graphsage_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

    scaler = checkpoint['scaler']

    gat_graphsage_model.eval()
    cnn_model.eval()
    combined_model.eval()

    print(f"Model loaded successfully!")
    print(f"Training best normalized MSE: {checkpoint['normalized_mse']:.4f}")
    print(f"Training best original MSE: {checkpoint['original_mse']:.4f}")

    print(f"Reading independent test set: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"Independent test set sample count: {len(test_df)}")

    test_data_list = []
    valid_indices = []

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        try:
            atom_features, edge_index = smiles_to_graph(smiles)
            bci_fp = get_bci_fingerprint(smiles)
            data = Data(x=atom_features, edge_index=edge_index)
            data.y = torch.tensor(row[target_column], dtype=torch.float)
            test_data_list.append((data, bci_fp))
            valid_indices.append(index)
        except Exception as e:
            print(f"Error processing SMILES (index {index}): {e}")

    print(f"Successfully processed test samples: {len(test_data_list)}")

    if len(test_data_list) == 0:
        print("No valid test samples!")
        return None

    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

    predictions = []
    true_values = []

    print("Making predictions...")
    with torch.no_grad():
        for batch_data, batch_bci_fp in test_loader:
            batch_data = batch_data.to(device)
            batch_bci_fp = batch_bci_fp.to(device)

            gat_graphsage_output = gat_graphsage_model(batch_data)
            cnn_output = cnn_model(batch_bci_fp)
            combined_output = torch.cat((gat_graphsage_output, cnn_output), dim=1)
            pred_normalized = combined_model(combined_output)

            pred_original = scaler.inverse_transform(pred_normalized.cpu().numpy())

            predictions.extend(pred_original.flatten())
            true_values.extend(batch_data.y.cpu().numpy())

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    pearson_corr, pearson_p = pearsonr(true_values, predictions)

    ss_res = np.sum((true_values - predictions) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("\n" + "=" * 60)
    print("Independent Test Set Evaluation Results:")
    print("=" * 60)
    print(f"Sample count: {len(predictions)}")
    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {np.sqrt(mse):.4f}")
    print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
    print(f"Pearson correlation p-value: {pearson_p:.2e}")
    print(f"R² (Coefficient of determination): {r2:.4f}")
    print("=" * 60)

    results_df = test_df.iloc[valid_indices].copy()
    results_df['predicted_value'] = predictions
    results_df['true_value'] = true_values
    results_df['absolute_error'] = np.abs(predictions - true_values)
    results_df['squared_error'] = (predictions - true_values) ** 2

    output_file = 'independent_test_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Prediction results saved to: {output_file}")

    results = {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'r2': r2,
        'predictions': predictions,
        'true_values': true_values,
        'sample_count': len(predictions)
    }

    return results

if __name__ == "__main__":
    model_path = 'best_model.pth'
    test_csv_path = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv'

    smiles_column = 'Smiles'
    target_column = 'pchembl'

    results = load_model_and_evaluate(
        model_path=model_path,
        test_csv_path=test_csv_path,
        smiles_column=smiles_column,
        target_column=target_column
    )

    if results:
        print(f"\nFinal evaluation metrics:")
        print(f"MSE: {results['mse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"Pearson correlation coefficient: {results['pearson_corr']:.4f}")