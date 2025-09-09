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

def mse_loss(pred, true):
    return F.mse_loss(pred, true)

def kl_loss(latent):
    mean = torch.mean(latent, dim=0)
    var = torch.var(latent, dim=0)
    kl_div = -0.5 * torch.sum(1 + torch.log(var + 1e-10) - mean.pow(2) - var)
    return kl_div

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

train_csv_file = 'train_data.csv'
test_csv_file = 'validation_data.csv'
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)
smiles_column = 'Smiles'
target_column = 'pchembl'

y_train = train_df[target_column].values.reshape(-1, 1)
y_test = test_df[target_column].values.reshape(-1, 1)

scaler = StandardScaler()
y_train_normalized = scaler.fit_transform(y_train)
y_test_normalized = scaler.transform(y_test)

train_df['pchembl_normalized'] = y_train_normalized
test_df['pchembl_normalized'] = y_test_normalized

train_df['pchembl_original'] = train_df[target_column]
test_df['pchembl_original'] = test_df[target_column]

train_data_list = []
for index, row in train_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index = smiles_to_graph(smiles)
        bci_fp = get_bci_fingerprint(smiles)
        data = Data(x=atom_features, edge_index=edge_index)
        data.y = torch.tensor(row['pchembl_normalized'], dtype=torch.float)
        data.y_original = torch.tensor(row['pchembl_original'], dtype=torch.float)
        train_data_list.append((data, bci_fp))
    except ValueError as e:
        print(e)

test_data_list = []
for index, row in test_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index = smiles_to_graph(smiles)
        bci_fp = get_bci_fingerprint(smiles)
        data = Data(x=atom_features, edge_index=edge_index)
        data.y = torch.tensor(row['pchembl_normalized'], dtype=torch.float)
        data.y_original = torch.tensor(row['pchembl_original'], dtype=torch.float)
        test_data_list.append((data, bci_fp))
    except ValueError as e:
        print(e)

train_loader = DataLoader(train_data_list, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

gat_graphsage_model = GAT_GraphSAGE(n_output=1, num_features_xd=35)
cnn_model = CNNNet(input_dim=1024, output_dim=1024)
combined_model = CombinedNet(input_dim=1025, hidden_dim=512, output_dim=1)

optimizer = torch.optim.Adam(
    list(gat_graphsage_model.parameters()) +
    list(cnn_model.parameters()) +
    list(combined_model.parameters()),
    lr=0.001,
    weight_decay=1e-4
)

best_mse = float('inf')
best_original_mse = float('inf')
best_model_file = 'best_model.pth'
lambda_kl = 0.001

for epoch in range(1000):
    gat_graphsage_model.train()
    cnn_model.train()
    combined_model.train()
    epoch_train_loss = 0.0
    num_batches = 0

    for batch_data, batch_bci_fp in train_loader:
        optimizer.zero_grad()
        gat_graphsage_output = gat_graphsage_model(batch_data)
        cnn_output = cnn_model(batch_bci_fp)
        combined_output = torch.cat((gat_graphsage_output, cnn_output), dim=1)
        final_output = combined_model(combined_output)

        batch_targets = batch_data.y.view(-1, 1)
        l_mse = mse_loss(final_output, batch_targets)
        l_kl = kl_loss(combined_output)
        total_loss = l_mse + lambda_kl * l_kl

        total_loss.backward()
        optimizer.step()

        epoch_train_loss += total_loss.item()
        num_batches += 1

    avg_train_loss = epoch_train_loss / num_batches

    gat_graphsage_model.eval()
    cnn_model.eval()
    combined_model.eval()
    test_mse_total = 0.0
    test_original_mse_total = 0.0

    with torch.no_grad():
        for batch_data, batch_bci_fp in test_loader:
            gat_graphsage_output = gat_graphsage_model(batch_data)
            cnn_output = cnn_model(batch_bci_fp)
            combined_output = torch.cat((gat_graphsage_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            batch_targets = batch_data.y.view(-1, 1)
            mse = mse_loss(final_output, batch_targets)
            test_mse_total += mse.item()

            denormalized_preds = torch.tensor(scaler.inverse_transform(final_output.cpu()), dtype=torch.float)
            original_targets = batch_data.y_original.view(-1, 1)
            original_mse = F.mse_loss(denormalized_preds, original_targets)
            test_original_mse_total += original_mse.item()

    test_mse_avg = test_mse_total / len(test_loader)
    test_original_mse_avg = test_original_mse_total / len(test_loader)

    print(
        f"Epoch {epoch + 1:4d} | Train Loss: {avg_train_loss:.4f} | Test MSE (Normalized): {test_mse_avg:.4f} | Test MSE (Original): {test_original_mse_avg:.4f}")

    if test_original_mse_avg < best_original_mse:
        best_mse = test_mse_avg
        best_original_mse = test_original_mse_avg
        torch.save({
            'gat_graphsage_model_state_dict': gat_graphsage_model.state_dict(),
            'cnn_model_state_dict': cnn_model.state_dict(),
            'combined_model_state_dict': combined_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalized_mse': test_mse_avg,
            'original_mse': test_original_mse_avg,
            'scaler': scaler,
        }, best_model_file)
        print(f"*** New best model saved at epoch {epoch + 1}, Normalized MSE: {best_mse:.4f}, Original MSE: {best_original_mse:.4f} ***")

print(f"\nTraining complete, Best normalized MSE: {best_mse:.4f}, Best original MSE: {best_original_mse:.4f}")