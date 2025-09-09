import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ExplainerConfig, ModelConfig
from torch_geometric.nn import GATConv, SAGEConv, global_max_pool
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdDepictor
import io
import base64
from PIL import Image
from collections import defaultdict, Counter
import warnings

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

try:
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:
    try:
        from rdkit.Chem import rdMolDraw2D
    except ImportError:
        rdMolDraw2D = None
        print("Warning: rdMolDraw2D not available")

warnings.filterwarnings('ignore')

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def one_of_k_encoding_unk(x, valid_entries):
    if x not in valid_entries:
        x = 'Unknown'
    return [1 if entry == x else 0 for entry in valid_entries]


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


class ExplainableGATGraphSAGE(nn.Module):
    def __init__(self, gat_graphsage_model):
        super(ExplainableGATGraphSAGE, self).__init__()
        self.gat_graphsage = gat_graphsage_model

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return self.gat_graphsage(data)


class SubstructureIdentifier:
    def __init__(self):
        self.common_substructures = {
            'hydroxyl': 'O',
            'amino': 'N',
            'carboxyl': 'C(=O)O',
            'carbonyl': 'C=O',
            'ester': 'C(=O)O[C,c]',
            'amide': 'C(=O)N',
            'ether': '[C,c]O[C,c]',
            'nitro': 'N(=O)=O',
            'sulfonyl': 'S(=O)(=O)',
            'phosphate': 'P(=O)',

            'benzene': 'c1ccccc1',
            'pyridine': 'c1ccncc1',
            'pyrimidine': 'c1cncnc1',
            'imidazole': 'c1c[nH]cn1',
            'thiophene': 'c1ccsc1',
            'furan': 'c1ccoc1',
            'indole': 'c1ccc2[nH]ccc2c1',
            'quinoline': 'c1ccc2ncccc2c1',

            'piperidine': 'C1CCNCC1',
            'piperazine': 'C1CNCCN1',
            'morpholine': 'C1COCCN1',
            'pyrrolidine': 'C1CCNC1',
            'tetrahydrofuran': 'C1CCOC1',

            'methylene': 'CC',
            'ethylene': 'CCC',
            'propylene': 'CCCC',
            'vinyl': 'C=C',
            'acetylene': 'C#C',
        }

    def identify_substructures_in_molecule(self, mol):
        if mol is None:
            return {}

        found_substructures = {}

        for name, smarts in self.common_substructures.items():
            try:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
                if matches:
                    found_substructures[name] = {
                        'pattern': smarts,
                        'matches': matches,
                        'count': len(matches)
                    }
            except:
                continue

        return found_substructures

    def extract_important_substructures(self, mol, important_atom_indices, radius=2):
        if mol is None:
            return []

        important_substructures = []

        for atom_idx in important_atom_indices:
            if atom_idx >= mol.GetNumAtoms():
                continue

            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)

            if submol:
                try:
                    sub_smiles = Chem.MolToSmiles(submol)
                    if sub_smiles and len(sub_smiles) > 1:
                        important_substructures.append({
                            'center_atom': atom_idx,
                            'smiles': sub_smiles,
                            'num_atoms': submol.GetNumAtoms(),
                            'radius': radius
                        })
                except:
                    continue

        return important_substructures

    def get_functional_groups(self, mol):
        if mol is None:
            return []

        functional_groups = []

        try:
            from rdkit.Chem import Descriptors, Fragments

            groups_to_check = {
                'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
                'Aliphatic_Rings': Descriptors.NumAliphaticRings(mol),
                'Hydroxyl_Groups': Fragments.fr_Al_OH(mol),
                'Carboxyl_Groups': Fragments.fr_COO(mol),
                'Amino_Groups': Fragments.fr_NH2(mol),
                'Ester_Groups': Fragments.fr_ester(mol),
                'Ether_Groups': Fragments.fr_ether(mol),
                'Amide_Groups': Fragments.fr_amide(mol),
                'Nitro_Groups': Fragments.fr_nitro(mol),
                'Benzene_Rings': Fragments.fr_benzene(mol),
                'Pyridine_Rings': Fragments.fr_pyridine(mol),
            }

            for group_name, count in groups_to_check.items():
                if count > 0:
                    functional_groups.append({
                        'name': group_name,
                        'count': count
                    })

        except Exception as e:
            print(f"Error in functional group analysis: {e}")

        return functional_groups


class SubstructureVisualizer:
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def visualize_substructure_importance_summary(self, important_substructures, save_path=None):
        all_substructure_types = {}
        all_functional_groups = Counter()

        for struct in important_substructures:
            for sub_name, sub_info in struct['known_substructures'].items():
                if sub_name not in all_substructure_types:
                    all_substructure_types[sub_name] = {
                        'count': 0,
                        'total_importance': 0,
                        'molecules': []
                    }
                all_substructure_types[sub_name]['count'] += len(sub_info['matches'])
                all_substructure_types[sub_name]['total_importance'] += sum(sub_info['importance_scores'])
                all_substructure_types[sub_name]['molecules'].append(struct['smiles'][:30])

            for fg in struct['functional_groups']:
                all_functional_groups[fg['name']] += fg['count']

        if not all_substructure_types:
            print("No substructure data available for visualization")
            return

        sorted_substructures = sorted(all_substructure_types.items(),
                                      key=lambda x: x[1]['count'], reverse=True)[:15]

        sub_names = [item[0] for item in sorted_substructures]
        sub_counts = [item[1]['count'] for item in sorted_substructures]

        plt.figure(figsize=(14, 10))
        bars1 = plt.barh(sub_names, sub_counts, color='skyblue', alpha=0.8)
        plt.title('Substructure Frequency (Top 15)', fontsize=20, fontweight='bold', pad=25)
        plt.xlabel('Frequency Count', fontsize=18, fontweight='bold')
        plt.ylabel('Substructure Type', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis='x', alpha=0.3)

        for bar, count in zip(bars1, sub_counts):
            plt.text(bar.get_width() + max(sub_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                     str(count), ha='left', va='center', fontsize=14, fontweight='bold')

        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)
        if save_path:
            plt.savefig(f"{save_path}_subplot1_frequency.png", dpi=300, bbox_inches='tight')
            print(f"Substructure subplot 1 saved: {save_path}_subplot1_frequency.png")
        plt.show()

        avg_importance = []
        for item in sorted_substructures:
            count = item[1]['count']
            total_imp = item[1]['total_importance']
            avg_imp = total_imp / count if count > 0 else 0
            avg_importance.append(avg_imp)

        plt.figure(figsize=(14, 10))
        bars2 = plt.barh(sub_names, avg_importance, color='lightcoral', alpha=0.8)
        plt.title('Average Substructure Importance (Top 15)', fontsize=20, fontweight='bold', pad=25)
        plt.xlabel('Average Importance Score', fontsize=18, fontweight='bold')
        plt.ylabel('Substructure Type', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis='x', alpha=0.3)

        for bar, imp in zip(bars2, avg_importance):
            plt.text(bar.get_width() + max(avg_importance) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{imp:.3f}', ha='left', va='center', fontsize=14, fontweight='bold')

        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)
        if save_path:
            plt.savefig(f"{save_path}_subplot2_importance.png", dpi=300, bbox_inches='tight')
            print(f"Substructure subplot 2 saved: {save_path}_subplot2_importance.png")
        plt.show()

        top_fg = all_functional_groups.most_common(12)
        fg_names = [item[0].replace('_', ' ') for item in top_fg]
        fg_counts = [item[1] for item in top_fg]

        plt.figure(figsize=(14, 12))
        colors = plt.cm.Set3(np.linspace(0, 1, len(fg_names)))

        def autopct_func_fg(pct, idx=0):
            if idx < len(fg_counts) - 4:
                return f'{pct:.1f}%'
            else:
                return ''

        counter = [0]

        def autopct_with_counter(pct):
            result = autopct_func_fg(pct, counter[0])
            counter[0] += 1
            return result

        wedges, texts, autotexts = plt.pie(fg_counts, autopct=autopct_with_counter,
                                           colors=colors, startangle=90,
                                           textprops={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Functional Group Distribution', fontsize=20, fontweight='bold', pad=25)

        for i, autotext in enumerate(autotexts):
            autotext.set_fontsize(14)
            autotext.set_color('black')
            autotext.set_fontweight('bold')

        plt.legend(wedges, fg_names, title="Functional Groups",
                   title_fontsize=16, fontsize=14,
                   loc="center left", bbox_to_anchor=(1.1, 0.5))

        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
        if save_path:
            plt.savefig(f"{save_path}_subplot3_functional_groups.png", dpi=300, bbox_inches='tight')
            print(f"Substructure subplot 3 saved: {save_path}_subplot3_functional_groups.png")
        plt.show()

        plt.figure(figsize=(14, 10))

        x_counts = [item[1]['count'] for item in sorted_substructures[:20]]
        y_importance = []
        colors_scatter = []
        labels = []

        for item in sorted_substructures[:20]:
            count = item[1]['count']
            total_imp = item[1]['total_importance']
            avg_imp = total_imp / count if count > 0 else 0
            y_importance.append(avg_imp)

            if avg_imp > 0.6:
                colors_scatter.append('red')
            elif avg_imp > 0.4:
                colors_scatter.append('orange')
            else:
                colors_scatter.append('blue')

            labels.append(item[0])

        scatter = plt.scatter(x_counts, y_importance, c=colors_scatter,
                              s=150, alpha=0.7, edgecolors='black')

        for i, label in enumerate(labels):
            if y_importance[i] > 0.5 or x_counts[i] > 200:
                plt.annotate(label, (x_counts[i], y_importance[i]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=12, alpha=0.8, fontweight='bold')

        plt.title('Substructure Importance vs Frequency', fontsize=20, fontweight='bold', pad=25)
        plt.xlabel('Frequency Count', fontsize=18, fontweight='bold')
        plt.ylabel('Average Importance', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='High Importance (>0.6)'),
                           Patch(facecolor='orange', alpha=0.7, label='Medium Importance (0.4-0.6)'),
                           Patch(facecolor='blue', alpha=0.7, label='Low Importance (<0.4)')]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=14)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        if save_path:
            plt.savefig(f"{save_path}_subplot4_scatter.png", dpi=300, bbox_inches='tight')
            print(f"Substructure subplot 4 saved: {save_path}_subplot4_scatter.png")
        plt.show()

        print(f"\nAll four substructure subplots have been generated and saved separately!")

    def visualize_substructure_in_molecules(self, full_dataset_substructures, num_examples=6, save_path=None):
        if not full_dataset_substructures:
            print("No important substructures available for visualization")
            return

        print(f"Searching for qualified molecules from {len(full_dataset_substructures)} molecules in full dataset...")

        selected_molecules = []

        for struct in full_dataset_substructures:
            if 'target_value' not in struct or struct['target_value'] <= 6:
                continue

            has_high_importance_substructure = False
            for sub_name, sub_info in struct['known_substructures'].items():
                if sub_info['importance_scores'] and max(sub_info['importance_scores']) > 0.5:
                    has_high_importance_substructure = True
                    break

            if has_high_importance_substructure:
                selected_molecules.append(struct)
                if len(selected_molecules) >= num_examples:
                    break

        print(f"Found {len(selected_molecules)} molecules with y > 6 and high importance substructures (> 0.5)")

        if not selected_molecules:
            print("No molecules found with y > 6 and high importance substructures (> 0.5)")
            print("Falling back to molecules with any high importance substructures...")
            for struct in full_dataset_substructures:
                if len(selected_molecules) >= num_examples:
                    break

                has_high_importance_substructure = False
                for sub_name, sub_info in struct['known_substructures'].items():
                    if sub_info['importance_scores'] and max(sub_info['importance_scores']) > 0.5:
                        has_high_importance_substructure = True
                        break

                if has_high_importance_substructure:
                    selected_molecules.append(struct)

        if not selected_molecules:
            print("No molecules with high importance substructures found")
            selected_molecules = full_dataset_substructures[:num_examples]

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()

        for i, struct in enumerate(selected_molecules[:num_examples]):
            try:
                mol = Chem.MolFromSmiles(struct['smiles'])
                if mol is None:
                    continue

                best_substructure = None
                best_importance = 0

                for sub_name, sub_info in struct['known_substructures'].items():
                    if sub_info['importance_scores']:
                        avg_importance = np.mean(sub_info['importance_scores'])
                        if avg_importance > best_importance:
                            best_importance = avg_importance
                            best_substructure = (sub_name, sub_info)

                if best_substructure is None:
                    img = Draw.MolToImage(mol, size=(400, 400))
                    axes[i].imshow(img)
                    y_value = struct.get('target_value', 'N/A')
                    axes[i].set_title(f'Molecule {i + 1}\nTrue y: {y_value}' +
                                      (f' (y={y_value:.3f})' if isinstance(y_value, (int, float)) else ''),
                                      fontsize=16, fontweight='bold', pad=10)
                else:
                    sub_name, sub_info = best_substructure
                    highlight_atoms = []

                    for match in sub_info['matches']:
                        highlight_atoms.extend(match)

                    highlight_atoms = list(set(highlight_atoms))

                    img = Draw.MolToImage(mol, size=(400, 400),
                                          highlightAtoms=highlight_atoms,
                                          highlightColors={atom: (1, 0.8, 0.8) for atom in highlight_atoms})

                    axes[i].imshow(img)
                    y_value = struct.get('target_value', 'N/A')
                    title_text = f'Molecule {i + 1}: {sub_name}\n'
                    if isinstance(y_value, (int, float)):
                        title_text += f'True y: {y_value:.3f}\n'
                    else:
                        title_text += f'True y: {y_value}\n'
                    title_text += f'Importance: {best_importance:.3f}'
                    axes[i].set_title(title_text, fontsize=16, fontweight='bold', pad=10)

                axes[i].axis('off')

            except Exception as e:
                print(f"Error drawing molecule {i + 1}: {e}")
                axes[i].text(0.5, 0.5, f'Molecule {i + 1}\nDrawing Failed',
                             ha='center', va='center', transform=axes[i].transAxes,
                             fontsize=16, fontweight='bold')
                axes[i].axis('off')

        for i in range(len(selected_molecules), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Important Substructures Highlighted in Molecules (Full Dataset: y > 6, Importance > 0.5)',
                     fontsize=20, fontweight='bold', y=0.95)
        plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.2)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Highlighted molecules plot saved: {save_path}")

        plt.show()

        return selected_molecules

    def create_substructure_heatmap(self, important_substructures, save_path=None):
        if not important_substructures:
            print("No data available for heatmap")
            return

        all_substructures = set()
        for struct in important_substructures:
            all_substructures.update(struct['known_substructures'].keys())

        all_substructures = sorted(list(all_substructures))

        if len(all_substructures) == 0:
            print("No substructures found")
            return

        matrix_data = []
        molecule_labels = []

        for i, struct in enumerate(important_substructures[:40]):
            row = []
            molecule_labels.append(f"Mol{i + 1}")

            for sub_name in all_substructures:
                if sub_name in struct['known_substructures']:
                    importance_scores = struct['known_substructures'][sub_name]['importance_scores']
                    avg_importance = np.mean(importance_scores) if importance_scores else 0
                    row.append(avg_importance)
                else:
                    row.append(0)

            matrix_data.append(row)

        df_heatmap = pd.DataFrame(matrix_data,
                                  columns=all_substructures,
                                  index=molecule_labels)

        df_heatmap = df_heatmap.loc[:, (df_heatmap != 0).any(axis=0)]

        if df_heatmap.empty:
            print("Insufficient data for heatmap")
            return

        fig_width = max(18, len(df_heatmap.columns) * 1.2)
        fig_height = max(14, len(df_heatmap.index) * 0.45)

        plt.figure(figsize=(fig_width, fig_height))

        try:
            ax = sns.heatmap(df_heatmap,
                             cmap='Blues',
                             annot=False,
                             fmt='.2f',
                             cbar_kws={'label': 'Importance Score', 'shrink': 0.8},
                             xticklabels=True,
                             yticklabels=True,
                             square=False)

            plt.title('Molecule-Substructure Importance Heatmap (Top 40 Molecules)',
                      fontsize=22, fontweight='bold', pad=30)
            plt.xlabel('Substructure Type', fontsize=18, fontweight='bold', labelpad=15)
            plt.ylabel('Molecules (Top 40)', fontsize=18, fontweight='bold', labelpad=15)

            plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='normal')
            plt.yticks(rotation=0, fontsize=14, fontweight='normal')

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label('Importance Score', fontsize=18, fontweight='bold', labelpad=20)

            plt.subplots_adjust(left=0.15, bottom=0.25, right=0.85, top=0.85)

        except Exception as e:
            print(f"Error creating heatmap: {e}")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Substructure heatmap (Top 40 molecules) saved: {save_path}")

        plt.show()


class MolecularExplainer:
    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        self.atom_symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']

        self.substructure_identifier = SubstructureIdentifier()

        self.substructure_visualizer = SubstructureVisualizer()

        try:
            self.explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=100, lr=0.01),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=ModelConfig(
                    mode='regression',
                    task_level='graph',
                    return_type='raw'
                )
            )
        except Exception as e:
            print(f"Warning: Could not initialize GNNExplainer: {e}")
            self.explainer = None

    def get_atom_symbol_from_features(self, atom_features):
        atom_idx = torch.argmax(atom_features[:10]).item()
        return self.atom_symbols[atom_idx]

    def simple_gradient_explanation(self, data):
        self.model.eval()
        data = data.to(self.device)

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        data.x.requires_grad_(True)
        prediction = self.model(data.x, data.edge_index, data.batch)

        prediction.backward()

        node_importance = torch.norm(data.x.grad, dim=1)

        return {
            'node_mask': node_importance.cpu().detach(),
            'edge_mask': None,
            'prediction': prediction.cpu().detach(),
            'data': data.cpu()
        }

    def explain_molecule(self, data):
        try:
            if self.explainer is not None:
                data = data.to(self.device)

                if not hasattr(data, 'batch') or data.batch is None:
                    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

                explanation = self.explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch
                )

                return {
                    'node_mask': explanation.node_mask.cpu() if explanation.node_mask is not None else None,
                    'edge_mask': explanation.edge_mask.cpu() if explanation.edge_mask is not None else None,
                    'prediction': explanation.prediction.cpu() if hasattr(explanation, 'prediction') else None,
                    'data': data.cpu()
                }
            else:
                return self.simple_gradient_explanation(data)

        except Exception as e:
            print(f"Error in explanation: {e}")
            try:
                return self.simple_gradient_explanation(data)
            except Exception as e2:
                print(f"Error in gradient explanation: {e2}")
                return None

    def process_node_importance(self, node_mask, num_nodes):
        if node_mask is None:
            return np.full(num_nodes, 0.5)

        node_mask_np = node_mask.numpy()

        if len(node_mask_np.shape) > 1:
            if node_mask_np.shape[0] == num_nodes:
                node_colors = np.linalg.norm(node_mask_np, axis=1)
            elif node_mask_np.shape[1] == num_nodes:
                node_colors = np.linalg.norm(node_mask_np, axis=0)
            else:
                node_colors = np.full(num_nodes, np.mean(node_mask_np))
        else:
            node_colors = node_mask_np

        if len(node_colors) > num_nodes:
            node_colors = node_colors[:num_nodes]
        elif len(node_colors) < num_nodes:
            avg_importance = node_colors.mean() if len(node_colors) > 0 else 0.5
            padded_colors = np.full(num_nodes, avg_importance)
            padded_colors[:len(node_colors)] = node_colors
            node_colors = padded_colors

        if len(node_colors) > 0 and node_colors.max() > node_colors.min():
            node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())
        else:
            node_colors = np.full(num_nodes, 0.5)

        return node_colors

    def visualize_molecule_explanation(self, explanation, smiles, target_value=None, title="Molecule Explanation",
                                       save_path=None):
        if explanation is None:
            print("No explanation to visualize")
            return

        data = explanation['data']
        node_mask = explanation['node_mask']
        edge_mask = explanation['edge_mask']
        prediction = explanation['prediction']

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(400, 400))
                axes[0].imshow(img)
                axes[0].set_title(f'Original Molecule\nSMILES: {smiles[:50]}...',
                                  fontsize=16, fontweight='bold', pad=15)
                axes[0].axis('off')
        except Exception as e:
            print(f"Warning: Could not draw molecule with RDKit: {e}")
            axes[0].text(0.5, 0.5, 'Could not draw molecule', ha='center', va='center',
                         fontsize=16, fontweight='bold')
            axes[0].set_title('Original Molecule', fontsize=16, fontweight='bold')
            axes[0].axis('off')

        num_nodes = data.x.size(0)

        node_colors = self.process_node_importance(node_mask, num_nodes)

        G = nx.Graph()

        for i in range(num_nodes):
            atom_symbol = self.get_atom_symbol_from_features(data.x[i])
            G.add_node(i, symbol=atom_symbol)

        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])

        try:
            pos = nx.spring_layout(G, seed=42)
        except:
            pos = nx.circular_layout(G)

        node_x = [pos[i][0] for i in range(num_nodes)]
        node_y = [pos[i][1] for i in range(num_nodes)]

        try:
            scatter = axes[1].scatter(
                node_x, node_y,
                c=node_colors,
                cmap='RdYlBu_r',
                s=600,
                alpha=0.8,
                edgecolors='black',
                linewidth=1.5
            )

            cbar = plt.colorbar(scatter, ax=axes[1], label='Node Importance', shrink=0.8)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Node Importance', fontsize=16, fontweight='bold')

        except Exception as e:
            print(f"Error in scatter plot: {e}")
            axes[1].scatter(node_x, node_y, c='lightblue', s=600, alpha=0.8, edgecolors='black')

        try:
            for edge in G.edges():
                x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                axes[1].plot(x_coords, y_coords, 'k-', alpha=0.5, linewidth=2)
        except Exception as e:
            print(f"Warning: Could not draw edges: {e}")

        try:
            for i in range(num_nodes):
                atom_symbol = self.get_atom_symbol_from_features(data.x[i])
                axes[1].text(pos[i][0], pos[i][1], atom_symbol,
                             ha='center', va='center', fontsize=14, fontweight='bold')
        except Exception as e:
            print(f"Warning: Could not add labels: {e}")

        pred_val = prediction.item() if prediction is not None else "N/A"
        if target_value is not None:
            title_text = f'Node Importance\nTrue y: {target_value:.4f}\nPrediction: {pred_val:.4f}'
        else:
            title_text = f'Node Importance\nPrediction: {pred_val:.4f}'

        axes[1].set_title(title_text, fontsize=16, fontweight='bold', pad=15)
        axes[1].axis('off')
        axes[1].set_aspect('equal')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Image saved: {save_path}")
            except Exception as e:
                print(f"Failed to save image: {e}")

        plt.show()

    def visualize_selected_molecule(self, explanation, smiles, target_value=None, title="Selected Molecule",
                                    save_path=None):
        if explanation is None:
            print("No explanation to visualize")
            return

        data = explanation['data']
        node_mask = explanation['node_mask']
        edge_mask = explanation['edge_mask']
        prediction = explanation['prediction']

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(400, 400))
                axes[0].imshow(img)
                axes[0].set_title(f'Original Molecule\nSMILES: {smiles[:50]}...',
                                  fontsize=16, fontweight='bold', pad=15)
                axes[0].axis('off')
        except Exception as e:
            print(f"Warning: Could not draw molecule with RDKit: {e}")
            axes[0].text(0.5, 0.5, 'Could not draw molecule', ha='center', va='center',
                         fontsize=16, fontweight='bold')
            axes[0].set_title('Original Molecule', fontsize=16, fontweight='bold')
            axes[0].axis('off')

        num_nodes = data.x.size(0)

        node_colors = self.process_node_importance(node_mask, num_nodes)

        G = nx.Graph()

        for i in range(num_nodes):
            atom_symbol = self.get_atom_symbol_from_features(data.x[i])
            G.add_node(i, symbol=atom_symbol)

        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])

        try:
            pos = nx.spring_layout(G, seed=42)
        except:
            pos = nx.circular_layout(G)

        node_x = [pos[i][0] for i in range(num_nodes)]
        node_y = [pos[i][1] for i in range(num_nodes)]

        try:
            scatter = axes[1].scatter(
                node_x, node_y,
                c=node_colors,
                cmap='RdYlBu_r',
                s=600,
                alpha=0.8,
                edgecolors='black',
                linewidth=1.5
            )

            cbar = plt.colorbar(scatter, ax=axes[1], label='Importance Score', shrink=0.8)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Importance Score', fontsize=16, fontweight='bold')

        except Exception as e:
            print(f"Error in scatter plot: {e}")
            axes[1].scatter(node_x, node_y, c='lightblue', s=600, alpha=0.8, edgecolors='black')

        try:
            for edge in G.edges():
                x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                axes[1].plot(x_coords, y_coords, 'k-', alpha=0.5, linewidth=2)
        except Exception as e:
            print(f"Warning: Could not draw edges: {e}")

        try:
            for i in range(num_nodes):
                atom_symbol = self.get_atom_symbol_from_features(data.x[i])
                axes[1].text(pos[i][0], pos[i][1], atom_symbol,
                             ha='center', va='center', fontsize=14, fontweight='bold')
        except Exception as e:
            print(f"Warning: Could not add labels: {e}")

        if target_value is not None:
            title_text = f'True y: {target_value:.4f}'
        else:
            title_text = 'True y: N/A'

        axes[1].set_title(title_text, fontsize=16, fontweight='bold', pad=15)
        axes[1].axis('off')
        axes[1].set_aspect('equal')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Image saved: {save_path}")
            except Exception as e:
                print(f"Failed to save image: {e}")

        plt.show()

    def analyze_feature_importance(self, explanations_data):
        all_node_importances = []
        all_atom_types = []
        all_predictions = []

        for exp_data in explanations_data:
            explanation = exp_data['explanation']
            if explanation is None or explanation['node_mask'] is None:
                continue

            node_mask = explanation['node_mask']
            data = explanation['data']
            prediction = explanation['prediction']
            num_nodes = data.x.size(0)

            node_importances = self.process_node_importance(node_mask, num_nodes)

            for i in range(num_nodes):
                atom_features = data.x[i]
                atom_symbol = self.get_atom_symbol_from_features(atom_features)

                all_node_importances.append(node_importances[i])
                all_atom_types.append(atom_symbol)
                all_predictions.append(prediction.item() if prediction is not None else 0)

        importance_df = pd.DataFrame({
            'atom_type': all_atom_types,
            'importance': all_node_importances,
            'prediction': all_predictions
        })

        return importance_df

    def find_important_substructures(self, explanations_data, importance_threshold=0.5):
        important_substructures = []

        for exp_data in explanations_data:
            explanation = exp_data['explanation']
            smiles = exp_data['smiles']
            target_value = exp_data.get('original_target', 0)

            if explanation is None or explanation['node_mask'] is None:
                continue

            data = explanation['data']
            node_mask = explanation['node_mask']
            prediction = explanation['prediction']
            num_nodes = data.x.size(0)

            node_importances = self.process_node_importance(node_mask, num_nodes)

            important_atoms = []
            important_atom_indices = []

            for i in range(num_nodes):
                if node_importances[i] > importance_threshold:
                    atom_symbol = self.get_atom_symbol_from_features(data.x[i])
                    important_atoms.append((i, atom_symbol, node_importances[i]))
                    important_atom_indices.append(i)

            if not important_atoms:
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                all_substructures = self.substructure_identifier.identify_substructures_in_molecule(mol)

                functional_groups = self.substructure_identifier.get_functional_groups(mol)

                local_substructures = self.substructure_identifier.extract_important_substructures(
                    mol, important_atom_indices, radius=2
                )

                relevant_substructures = {}
                for sub_name, sub_info in all_substructures.items():
                    for match in sub_info['matches']:
                        if any(atom_idx in match for atom_idx in important_atom_indices):
                            if sub_name not in relevant_substructures:
                                relevant_substructures[sub_name] = {
                                    'pattern': sub_info['pattern'],
                                    'matches': [],
                                    'importance_scores': []
                                }

                            match_importance = []
                            for atom_idx in match:
                                if atom_idx in important_atom_indices:
                                    idx_in_list = important_atom_indices.index(atom_idx)
                                    match_importance.append(important_atoms[idx_in_list][2])

                            if match_importance:
                                avg_importance = np.mean(match_importance)
                                relevant_substructures[sub_name]['matches'].append(match)
                                relevant_substructures[sub_name]['importance_scores'].append(avg_importance)

                edge_index = data.edge_index.cpu().numpy()
                important_edges = []

                for i in range(edge_index.shape[1]):
                    if (edge_index[0, i] in important_atom_indices and
                            edge_index[1, i] in important_atom_indices):
                        important_edges.append((edge_index[0, i], edge_index[1, i]))

                important_substructures.append({
                    'smiles': smiles,
                    'important_atoms': important_atoms,
                    'important_edges': important_edges,
                    'prediction': prediction.item() if prediction is not None else 0,
                    'target_value': target_value,
                    'num_important_atoms': len(important_atoms),

                    'known_substructures': relevant_substructures,
                    'functional_groups': functional_groups,
                    'local_substructures': local_substructures,

                    'substructure_summary': {
                        'num_known_substructures': len(relevant_substructures),
                        'num_functional_groups': len(functional_groups),
                        'num_local_fragments': len(local_substructures)
                    }
                })

            except Exception as e:
                print(f"Error analyzing substructures for {smiles}: {e}")
                important_substructures.append({
                    'smiles': smiles,
                    'important_atoms': important_atoms,
                    'important_edges': important_edges,
                    'prediction': prediction.item() if prediction is not None else 0,
                    'target_value': target_value,
                    'num_important_atoms': len(important_atoms),
                    'known_substructures': {},
                    'functional_groups': [],
                    'local_substructures': [],
                    'substructure_summary': {
                        'num_known_substructures': 0,
                        'num_functional_groups': 0,
                        'num_local_fragments': 0
                    }
                })

        return important_substructures

    def analyze_full_dataset_substructures(self, test_csv_file, importance_threshold=0.3):
        print("\n" + "=" * 50)
        print("Full Dataset Substructure Analysis")
        print("=" * 50)

        test_df = pd.read_csv(test_csv_file)
        full_dataset_substructures = []
        successful_count = 0

        print(f"Analyzing substructures for all {len(test_df)} molecules in the dataset...")

        for i, row in test_df.iterrows():
            try:
                smiles = str(row['Smiles'])
                target_value = row.get('pchembl', 0)

                atom_features, edge_index = smiles_to_graph(smiles)
                data = Data(x=atom_features, edge_index=edge_index)

                explanation = self.explain_molecule(data)

                if explanation is not None:
                    node_mask = explanation['node_mask']
                    num_nodes = data.x.size(0)
                    node_importances = self.process_node_importance(node_mask, num_nodes)

                    important_atoms = []
                    important_atom_indices = []

                    for j in range(num_nodes):
                        if node_importances[j] > importance_threshold:
                            atom_symbol = self.get_atom_symbol_from_features(data.x[j])
                            important_atoms.append((j, atom_symbol, node_importances[j]))
                            important_atom_indices.append(j)

                    if important_atoms:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            all_substructures = self.substructure_identifier.identify_substructures_in_molecule(mol)
                            functional_groups = self.substructure_identifier.get_functional_groups(mol)
                            local_substructures = self.substructure_identifier.extract_important_substructures(
                                mol, important_atom_indices, radius=2
                            )

                            relevant_substructures = {}
                            for sub_name, sub_info in all_substructures.items():
                                for match in sub_info['matches']:
                                    if any(atom_idx in match for atom_idx in important_atom_indices):
                                        if sub_name not in relevant_substructures:
                                            relevant_substructures[sub_name] = {
                                                'pattern': sub_info['pattern'],
                                                'matches': [],
                                                'importance_scores': []
                                            }

                                        match_importance = []
                                        for atom_idx in match:
                                            if atom_idx in important_atom_indices:
                                                idx_in_list = important_atom_indices.index(atom_idx)
                                                match_importance.append(important_atoms[idx_in_list][2])

                                        if match_importance:
                                            avg_importance = np.mean(match_importance)
                                            relevant_substructures[sub_name]['matches'].append(match)
                                            relevant_substructures[sub_name]['importance_scores'].append(avg_importance)

                            edge_index = data.edge_index.cpu().numpy()
                            important_edges = []

                            for j in range(edge_index.shape[1]):
                                if (edge_index[0, j] in important_atom_indices and
                                        edge_index[1, j] in important_atom_indices):
                                    important_edges.append((edge_index[0, j], edge_index[1, j]))

                            full_dataset_substructures.append({
                                'smiles': smiles,
                                'important_atoms': important_atoms,
                                'important_edges': important_edges,
                                'prediction': explanation['prediction'].item() if explanation[
                                                                                      'prediction'] is not None else 0,
                                'target_value': target_value,
                                'num_important_atoms': len(important_atoms),
                                'known_substructures': relevant_substructures,
                                'functional_groups': functional_groups,
                                'local_substructures': local_substructures,
                                'substructure_summary': {
                                    'num_known_substructures': len(relevant_substructures),
                                    'num_functional_groups': len(functional_groups),
                                    'num_local_fragments': len(local_substructures)
                                }
                            })
                            successful_count += 1

            except Exception as e:
                continue

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(test_df)}, successful: {successful_count}")

        print(f"Full dataset analysis completed: {successful_count} molecules with important substructures")
        return full_dataset_substructures

    def plot_feature_importance_summary(self, importance_df, save_path=None):
        if importance_df.empty:
            print("No importance data to plot")
            return

        custom_colors = ['#98CFE6', '#ADE7A8', '#F39F4E', '#EEB7D3', '#DBDAD3', '#FFDF97']

        try:
            atom_importance = importance_df.groupby('atom_type')['importance'].agg(['mean', 'std', 'count'])
            atom_importance = atom_importance.sort_values('mean', ascending=False)
            atom_counts = importance_df['atom_type'].value_counts()

            plt.figure(figsize=(14, 10))
            for i, (atom_type, stats) in enumerate(atom_importance.iterrows()):
                color_idx = i % len(custom_colors)
                plt.bar(atom_type, stats['mean'], yerr=stats['std'], capsize=5,
                        alpha=0.8, color=custom_colors[color_idx], edgecolor='white', linewidth=1)

            plt.title('Average Atom Importance', fontsize=20, fontweight='bold', pad=25)
            plt.xlabel('Atom Type', fontsize=18, fontweight='bold')
            plt.ylabel('Average Importance', fontsize=18, fontweight='bold')
            plt.xticks(rotation=45, fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(axis='y', alpha=0.3)

            for i, (atom_type, stats) in enumerate(atom_importance.iterrows()):
                plt.text(i, stats['mean'] + stats['std'] + 0.01, f'{stats["mean"]:.3f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
            if save_path:
                plt.savefig(f"{save_path}_subplot1_atom_importance.png", dpi=300, bbox_inches='tight')
                print(f"Subplot 1 saved: {save_path}_subplot1_atom_importance.png")
            plt.show()

            plt.figure(figsize=(14, 10))

            total_contribution = importance_df.groupby('atom_type')['importance'].sum().sort_values(ascending=False)
            cumulative_contribution = total_contribution.cumsum() / total_contribution.sum() * 100

            plt.plot(range(1, len(cumulative_contribution) + 1), cumulative_contribution.values,
                     'o-', linewidth=6, markersize=12, color=custom_colors[0], markerfacecolor='white',
                     markeredgewidth=3, markeredgecolor=custom_colors[0])

            plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Contribution', linewidth=4)

            plt.xlabel('Atom Type Rank', fontsize=18, fontweight='bold')
            plt.ylabel('Cumulative Contribution (%)', fontsize=18, fontweight='bold')
            plt.title('Cumulative Importance Contribution', fontsize=20, fontweight='bold', pad=25)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=16, loc='lower right')

            for i, (atom_type, contrib) in enumerate(cumulative_contribution.items()):
                if i < 8:
                    plt.annotate(atom_type, (i + 1, contrib), xytext=(0, 15),
                                 textcoords='offset points', fontsize=14, fontweight='bold',
                                 ha='center', va='bottom')

            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
            if save_path:
                plt.savefig(f"{save_path}_subplot2_cumulative_contribution.png", dpi=300, bbox_inches='tight')
                print(f"Subplot 2 saved: {save_path}_subplot2_cumulative_contribution.png")
            plt.show()

            plt.figure(figsize=(14, 12))
            pie_colors = [custom_colors[i % len(custom_colors)] for i in range(len(atom_counts))]

            def autopct_func_atoms(pct, idx=0):
                if idx < len(atom_counts) - 5:
                    return f'{pct:.1f}%'
                else:
                    return ''

            counter = [0]

            def autopct_with_counter_atoms(pct):
                result = autopct_func_atoms(pct, counter[0])
                counter[0] += 1
                return result

            wedges, texts, autotexts = plt.pie(atom_counts.values, autopct=autopct_with_counter_atoms,
                                               colors=pie_colors, startangle=90,
                                               textprops={'fontsize': 16, 'fontweight': 'bold'})
            plt.title('Atom Type Distribution', fontsize=20, fontweight='bold', pad=25)

            for autotext in autotexts:
                autotext.set_fontsize(16)
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            plt.legend(wedges, atom_counts.index, title="Atom Types",
                       title_fontsize=16, fontsize=14,
                       loc="center left", bbox_to_anchor=(1.1, 0.5))

            plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
            if save_path:
                plt.savefig(f"{save_path}_subplot3_atom_distribution.png", dpi=300, bbox_inches='tight')
                print(f"Subplot 3 saved: {save_path}_subplot3_atom_distribution.png")
            plt.show()

            plt.figure(figsize=(14, 10))

            atom_types_for_box = []
            importance_values_for_box = []

            for atom_type in atom_importance.index[:10]:
                atom_data = importance_df[importance_df['atom_type'] == atom_type]['importance']
                atom_types_for_box.extend([atom_type] * len(atom_data))
                importance_values_for_box.extend(atom_data.values)

            box_df = pd.DataFrame({
                'atom_type': atom_types_for_box,
                'importance': importance_values_for_box
            })

            import seaborn as sns

            sns.set_context("paper", font_scale=1.5)

            sns.boxplot(data=box_df, x='atom_type', y='importance',
                        palette=custom_colors[:len(atom_importance.index[:10])])

            plt.title('Importance Distribution by Atom Type', fontsize=20, fontweight='bold', pad=25)
            plt.xlabel('Atom Type', fontsize=18, fontweight='bold')
            plt.ylabel('Importance Score', fontsize=18, fontweight='bold')
            plt.xticks(rotation=45, fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(axis='y', alpha=0.3)

            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
            if save_path:
                plt.savefig(f"{save_path}_subplot4_importance_distribution.png", dpi=300, bbox_inches='tight')
                print(f"Subplot 4 saved: {save_path}_subplot4_importance_distribution.png")
            plt.show()

            sns.reset_defaults()

            print(f"\nAll four subplots have been generated and saved separately!")

        except Exception as e:
            print(f"Error plotting feature importance: {e}")

    def visualize_all_substructures(self, important_substructures, full_dataset_substructures,
                                    base_save_path="substructure_analysis"):
        print("\nGenerating substructure visualizations...")

        print("  - Generating substructure importance summary (4 separate plots)...")
        self.substructure_visualizer.visualize_substructure_importance_summary(
            important_substructures,
            save_path=f"{base_save_path}_importance_summary"
        )

        print("  - Generating highlighted molecules from full dataset...")
        selected_molecules = self.substructure_visualizer.visualize_substructure_in_molecules(
            full_dataset_substructures,
            num_examples=6,
            save_path=f"{base_save_path}_highlighted_molecules.png"
        )

        print("  - Generating substructure heatmap...")
        self.substructure_visualizer.create_substructure_heatmap(
            important_substructures,
            save_path=f"{base_save_path}_heatmap.png"
        )

        print("Substructure visualization completed!")

        return selected_molecules


def load_best_model(model_path='best_model.pth'):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        gat_graphsage_model = GAT_GraphSAGE(n_output=1, num_features_xd=35)

        gat_graphsage_model.load_state_dict(checkpoint['gat_graphsage_model_state_dict'])

        scaler = checkpoint.get('scaler', None)

        print("Model loaded successfully!")
        return gat_graphsage_model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


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


def quick_importance_analysis_all(test_csv_file, model_path):
    print("Stage 1: Quick analysis of all molecules...")
    print("-" * 50)

    test_df = pd.read_csv(test_csv_file)
    gat_model, scaler = load_best_model(model_path)
    explainable_model = ExplainableGATGraphSAGE(gat_model)
    explainer = MolecularExplainer(explainable_model)

    molecule_info = []
    successful_count = 0

    for i, row in test_df.iterrows():
        try:
            smiles = str(row['Smiles'])
            atom_features, edge_index = smiles_to_graph(smiles)
            data = Data(x=atom_features, edge_index=edge_index)

            explanation = explainer.simple_gradient_explanation(data)

            if explanation:
                molecule_info.append({
                    'index': i,
                    'smiles': smiles,
                    'prediction': explanation['prediction'].item(),
                    'avg_importance': explanation['node_mask'].mean().item(),
                    'max_importance': explanation['node_mask'].max().item(),
                    'std_importance': explanation['node_mask'].std().item(),
                    'num_atoms': data.x.size(0),
                    'target': row.get('pchembl', 0)
                })
                successful_count += 1

        except Exception as e:
            continue

        if (i + 1) % 100 == 0:
            print(f"  Quick analysis progress: {i + 1}/961, successful: {successful_count}")

    print(f"Stage 1 completed: Successfully analyzed {successful_count} molecules")
    return molecule_info


def stratified_sample_by_column(df, column, target_count):
    try:
        df_copy = df.copy()
        df_copy['quartile'] = pd.qcut(df_copy[column], q=5, labels=False, duplicates='drop')

        sampled_indices = []
        samples_per_quartile = target_count // 5

        for q in df_copy['quartile'].unique():
            if pd.isna(q):
                continue
            quartile_data = df_copy[df_copy['quartile'] == q]
            if len(quartile_data) > 0:
                sample_count = min(samples_per_quartile, len(quartile_data))
                sampled = quartile_data.sample(n=sample_count, random_state=42)
                sampled_indices.extend(sampled['index'].tolist())

        return sampled_indices
    except Exception as e:
        print(f"Stratified sampling failed, using random sampling: {e}")
        return df.sample(n=min(target_count, len(df)), random_state=42)['index'].tolist()


def select_representative_molecules(molecule_info, target_count=200):
    print("\nStage 2: Selecting representative molecules for detailed analysis...")
    print("-" * 50)

    df = pd.DataFrame(molecule_info)

    if len(df) < target_count:
        print(f"Available molecules ({len(df)}) < target count ({target_count}), will analyze all available")
        return df['index'].tolist()

    selected_indices = []

    print("  - Stratified sampling by prediction values...")
    pred_samples = stratified_sample_by_column(df, 'prediction', int(target_count * 0.4))
    selected_indices.extend(pred_samples)

    print("  - Stratified sampling by average importance...")
    remaining_df = df[~df['index'].isin(selected_indices)]
    if len(remaining_df) > 0:
        imp_samples = stratified_sample_by_column(remaining_df, 'avg_importance', int(target_count * 0.3))
        selected_indices.extend(imp_samples)

    print("  - Stratified sampling by molecule size...")
    remaining_df = df[~df['index'].isin(selected_indices)]
    if len(remaining_df) > 0:
        size_samples = stratified_sample_by_column(remaining_df, 'num_atoms', int(target_count * 0.2))
        selected_indices.extend(size_samples)

    print("  - Random sampling for remaining molecules...")
    remaining_df = df[~df['index'].isin(selected_indices)]
    remaining_count = target_count - len(selected_indices)
    if remaining_count > 0 and len(remaining_df) > 0:
        random_samples = remaining_df.sample(n=min(remaining_count, len(remaining_df)),
                                             random_state=42)['index'].tolist()
        selected_indices.extend(random_samples)

    print(f"Selected {len(selected_indices)} representative molecules for detailed analysis")

    selected_df = df[df['index'].isin(selected_indices)]
    print(f"\nSampling statistics:")
    print(f"  Prediction range: {selected_df['prediction'].min():.3f} - {selected_df['prediction'].max():.3f}")
    print(f"  Importance range: {selected_df['avg_importance'].min():.3f} - {selected_df['avg_importance'].max():.3f}")
    print(f"  Molecule size range: {selected_df['num_atoms'].min()} - {selected_df['num_atoms'].max()} atoms")

    return selected_indices


def perform_detailed_analysis(test_csv_file, model_path, selected_indices, importance_threshold=0.3):
    print("\nStage 3: Detailed substructure analysis of selected molecules...")
    print("-" * 50)

    test_df = pd.read_csv(test_csv_file)
    gat_model, scaler = load_best_model(model_path)
    explainable_model = ExplainableGATGraphSAGE(gat_model)
    explainer = MolecularExplainer(explainable_model)

    explanations_data = []
    successful_count = 0

    for i, idx in enumerate(selected_indices):
        row = test_df.iloc[idx]
        smiles = str(row['Smiles'])

        try:
            atom_features, edge_index = smiles_to_graph(smiles)
            data = Data(x=atom_features, edge_index=edge_index)
            explanation = explainer.explain_molecule(data)

            if explanation is not None:
                explanations_data.append({
                    'smiles': smiles,
                    'explanation': explanation,
                    'original_target': row.get('pchembl', 0),
                    'original_index': idx
                })
                successful_count += 1

            if (i + 1) % 20 == 0:
                print(f"  Detailed analysis progress: {i + 1}/{len(selected_indices)}, successful: {successful_count}")

        except Exception as e:
            print(f"  Molecule {idx} processing failed: {e}")
            continue

    print(f"Stage 3 completed: Successfully explained {successful_count} molecules")

    print("\nPerforming feature importance analysis...")
    importance_df = explainer.analyze_feature_importance(explanations_data)

    print("Identifying important substructures from sampled data...")
    important_substructures = explainer.find_important_substructures(
        explanations_data, importance_threshold
    )

    print("Analyzing full dataset for qualified molecules...")
    full_dataset_substructures = explainer.analyze_full_dataset_substructures(
        test_csv_file, importance_threshold
    )

    return {
        'explanations_data': explanations_data,
        'importance_df': importance_df,
        'important_substructures': important_substructures,
        'full_dataset_substructures': full_dataset_substructures,
        'explainer': explainer
    }


def combine_quick_and_detailed_results(quick_results, detailed_results):
    print("\nStage 4: Combining and summarizing analysis results...")
    print("-" * 50)

    quick_df = pd.DataFrame(quick_results)

    global_stats = {
        'total_molecules_analyzed': len(quick_df),
        'prediction_range': (quick_df['prediction'].min(), quick_df['prediction'].max()),
        'prediction_mean': quick_df['prediction'].mean(),
        'prediction_std': quick_df['prediction'].std(),
        'avg_importance_range': (quick_df['avg_importance'].min(), quick_df['avg_importance'].max()),
        'avg_importance_mean': quick_df['avg_importance'].mean(),
        'molecule_size_range': (quick_df['num_atoms'].min(), quick_df['num_atoms'].max()),
        'molecule_size_mean': quick_df['num_atoms'].mean()
    }

    combined_results = {
        'global_statistics': global_stats,
        'quick_analysis_results': quick_results,
        'detailed_analysis_results': detailed_results,
        'summary': {
            'total_molecules': len(quick_df),
            'detailed_molecules': len(detailed_results['explanations_data']),
            'identified_substructures': len(detailed_results['important_substructures']),
            'full_dataset_substructures': len(detailed_results['full_dataset_substructures']),
            'analysis_completeness': len(detailed_results['explanations_data']) / len(quick_df) * 100
        }
    }

    return combined_results


def hybrid_analysis_strategy(test_csv_file, model_path, target_detailed_count=200, importance_threshold=0.3):
    print("=" * 70)
    print("Hybrid Analysis Strategy: Molecular Model Explainability Analysis")
    print("=" * 70)
    print(f"Data file: {test_csv_file}")
    print(f"Model file: {model_path}")
    print(f"Target detailed analysis count: {target_detailed_count}")
    print(f"Importance threshold: {importance_threshold}")
    print("=" * 70)

    quick_results = quick_importance_analysis_all(test_csv_file, model_path)

    if not quick_results:
        print("Quick analysis failed, cannot continue")
        return None

    representative_indices = select_representative_molecules(
        quick_results,
        target_count=target_detailed_count
    )

    detailed_results = perform_detailed_analysis(
        test_csv_file,
        model_path,
        representative_indices,
        importance_threshold
    )

    final_results = combine_quick_and_detailed_results(quick_results, detailed_results)

    generate_comprehensive_report(final_results)

    return final_results


def generate_comprehensive_report(results):
    print("\n" + "=" * 70)
    print("Comprehensive Analysis Report")
    print("=" * 70)

    global_stats = results['global_statistics']
    summary = results['summary']

    print(f"\n[Global Statistics - Based on {global_stats['total_molecules_analyzed']} molecules]")
    print("-" * 50)
    print(f"Prediction distribution:")
    print(f"  Range: {global_stats['prediction_range'][0]:.3f} - {global_stats['prediction_range'][1]:.3f}")
    print(f"  Mean: {global_stats['prediction_mean']:.3f} ± {global_stats['prediction_std']:.3f}")

    print(f"\nImportance distribution:")
    print(f"  Range: {global_stats['avg_importance_range'][0]:.3f} - {global_stats['avg_importance_range'][1]:.3f}")
    print(f"  Mean: {global_stats['avg_importance_mean']:.3f}")

    print(f"\nMolecule size distribution:")
    print(f"  Range: {global_stats['molecule_size_range'][0]} - {global_stats['molecule_size_range'][1]} atoms")
    print(f"  Mean: {global_stats['molecule_size_mean']:.1f} atoms")

    detailed = results['detailed_analysis_results']

    print(f"\n[Detailed Analysis Results - Based on {summary['detailed_molecules']} representative molecules]")
    print("-" * 50)

    if not detailed['importance_df'].empty:
        print("Atom importance statistics:")
        atom_stats = detailed['importance_df'].groupby('atom_type')['importance'].agg(['mean', 'std', 'count'])
        atom_stats = atom_stats.sort_values('mean', ascending=False)

        for atom_type, stats in atom_stats.head(8).iterrows():
            print(
                f"  {atom_type}: Average importance {stats['mean']:.3f} (±{stats['std']:.3f}), appeared {stats['count']} times")

    important_substructures = detailed['important_substructures']
    full_dataset_substructures = detailed['full_dataset_substructures']

    if important_substructures:
        print(f"\nFound {len(important_substructures)} molecules containing important substructures (from sample)")
        print(
            f"Found {len(full_dataset_substructures)} molecules containing important substructures (from full dataset)")

        all_substructure_types = {}
        all_functional_groups = Counter()

        for struct in important_substructures:
            for sub_name, sub_info in struct['known_substructures'].items():
                if sub_name not in all_substructure_types:
                    all_substructure_types[sub_name] = {
                        'count': 0,
                        'total_importance': 0,
                        'molecules': []
                    }
                all_substructure_types[sub_name]['count'] += len(sub_info['matches'])
                all_substructure_types[sub_name]['total_importance'] += sum(sub_info['importance_scores'])
                all_substructure_types[sub_name]['molecules'].append(struct['smiles'][:30])

            for fg in struct['functional_groups']:
                all_functional_groups[fg['name']] += fg['count']

        print(f"\nMost common important substructures (Top 10):")
        sorted_substructures = sorted(all_substructure_types.items(),
                                      key=lambda x: x[1]['count'], reverse=True)

        for i, (sub_name, info) in enumerate(sorted_substructures[:10], 1):
            avg_importance = info['total_importance'] / info['count'] if info['count'] > 0 else 0
            print(f"  {i:2d}. {sub_name:15s}: appeared {info['count']:3d} times, avg importance {avg_importance:.3f}")

        print(f"\nMost common functional groups (Top 10):")
        for i, (fg_name, count) in enumerate(all_functional_groups.most_common(10), 1):
            print(f"  {i:2d}. {fg_name:20s}: {count:3d} times")

    print(f"\n[Analysis Completeness]")
    print("-" * 30)
    print(f"Total molecules: {summary['total_molecules']}")
    print(f"Detailed analysis molecules: {summary['detailed_molecules']}")
    print(f"Full dataset substructure analysis: {summary['full_dataset_substructures']}")
    print(f"Analysis coverage: {summary['analysis_completeness']:.1f}%")
    print(f"Identified important substructures: {summary['identified_substructures']}")

    if not detailed['importance_df'].empty:
        print(f"\nGenerating visualizations...")
        try:
            detailed['explainer'].plot_feature_importance_summary(
                detailed['importance_df'],
                'enhanced_feature_importance_analysis'
            )

            selected_highlighted_molecules = detailed['explainer'].visualize_all_substructures(
                detailed['important_substructures'],
                detailed['full_dataset_substructures'],
                base_save_path="substructure_analysis"
            )

            print("  - Generating selected molecule visualizations from full dataset (True y only)...")

            qualified_molecules = []
            for struct in full_dataset_substructures:
                if (struct.get('target_value', 0) > 6 and
                        any(max(sub_info['importance_scores']) > 0.5
                            for sub_info in struct['known_substructures'].values()
                            if sub_info['importance_scores'])):
                    qualified_molecules.append(struct)
                    if len(qualified_molecules) >= 6:
                        break

            print(f"    Found {len(qualified_molecules)} qualified molecules from full dataset")

            for i, struct_data in enumerate(qualified_molecules):
                try:
                    smiles = struct_data['smiles']
                    target_value = struct_data.get('target_value')

                    atom_features, edge_index = smiles_to_graph(smiles)
                    data = Data(x=atom_features, edge_index=edge_index)
                    explanation = detailed['explainer'].explain_molecule(data)

                    if explanation is not None:
                        print(f"    Generating visualization for selected molecule {i + 1}...")
                        detailed['explainer'].visualize_selected_molecule(
                            explanation, smiles,
                            target_value=target_value,
                            title=f'Selected Molecule {i + 1}',
                            save_path=f'selected_molecule_{i + 1}.png'
                        )

                except Exception as e:
                    print(f"    Error generating visualization for molecule {i + 1}: {e}")

        except Exception as e:
            print(f"Error generating visualizations: {e}")

    print(f"\n" + "=" * 70)
    print("Analysis completed!")
    print("=" * 70)
    print(f"Generated files:")
    print(f"- enhanced_feature_importance_analysis_subplot1_atom_importance.png: Average Atom Importance")
    print(f"- enhanced_feature_importance_analysis_subplot2_cumulative_contribution.png: Cumulative contribution")
    print(f"- enhanced_feature_importance_analysis_subplot3_atom_distribution.png: Atom type distribution")
    print(f"- enhanced_feature_importance_analysis_subplot4_importance_distribution.png: Importance distribution")
    print(f"- substructure_analysis_importance_summary_subplot1_frequency.png: Substructure frequency")
    print(f"- substructure_analysis_importance_summary_subplot2_importance.png: Substructure importance")
    print(f"- substructure_analysis_importance_summary_subplot3_functional_groups.png: Functional group distribution")
    print(f"- substructure_analysis_importance_summary_subplot4_scatter.png: Importance vs frequency scatter")
    print(
        f"- substructure_analysis_highlighted_molecules.png: Highlighted substructure molecules (Full Dataset: y > 6, Importance > 0.5)")
    print(f"- substructure_analysis_heatmap.png: Substructure-molecule heatmap (Top 40 molecules)")
    print(
        f"- selected_molecule_1.png to selected_molecule_6.png: Selected molecules (Full Dataset: y > 6, Importance > 0.5, True y only, No predict values)")


def display_analysis_results(results):
    if results is None:
        print("No analysis results to display")
        return

    print("\n" + "=" * 60)
    print("Detailed Analysis Results View")
    print("=" * 60)

    quick_df = pd.DataFrame(results['quick_analysis_results'])
    print(f"\n=== Global prediction distribution (based on {len(quick_df)} molecules) ===")
    print(quick_df['prediction'].describe())

    print(f"\n=== Importance distribution ===")
    print(quick_df['avg_importance'].describe())

    substructures = results['detailed_analysis_results']['important_substructures']
    full_dataset_substructures = results['detailed_analysis_results']['full_dataset_substructures']

    print(f'\n=== Important substructure analysis ===')
    print(f'Found {len(substructures)} molecules containing important substructures (from sample)')
    print(f'Found {len(full_dataset_substructures)} molecules containing important substructures (from full dataset)')

    all_substructure_types = {}
    for struct in substructures:
        for sub_name, sub_info in struct['known_substructures'].items():
            if sub_name not in all_substructure_types:
                all_substructure_types[sub_name] = 0
            all_substructure_types[sub_name] += len(sub_info['matches'])

    if all_substructure_types:
        print(f"\nMost common chemical substructures:")
        sorted_subs = sorted(all_substructure_types.items(), key=lambda x: x[1], reverse=True)
        for sub_name, count in sorted_subs[:10]:
            print(f"  {sub_name}: {count} times")

    importance_df = results['detailed_analysis_results']['importance_df']
    if not importance_df.empty:
        print(f"\n=== Atom importance analysis ===")
        atom_stats = importance_df.groupby('atom_type')['importance'].agg(['mean', 'std', 'count'])
        atom_stats = atom_stats.sort_values('mean', ascending=False)
        print("Atom type importance ranking:")
        for atom_type, stats in atom_stats.head(8).iterrows():
            print(
                f"  {atom_type}: Average importance {stats['mean']:.3f} (±{stats['std']:.3f}), appeared {stats['count']} times")

    print(f"\n=== Specific molecule examples (from full dataset) ===")
    qualified_count = 0
    for struct in full_dataset_substructures[:10]:
        if (struct.get('target_value', 0) > 6 and
                any(max(sub_info['importance_scores']) > 0.5
                    for sub_info in struct['known_substructures'].values()
                    if sub_info['importance_scores'])):
            qualified_count += 1
            print(f"\nQualified Molecule {qualified_count}:")
            print(f"  SMILES: {struct['smiles'][:60]}...")
            print(f"  Prediction: {struct['prediction']:.4f}")
            print(f"  True y value: {struct.get('target_value', 'N/A')}")
            print(f"  Important atoms: {struct['num_important_atoms']}")

            if struct['known_substructures']:
                print(f"  Found substructures:")
                for sub_name, sub_info in list(struct['known_substructures'].items())[:3]:
                    avg_imp = np.mean(sub_info['importance_scores']) if sub_info['importance_scores'] else 0
                    print(f"    • {sub_name}: average importance {avg_imp:.3f}")

            if qualified_count >= 3:
                break

    print(
        f"\nTotal qualified molecules found in full dataset: {sum(1 for struct in full_dataset_substructures if struct.get('target_value', 0) > 6 and any(max(sub_info['importance_scores']) > 0.5 for sub_info in struct['known_substructures'].values() if sub_info['importance_scores']))}")


if __name__ == "__main__":
    results = hybrid_analysis_strategy(
        test_csv_file='D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv',
        model_path='D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\best_model.pth',
        target_detailed_count=200,
        importance_threshold=0.3
    )

    if results is not None:
        print("\nAnalysis results saved in 'results' variable")
        print("You can access them through:")
        print("- results['global_statistics']: Global statistics")
        print("- results['quick_analysis_results']: Quick analysis results")
        print("- results['detailed_analysis_results']: Detailed analysis results")
        print("- results['summary']: Analysis summary")

        display_analysis_results(results)

        print("\nYou can use the following commands for further analysis:")
        print("# View specific atom type importance")
        print("importance_df = results['detailed_analysis_results']['importance_df']")
        print("print(importance_df[importance_df['atom_type'] == 'N'].describe())")

        print("\n# View molecules with highest predictions")
        print("quick_df = pd.DataFrame(results['quick_analysis_results'])")
        print("top_predictions = quick_df.nlargest(10, 'prediction')")
        print("print(top_predictions[['smiles', 'prediction', 'avg_importance']])")