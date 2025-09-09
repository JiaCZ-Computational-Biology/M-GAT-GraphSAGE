# Import necessary libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori, association_rules
from rdkit.Chem import DataStructs
from collections import Counter
from scipy.stats import contingency

import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans',
                                          'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

file_path = r"D:\pycharm\gutingle\pythonProject2\突出核蛋白\ki_data.csv"
data = pd.read_csv(file_path)

smiles_column = "Smiles"
pchembl_column = "pchembl"

if smiles_column not in data.columns or pchembl_column not in data.columns:
    raise ValueError(f"File must contain '{smiles_column}' and '{pchembl_column}' columns!")

data = data[[smiles_column, pchembl_column]].rename(columns={smiles_column: "SMILES", pchembl_column: "pChEMBL"})

data = data.dropna(subset=['SMILES'])
data['SMILES'] = data['SMILES'].astype(str)


def safe_mol_from_smiles(smiles):
    """Safely convert SMILES to molecule object, return None if failed"""
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None


data['Mol'] = data['SMILES'].apply(safe_mol_from_smiles)
data = data[data['Mol'].notnull()]

threshold = data['pChEMBL'].median()
data['Affinity_Group'] = data['pChEMBL'].apply(lambda x: 'High' if x >= threshold else 'Low')

functional_groups = [
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "TPSA",
    "MolLogP",
    "MolWt",
    "HeavyAtomCount",
    "BertzCT"
]

for fg in functional_groups:
    if hasattr(Descriptors, fg):
        data[fg] = data['Mol'].apply(lambda mol: getattr(Descriptors, fg)(mol) if mol else None)
    elif hasattr(rdMolDescriptors, fg):
        data[fg] = data['Mol'].apply(lambda mol: getattr(rdMolDescriptors, fg)(mol) if mol else None)
    else:
        raise AttributeError(f"Cannot find descriptor: {fg}")


def calculate_odds_ratio_and_ci(high_values, low_values, alpha=0.05):
    """
    Calculate odds ratio and confidence interval for continuous variables
    Dichotomize continuous variables: above median as 1, below median as 0
    """
    from scipy.stats import norm

    high_median = np.median(high_values)
    low_median = np.median(low_values)
    overall_median = np.median(np.concatenate([high_values, low_values]))

    high_above_median = np.sum(high_values > overall_median)
    high_below_median = len(high_values) - high_above_median

    low_above_median = np.sum(low_values > overall_median)
    low_below_median = len(low_values) - low_above_median

    if high_below_median == 0 or low_above_median == 0:
        high_above_median += 0.5
        high_below_median += 0.5
        low_above_median += 0.5
        low_below_median += 0.5

    odds_ratio = (high_above_median * low_below_median) / (high_below_median * low_above_median)

    se_log_or = np.sqrt(1 / high_above_median + 1 / high_below_median + 1 / low_above_median + 1 / low_below_median)

    z_score = norm.ppf(1 - alpha / 2)
    log_or = np.log(odds_ratio)
    ci_lower = np.exp(log_or - z_score * se_log_or)
    ci_upper = np.exp(log_or + z_score * se_log_or)

    return odds_ratio, ci_lower, ci_upper


forest_data = []

for desc in functional_groups:
    high_values = data[data['Affinity_Group'] == 'High'][desc].dropna()
    low_values = data[data['Affinity_Group'] == 'Low'][desc].dropna()

    if len(high_values) > 0 and len(low_values) > 0:
        t_stat, p_value = ttest_ind(high_values, low_values)

        or_value, ci_lower, ci_upper = calculate_odds_ratio_and_ci(high_values, low_values)

        forest_data.append({
            'Feature': desc,
            'OR': or_value,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        })

forest_df = pd.DataFrame(forest_data)


def create_forest_plot(forest_df, title="Molecular Descriptors Odds Ratio Forest Plot"):
    """Create professional forest plot"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10),
                                   gridspec_kw={'width_ratios': [3, 2]})

    forest_df_sorted = forest_df.sort_values('OR', ascending=True)

    y_spacing = 1.0
    y_pos = np.arange(len(forest_df_sorted)) * y_spacing

    colors = ['#FFB3BA' if sig else '#B3D9FF' for sig in forest_df_sorted['Significant']]

    for i, (idx, row) in enumerate(forest_df_sorted.iterrows()):
        y = i * y_spacing

        ax1.plot([row['CI_Lower'], row['CI_Upper']], [y, y],
                 color='#CCCCCC', linewidth=1.5, alpha=0.8, zorder=1)

        ax1.plot([row['CI_Lower'], row['CI_Upper']], [y, y],
                 color='#CCCCCC', marker='|', markersize=8, linewidth=1.5, alpha=0.8, zorder=2)

        ax1.plot(row['OR'], y, 'o', color=colors[i], markersize=12,
                 markeredgecolor='white', markeredgewidth=1.5, zorder=3)

    ax1.axvline(x=1, color='#999999', linestyle='--', alpha=0.9, linewidth=2, zorder=0)

    ax1.set_xscale('log')

    ax1.set_xlabel('Odds Ratio (log scale)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([])

    x_min = 0.6
    x_max = 1.2

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(-0.5, (len(forest_df_sorted) - 1) * y_spacing + 0.5)

    tick_positions = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    ax1.set_xticks(tick_positions)

    ax1.set_xticklabels([f'{tick:.1f}' for tick in tick_positions], fontsize=11)

    ax1.grid(True, alpha=0.2, color='#F0F0F0', linewidth=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFB3BA', label='Significant (p < 0.05)', edgecolor='white'),
        Patch(facecolor='#B3D9FF', label='Non-significant (p >= 0.05)', edgecolor='white')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', frameon=True,
               fancybox=True, shadow=True, fontsize=11)

    ax2.axis('off')

    y_header = (len(forest_df_sorted) - 1) * y_spacing + 0.5
    headers = ['Feature', 'OR', '95% CI', 'P-value']
    x_positions = [0.0, 0.35, 0.6, 0.85]

    for i, header in enumerate(headers):
        ax2.text(x_positions[i], y_header, header, fontweight='bold', fontsize=11, ha='center', color='black')

    for i, (idx, row) in enumerate(forest_df_sorted.iterrows()):
        y = i * y_spacing
        weight = 'bold' if row['Significant'] else 'normal'

        ax2.text(x_positions[0], y, row['Feature'], fontsize=10, ha='center', va='center',
                 color='black', fontweight=weight)

        or_text = f"{row['OR']:.2f}"
        ax2.text(x_positions[1], y, or_text, fontsize=10, ha='center', va='center',
                 color='black', fontweight=weight)

        ci_text = f"({row['CI_Lower']:.2f}, {row['CI_Upper']:.2f})"
        ax2.text(x_positions[2], y, ci_text, fontsize=10, ha='center', va='center',
                 color='black', fontweight=weight)

        p_text = f"{row['P_Value']:.3f}" if row['P_Value'] >= 0.001 else "<0.001"
        ax2.text(x_positions[3], y, p_text, fontsize=10, ha='center', va='center',
                 color='black', fontweight=weight)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, (len(forest_df_sorted) - 1) * y_spacing + 0.8)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    ax1.text(0.98, 0.02,
             'OR > 1: Higher probability in high affinity group\nOR < 1: Higher probability in low affinity group',
             transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA', alpha=0.9, edgecolor='#E8E8E8'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    return fig


fig = create_forest_plot(forest_df)

print("Forest Plot Statistical Results:")
print("=" * 80)
for idx, row in forest_df.iterrows():
    significance = "Significant" if row['Significant'] else "Non-significant"
    print(
        f"{row['Feature']:<20} OR: {row['OR']:.3f} (95% CI: {row['CI_Lower']:.3f}-{row['CI_Upper']:.3f}) P: {row['P_Value']:.3e} ({significance})")

scaler = StandardScaler()
X = scaler.fit_transform(data[functional_groups].dropna())

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

data_clean = data.dropna(subset=functional_groups)
data_clean = data_clean.copy()
data_clean['PCA1'] = pca_result[:, 0]
data_clean['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_clean, x='PCA1', y='PCA2', hue='Affinity_Group',
                palette=['#A8D8A8', '#F5C49A'])
plt.title("PCA Analysis of High vs Low Affinity Compounds")
plt.legend(title="Affinity Group", frameon=False)
plt.tight_layout()
plt.show()

X = data[functional_groups].dropna()
y = data.loc[X.index, 'Affinity_Group'].apply(lambda x: 1 if x == 'High' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

importances = pd.DataFrame({"Feature": functional_groups, "Importance": clf.feature_importances_})
importances = importances.sort_values("Importance", ascending=False)
print("\nDescriptor Importance:")
print(importances)

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))