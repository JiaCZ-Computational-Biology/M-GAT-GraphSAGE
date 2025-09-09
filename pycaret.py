import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import traceback
import sys
import os
import time
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)
random.seed(seed)


def one_of_k_encoding_unk(x, valid_entries):
    """One-hot encoding function for handling unknown values"""
    if x not in valid_entries:
        x = 'Unknown'
    return [1 if entry == x else 0 for entry in valid_entries]


def get_ecfp(smiles, radius=2, nBits=1024):
    """Generate ECFP molecular fingerprints from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to generate molecule from SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)


def get_atom_features(smiles):
    """Extract atom features from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to generate molecule from SMILES: {smiles}")

    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_symbol = one_of_k_encoding_unk(atom.GetSymbol(),
                                            ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown'])

        atom_degree = one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])

        implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])

        hybridization = one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ])

        is_aromatic = [atom.GetIsAromatic()]

        total_num_hs = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        atom_feature = (atom_symbol + atom_degree + implicit_valence +
                        hybridization + is_aromatic + total_num_hs)

        atom_features_list.append(atom_feature)

    atom_features_array = np.array(atom_features_list, dtype=np.float32)

    mean_features = np.mean(atom_features_array, axis=0)
    max_features = np.max(atom_features_array, axis=0)
    sum_features = np.sum(atom_features_array, axis=0)
    min_features = np.min(atom_features_array, axis=0)
    std_features = np.std(atom_features_array, axis=0)

    aggregated_features = np.concatenate([
        mean_features,
        max_features,
        sum_features,
        min_features,
        std_features
    ])

    return aggregated_features


def get_combined_features(smiles, ecfp_radius=2, ecfp_nBits=1024):
    """Get combined ECFP + atom features"""
    try:
        ecfp_features = get_ecfp(smiles, radius=ecfp_radius, nBits=ecfp_nBits)
        atom_features = get_atom_features(smiles)
        combined_features = np.concatenate([ecfp_features, atom_features])
        return combined_features

    except ValueError as e:
        raise ValueError(f"Unable to extract features from SMILES: {smiles}, error: {e}")


def process_data(csv_file, smiles_column='Smiles', target_column='pchembl', data_type="data"):
    """Process single data file and extract combined ECFP+atom features"""
    print(f"Reading {data_type} file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"{data_type} original sample count: {len(df)}")

    combined_features_list = []
    targets = []
    valid_indices = []
    invalid_count = 0

    print(f"Extracting {data_type} combined features (ECFP + atom features)...")
    for index, row in df.iterrows():
        smiles = str(row[smiles_column])
        try:
            combined_features = get_combined_features(smiles)
            combined_features_list.append(combined_features)
            targets.append(row[target_column])
            valid_indices.append(index)
        except ValueError as e:
            invalid_count += 1
            if invalid_count <= 5:
                print(f"Skipping invalid SMILES: {smiles}, error: {e}")

    if invalid_count > 5:
        print(f"... Total skipped {invalid_count} invalid SMILES")

    combined_df = pd.DataFrame(combined_features_list)

    ecfp_cols = [f'ECFP_{i}' for i in range(1024)]
    atom_feat_cols = []

    base_atom_features = ['AtomSymbol_' + str(i) for i in range(10)] + \
                         ['AtomDegree_' + str(i) for i in range(7)] + \
                         ['ImplicitValence_' + str(i) for i in range(7)] + \
                         ['Hybridization_' + str(i) for i in range(5)] + \
                         ['IsAromatic'] + \
                         ['TotalNumHs_' + str(i) for i in range(5)]

    for agg_type in ['Mean', 'Max', 'Sum', 'Min', 'Std']:
        for feat_name in base_atom_features:
            atom_feat_cols.append(f'{agg_type}_{feat_name}')

    all_feature_cols = ecfp_cols + atom_feat_cols
    combined_df.columns = all_feature_cols
    combined_df[target_column] = targets

    print(f"{data_type} combined feature extraction completed, valid samples: {len(combined_features_list)}")
    print(f"Feature count: ECFP({len(ecfp_cols)}) + atom features({len(atom_feat_cols)}) = total {len(all_feature_cols)} dimensions")
    print(f"{data_type} target variable range: {min(targets):.2f} ~ {max(targets):.2f}")
    print("-" * 60)

    return combined_df, valid_indices


def check_data_quality(df, name):
    """Check data quality"""
    print(f"\n🔍 Checking {name} data quality:")
    print(f"  - Data shape: {df.shape}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Data type check:")

    ecfp_cols = [col for col in df.columns if col.startswith('ECFP_')]
    print(f"    ECFP feature count: {len(ecfp_cols)}")

    atom_cols = [col for col in df.columns if
                 any(col.startswith(prefix) for prefix in ['Mean_', 'Max_', 'Sum_', 'Min_', 'Std_'])]
    print(f"    Atom feature count: {len(atom_cols)}")

    target_col = 'pchembl'
    print(f"    Target variable: {target_col}")

    for col in list(df.columns)[:5]:
        print(f"    {col}: {df[col].dtype}")

    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"  - Infinite values count: {inf_count}")

    target_stats = df[target_col].describe()
    print(f"  - Target variable statistics:\n{target_stats}")

    return df


def comprehensive_modeling():
    """Comprehensive machine learning modeling using combined features"""

    print("\n" + "=" * 80)
    print("🚀 Comprehensive Machine Learning Modeling - Using ECFP + Atom Feature Combination")
    print("=" * 80)

    from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                  ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor)
    from sklearn.linear_model import (Ridge, Lasso, ElasticNet, LinearRegression,
                                      Lars, LassoLars, OrthogonalMatchingPursuit,
                                      BayesianRidge, ARDRegression, HuberRegressor,
                                      PassiveAggressiveRegressor, TheilSenRegressor)
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import RANSACRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    import joblib

    try:
        import xgboost as xgb
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("⚠️ XGBoost not installed")

    try:
        import lightgbm as lgb
        has_lightgbm = True
    except ImportError:
        has_lightgbm = False
        print("⚠️ LightGBM not installed")

    try:
        import catboost as cb
        has_catboost = True
    except ImportError:
        has_catboost = False
        print("⚠️ CatBoost not installed")

    feature_cols = [col for col in combined_train_df.columns if col != 'pchembl']
    X_train = combined_train_df[feature_cols].values
    y_train = combined_train_df['pchembl'].values

    if has_test_set:
        X_test = test_combined_df[feature_cols].values
        y_test = test_combined_df['pchembl'].values
        test_name = "Independent test set"
    else:
        X_test = validation_combined_df[feature_cols].values
        y_test = validation_combined_df['pchembl'].values
        test_name = "Validation set"

    print(f"Training set size: {X_train.shape}")
    print(f"{test_name} size: {X_test.shape}")
    print(f"Feature dimensions: ECFP(1024) + atom features(175) = total {X_train.shape[1]} dimensions")

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=seed),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=seed),
        'Decision Tree': DecisionTreeRegressor(random_state=seed),

        'Ridge': Ridge(random_state=seed),
        'Lasso': Lasso(random_state=seed, max_iter=2000),
        'Elastic Net': ElasticNet(random_state=seed, max_iter=2000),
        'Linear Regression': LinearRegression(),
        'Lars': Lars(),
        'Lasso Lars': LassoLars(random_state=seed),
        'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
        'Bayesian Ridge': BayesianRidge(),
        'ARD Regression': ARDRegression(),

        'SVR': SVR(gamma='scale'),
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64), random_state=seed, max_iter=500),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Gaussian Process': GaussianProcessRegressor(random_state=seed),
        'Kernel Ridge': KernelRidge(),

        'Bagging': BaggingRegressor(random_state=seed, n_jobs=-1),

        'Huber': HuberRegressor(),
        'Passive Aggressive': PassiveAggressiveRegressor(random_state=seed),
        'RANSAC': RANSACRegressor(random_state=seed),
        'TheilSen': TheilSenRegressor(random_state=seed)
    }

    if has_xgboost:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=seed, n_jobs=-1)

    if has_lightgbm:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=seed, n_jobs=-1, verbose=-1)

    if has_catboost:
        models['CatBoost'] = cb.CatBoostRegressor(iterations=100, random_state=seed, verbose=False)

    print(f"\n🔄 Training and evaluating {len(models)} machine learning algorithms...")
    print("⏱️ This may take several minutes, please be patient...")

    results = {}
    predictions = {}
    feature_importances = {}

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"\n📊 [{i:2d}/{len(models)}] Training {name}...")
        start_time = time.time()

        try:
            needs_scaling = name in ['SVR', 'KNN', 'Ridge', 'Lasso', 'Elastic Net', 'Lars',
                                     'Lasso Lars', 'Orthogonal Matching Pursuit', 'Bayesian Ridge',
                                     'ARD Regression', 'MLP', 'Gaussian Process', 'Kernel Ridge',
                                     'Huber', 'Passive Aggressive']

            if needs_scaling:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                joblib.dump(scaler, f'{name.lower().replace(" ", "_")}_scaler.pkl')

                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                                cv=5, scoring='neg_mean_squared_error')
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    cv_std = np.sqrt(cv_scores.std())
                except:
                    cv_rmse = np.nan
                    cv_std = np.nan
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                try:
                    cv_scores = cross_val_score(model, X_train, y_train,
                                                cv=5, scoring='neg_mean_squared_error')
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    cv_std = np.sqrt(cv_scores.std())
                except:
                    cv_rmse = np.nan
                    cv_std = np.nan

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            pearson_corr, pearson_p = pearsonr(y_test, y_pred)

            end_time = time.time()
            training_time = end_time - start_time

            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAE': mae,
                'Pearson_r': pearson_corr,
                'Pearson_p': pearson_p,
                'CV_RMSE': cv_rmse,
                'CV_STD': cv_std,
                'Training_Time': training_time
            }

            predictions[name] = y_pred

            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_
            elif hasattr(model, 'coef_') and model.coef_ is not None:
                if len(model.coef_.shape) == 1:
                    feature_importances[name] = np.abs(model.coef_)
                else:
                    feature_importances[name] = np.abs(model.coef_[0])

            print(f"  ✅ MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            print(f"  📊 Pearson r: {pearson_corr:.4f} (p={pearson_p:.4e})")
            print(f"  📊 CV_RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")
            print(f"  ⏱️  Training time: {training_time:.2f}s")

            joblib.dump(model, f'{name.lower().replace(" ", "_")}_combined_model.pkl')

        except Exception as e:
            print(f"  ❌ {name} training failed: {e}")
            continue

    print("\n" + "=" * 110)
    print("🏆 All Model Performance Comparison (Using ECFP+Atom Feature Combination, Sorted by RMSE)")
    print("=" * 110)

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('RMSE')

    print(
        f"{'Model Name':<25} {'MSE':<10} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'Pearson_r':<12} {'CV_RMSE':<12} {'Time(s)':<10}")
    print("-" * 110)

    for idx, (model_name, row) in enumerate(results_df.iterrows()):
        rank_symbol = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{idx + 1:2d}."
        print(f"{rank_symbol} {model_name:<22} {row['MSE']:<10.4f} {row['RMSE']:<10.4f} {row['R²']:<10.4f} "
              f"{row['MAE']:<10.4f} {row['Pearson_r']:<12.4f} {row['CV_RMSE']:<12.4f} {row['Training_Time']:<10.2f}")

    best_model_name = results_df.index[0]
    best_results = results_df.iloc[0]

    print(f"\n🎉 Model ranking summary using combined features:")
    print(f"🥇 Best model: {best_model_name}")
    print(f"📊 Best performance: MSE={best_results['MSE']:.4f}, RMSE={best_results['RMSE']:.4f}, R²={best_results['R²']:.4f}")
    print(f"📊 Pearson correlation coefficient: r={best_results['Pearson_r']:.4f} (p={best_results['Pearson_p']:.4e})")

    print(f"\n🏆 Top 5 models:")
    for i, (model_name, row) in enumerate(results_df.head(5).iterrows()):
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        print(
            f"  {medals[i]} {model_name}: RMSE={row['RMSE']:.4f}, R²={row['R²']:.4f}, Pearson r={row['Pearson_r']:.4f}")

    results_df.to_csv('combined_features_model_results.csv')

    best_predictions = predictions[best_model_name]

    results_comparison = pd.DataFrame({
        'actual': y_test,
        'predicted': best_predictions,
        'residuals': y_test - best_predictions,
        'abs_residuals': np.abs(y_test - best_predictions)
    })

    results_comparison.to_csv(f'best_combined_model_predictions_{best_model_name.lower().replace(" ", "_")}.csv',
                              index=False)

    all_predictions_df = pd.DataFrame(predictions)
    all_predictions_df['actual'] = y_test
    all_predictions_df.to_csv('all_combined_models_predictions.csv', index=False)

    if feature_importances:
        print(f"\n🔍 Combined feature importance analysis (Top 15 most important features):")
        feature_names = feature_cols

        for model_name, importances in list(feature_importances.items())[:3]:
            print(f"\n📊 {model_name} feature importance:")
            top_indices = np.argsort(importances)[-15:]
            for i, idx in enumerate(reversed(top_indices)):
                feature_type = "ECFP" if feature_names[idx].startswith('ECFP_') else "Atom Feature"
                print(f"   {i + 1:2d}. {feature_names[idx]} ({feature_type}): {importances[idx]:.4f}")

    if feature_importances:
        importance_df = pd.DataFrame(feature_importances, index=feature_cols)
        importance_df.to_csv('combined_feature_importances.csv')

    print(f"\n💾 Result files saved:")
    print(f"  📊 combined_features_model_results.csv - All model performance comparison")
    print(f"  🥇 best_combined_model_predictions_{best_model_name.lower().replace(' ', '_')}.csv - Best model predictions")
    print(f"  📈 all_combined_models_predictions.csv - All model prediction results")
    if feature_importances:
        print(f"  🔍 combined_feature_importances.csv - Combined feature importance analysis")
    print(f"  🤖 {len(results)} combined feature model files (.pkl)")

    return results_df, best_model_name, results_comparison


try:
    print("=" * 60)
    print("🧬 Molecular Property Prediction - ECFP + Atom Feature Combined Modeling")
    print("=" * 60)

    train_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\train_data.csv'
    train_combined_df, train_valid_indices = process_data(train_csv_file, data_type="training set")

    validation_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\validation_data.csv'
    validation_combined_df, validation_valid_indices = process_data(validation_csv_file, data_type="validation set")

    test_csv_file = 'D:\\pycharm\\gutingle\\pythonProject2\\突出核蛋白\\test_data.csv'
    try:
        test_combined_df, test_valid_indices = process_data(test_csv_file, data_type="independent test set")
        has_test_set = True
    except FileNotFoundError:
        print("⚠️  Independent test set file not found")
        test_combined_df = None
        has_test_set = False

    train_combined_df = check_data_quality(train_combined_df, "training set")
    validation_combined_df = check_data_quality(validation_combined_df, "validation set")
    if has_test_set:
        test_combined_df = check_data_quality(test_combined_df, "independent test set")

    print(f"\nTraining set sample count: {len(train_combined_df)}")
    print(f"Validation set sample count: {len(validation_combined_df)}")
    if has_test_set:
        print(f"Independent test set sample count: {len(test_combined_df)}")

    combined_train_df = pd.concat([train_combined_df, validation_combined_df], ignore_index=True)
    print(f"Combined training data sample count: {len(combined_train_df)}")

    results_df, best_model_name, results_comparison = comprehensive_modeling()

    print(f"\n🔍 {best_model_name} detailed performance analysis:")
    print(f"  📊 Predicted value range: {results_comparison['predicted'].min():.3f} ~ {results_comparison['predicted'].max():.3f}")
    print(f"  🎯 Actual value range: {results_comparison['actual'].min():.3f} ~ {results_comparison['actual'].max():.3f}")
    print(f"  📈 Residual statistics:")
    print(f"    • Mean residual: {results_comparison['residuals'].mean():.4f}")
    print(f"    • Residual standard deviation: {results_comparison['residuals'].std():.4f}")
    print(f"    • Residual range: {results_comparison['residuals'].min():.3f} ~ {results_comparison['residuals'].max():.3f}")
    print(f"    • Mean absolute residual: {results_comparison['abs_residuals'].mean():.4f}")

    best_pearson_r, best_pearson_p = pearsonr(results_comparison['actual'], results_comparison['predicted'])
    print(f"  📊 Final Pearson correlation coefficient: r={best_pearson_r:.4f} (p={best_pearson_p:.4e})")

    print("\n" + "=" * 60)
    print("🎉 ECFP + Atom Feature Combined Modeling Complete!")
    print("=" * 60)
    print(f"📁 All result files saved to: {os.getcwd()}")
    print(f"🤖 Total trained {len(results_df)} machine learning models")
    print(f"🏆 Best model: {best_model_name}")
    print(f"🔬 Feature composition: ECFP(1024 dimensions) + atom features(175 dimensions) = total 1199 dimensions")
    print(
        f"📊 Best performance: RMSE={results_df.iloc[0]['RMSE']:.4f}, R²={results_df.iloc[0]['R²']:.4f}, Pearson r={results_df.iloc[0]['Pearson_r']:.4f}")

except Exception as e:
    print(f"\n❌ Program execution failed:")
    print(f"Error: {e}")
    traceback.print_exc()