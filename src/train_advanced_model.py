"""
高精度モデル（AUC 0.99を目指す）
- 高度な特徴量エンジニアリング（Target Encoding、統計量集約）
- Deep Learning（TabNet、カスタムNN）
- 長時間ハイパーパラメータ最適化（50-100 trials）
- Weighted Blending
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import optuna
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class DeepNN(nn.Module):
    """Deep Neural Network with advanced architecture"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

def advanced_feature_engineering(df, is_train=True, target_encodings=None):
    """高度な特徴量エンジニアリング"""
    df = df.copy()

    # 既存の基本的な特徴量
    df['age_group'] = pd.cut(df['age'], bins=16).astype(str)
    df['age_group'] = df['age_group'].str.replace(r'[(),.\[\] ]', '_', regex=True)

    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)
    df['balance_squared'] = df['balance'] ** 2
    df['balance_sqrt'] = np.sqrt(df['balance'] - df['balance'].min() + 1)

    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['campaign_efficiency'] = df['duration'] / (df['campaign'] + 1)
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_squared'] = df['duration'] ** 2
    df['duration_sqrt'] = np.sqrt(df['duration'])

    # 新しい統計量特徴量
    df['age_balance_ratio'] = df['age'] / (df['balance'].abs() + 1)
    df['duration_campaign_product'] = df['duration'] * df['campaign']
    df['pdays_previous_ratio'] = df['pdays'] / (df['previous'] + 1)

    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    # 月の周期的エンコーディング
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

    # 曜日の周期的エンコーディング（dayを曜日と仮定）
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['total_loans'] = (df['housing'] == 'yes').astype(int) + (df['loan'] == 'yes').astype(int)
    df['has_any_loan'] = (df['total_loans'] > 0).astype(int)

    # バイナリ変換
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # 交互作用特徴量
    df['job_education'] = df['job'].astype(str) + '_' + df['education'].astype(str)
    df['contact_month'] = df['contact'].astype(str) + '_' + df['month'].astype(str)
    df['marital_education'] = df['marital'].astype(str) + '_' + df['education'].astype(str)
    df['job_marital'] = df['job'].astype(str) + '_' + df['marital'].astype(str)

    # Target Encoding（訓練データでのみ計算）
    categorical_for_target_encoding = ['job', 'marital', 'education', 'contact', 'poutcome']

    if is_train and target_encodings is None:
        target_encodings = {}

    if target_encodings is not None:
        for col in categorical_for_target_encoding:
            if is_train:
                # 訓練データ: 各カテゴリの目的変数の平均を計算（CVで適切に処理する必要あり）
                # ここでは後でCV内で計算するため、プレースホルダー
                pass
            else:
                # テストデータ: 訓練データで計算したエンコーディングを使用
                if col + '_target_encoded' in target_encodings:
                    df[col + '_target_mean'] = df[col].map(target_encodings[col + '_target_encoded'])
                    df[col + '_target_mean'] = df[col + '_target_mean'].fillna(target_encodings[col + '_global_mean'])

    categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'age_group',
                        'job_education', 'contact_month', 'marital_education', 'job_marital']

    df = df.drop(columns=['month', 'month_numeric'])

    return df, categorical_cols, target_encodings

def prepare_data_advanced():
    """データの読み込みと高度な前処理"""
    print("データの読み込みと高度な前処理...")

    train_df = pd.read_csv("/home/user/bank/data/train.csv")
    test_df = pd.read_csv("/home/user/bank/data/test.csv")

    train_processed, categorical_cols, _ = advanced_feature_engineering(train_df, is_train=True)
    test_processed, _, _ = advanced_feature_engineering(test_df, is_train=False)

    train_encoded = pd.get_dummies(train_processed, columns=categorical_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_processed, columns=categorical_cols, drop_first=True)

    # カラムを揃える
    missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
    for col in missing_cols:
        if col != 'y':
            test_encoded[col] = 0

    extra_cols = set(test_encoded.columns) - set(train_encoded.columns)
    test_encoded = test_encoded.drop(columns=list(extra_cols))
    test_encoded = test_encoded[train_encoded.drop(columns=['y']).columns]

    y = train_encoded['y']
    X = train_encoded.drop(columns=['id', 'y'])
    X_test = test_encoded.drop(columns=['id'])

    # Target Encoding（CV内で正しく実装）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    categorical_for_target_encoding = ['job', 'marital', 'education', 'contact', 'poutcome']
    target_encoded_features = pd.DataFrame(index=X.index)

    for col in categorical_for_target_encoding:
        if col in train_processed.columns:
            target_encoded_col = np.zeros(len(X))

            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                train_fold_df = train_processed.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]

                # 各カテゴリの目的変数の平均を計算
                target_mean = train_fold_df.groupby(col)[col].count()
                target_mean = train_fold_df[[col]].copy()
                target_mean['target'] = y_train_fold.values
                target_mean = target_mean.groupby(col)['target'].mean()

                global_mean = y_train_fold.mean()

                # validationデータに適用
                valid_fold_df = train_processed.iloc[valid_idx]
                target_encoded_col[valid_idx] = valid_fold_df[col].map(target_mean).fillna(global_mean)

            target_encoded_features[col + '_target_mean'] = target_encoded_col

    # テストデータ用のTarget Encoding
    test_target_encoded = pd.DataFrame(index=X_test.index)
    for col in categorical_for_target_encoding:
        if col in train_processed.columns:
            target_mean = train_processed[[col]].copy()
            target_mean['target'] = y.values
            target_mean = target_mean.groupby(col)['target'].mean()
            global_mean = y.mean()

            test_target_encoded[col + '_target_mean'] = test_processed[col].map(target_mean).fillna(global_mean)

    # 結合
    X = pd.concat([X, target_encoded_features], axis=1)
    X_test = pd.concat([X_test, test_target_encoded], axis=1)

    print(f"特徴量数: {X.shape[1]}")

    return X, y, X_test, test_df['id']

def train_lightgbm_optimized(X, y, X_test, cv, n_trials=50):
    """LightGBMの最適化（試行回数を増やす）"""
    print("\n" + "=" * 80)
    print("LightGBM最適化（50 trials）")
    print("=" * 80)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "n_estimators": 5000,
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced"
        }

        cv_scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_valid_fold, y_valid_fold)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)]
            )

            preds = model.predict_proba(X_valid_fold)[:, 1]
            score = roc_auc_score(y_valid_fold, preds)
            cv_scores.append(score)

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize", study_name="lightgbm")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    # 最適パラメータで全データ学習
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 5000,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced"
    })

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Training final model Fold {fold + 1}/5...")
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)]
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"Final LightGBM OOF AUC: {oof_auc:.5f}")

    return oof_preds, test_preds, oof_auc

def train_xgboost_optimized(X, y, X_test, cv, n_trials=50):
    """XGBoostの最適化（試行回数を増やす）"""
    print("\n" + "=" * 80)
    print("XGBoost最適化（50 trials）")
    print("=" * 80)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 5000,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            "verbosity": 0
        }

        cv_scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_valid_fold, y_valid_fold)],
                verbose=False
            )

            preds = model.predict_proba(X_valid_fold)[:, 1]
            score = roc_auc_score(y_valid_fold, preds)
            cv_scores.append(score)

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize", study_name="xgboost")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    # 最適パラメータで全データ学習
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 5000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "verbosity": 0
    })

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Training final model Fold {fold + 1}/5...")
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            verbose=False
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"Final XGBoost OOF AUC: {oof_auc:.5f}")

    return oof_preds, test_preds, oof_auc

def train_tabnet(X, y, X_test, cv):
    """TabNetの学習"""
    print("\n" + "=" * 80)
    print("TabNet学習")
    print("=" * 80)

    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Training Fold {fold + 1}/5...")

        X_train_fold = X_scaled[train_idx]
        X_valid_fold = X_scaled[valid_idx]
        y_train_fold = y.iloc[train_idx].values.reshape(-1, 1)
        y_valid_fold = y.iloc[valid_idx].values.reshape(-1, 1)

        model = TabNetClassifier(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type="entmax",
            verbose=0,
            seed=RANDOM_STATE
        )

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            eval_metric=['auc']
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_preds += model.predict_proba(X_test_scaled)[:, 1] / 5

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"TabNet OOF AUC: {oof_auc:.5f}")

    return oof_preds, test_preds, oof_auc

def train_deep_nn(X, y, X_test, cv):
    """Deep Neural Networkの学習"""
    print("\n" + "=" * 80)
    print("Deep Neural Network学習")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Training Fold {fold + 1}/5...")

        X_train_fold = torch.FloatTensor(X_scaled[train_idx]).to(device)
        X_valid_fold = torch.FloatTensor(X_scaled[valid_idx]).to(device)
        y_train_fold = torch.FloatTensor(y.iloc[train_idx].values).to(device)
        y_valid_fold = torch.FloatTensor(y.iloc[valid_idx].values).to(device)

        # モデル
        model = DeepNN(input_dim=X.shape[1], hidden_dims=[512, 256, 128, 64], dropout=0.3).to(device)

        # 損失関数とオプティマイザー
        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

        # DataLoader
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        # 訓練
        best_val_auc = 0
        patience_counter = 0
        max_patience = 15

        for epoch in range(200):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = torch.sigmoid(model(X_valid_fold))
                val_auc = roc_auc_score(y_valid_fold.cpu().numpy(), val_outputs.cpu().numpy())

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                break

        # 最良モデルで予測
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            oof_preds[valid_idx] = torch.sigmoid(model(X_valid_fold)).cpu().numpy()
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            test_preds += torch.sigmoid(model(X_test_tensor)).cpu().numpy() / 5

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"Deep NN OOF AUC: {oof_auc:.5f}")

    return oof_preds, test_preds, oof_auc

def optimize_blend_weights(oof_preds_list, y):
    """アンサンブルの重みを最適化"""
    from scipy.optimize import minimize

    def objective(weights):
        weights = weights / weights.sum()
        blend = np.zeros(len(y))
        for i, oof in enumerate(oof_preds_list):
            blend += weights[i] * oof
        return -roc_auc_score(y, blend)

    initial_weights = np.ones(len(oof_preds_list)) / len(oof_preds_list)
    bounds = [(0, 1) for _ in range(len(oof_preds_list))]
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x / result.x.sum()

def main():
    print("=" * 80)
    print("高精度モデル訓練開始（AUC 0.99を目指す）")
    print("=" * 80)

    X, y, X_test, test_ids = prepare_data_advanced()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 各モデルの学習
    lgb_oof, lgb_test, lgb_auc = train_lightgbm_optimized(X, y, X_test, cv, n_trials=50)
    xgb_oof, xgb_test, xgb_auc = train_xgboost_optimized(X, y, X_test, cv, n_trials=50)
    tabnet_oof, tabnet_test, tabnet_auc = train_tabnet(X, y, X_test, cv)
    nn_oof, nn_test, nn_auc = train_deep_nn(X, y, X_test, cv)

    # アンサンブル
    print("\n" + "=" * 80)
    print("アンサンブル最適化")
    print("=" * 80)

    oof_preds_list = [lgb_oof, xgb_oof, tabnet_oof, nn_oof]
    test_preds_list = [lgb_test, xgb_test, tabnet_test, nn_test]
    model_names = ['LightGBM', 'XGBoost', 'TabNet', 'Deep NN']

    # 重みの最適化
    optimal_weights = optimize_blend_weights(oof_preds_list, y)

    print("\n最適な重み:")
    for name, weight in zip(model_names, optimal_weights):
        print(f"  {name}: {weight:.4f}")

    # 最適な重みでアンサンブル
    oof_blend = np.zeros(len(y))
    test_blend = np.zeros(len(X_test))

    for i, (oof, test, weight) in enumerate(zip(oof_preds_list, test_preds_list, optimal_weights)):
        oof_blend += weight * oof
        test_blend += weight * test

    blend_auc = roc_auc_score(y, oof_blend)
    print(f"\nWeighted Blend OOF AUC: {blend_auc:.5f}")

    # 最適閾値
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (oof_blend >= threshold).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n最適閾値: {best_threshold:.3f}")
    print(f"最適F1スコア: {best_f1:.5f}")

    # 提出ファイル作成
    test_preds_binary = (test_blend >= best_threshold).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'y': test_preds_binary
    })

    submission.to_csv("/home/user/bank/data/advanced_submission.csv", index=False)

    print(f"\n提出ファイルを作成しました: advanced_submission.csv")
    print(f"予測分布:\n{submission['y'].value_counts()}")
    print(f"Positive予測率: {submission['y'].mean():.4f}")

    print("\n学習完了!")
    print("=" * 80)
    print(f"最終 OOF AUC: {blend_auc:.5f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
