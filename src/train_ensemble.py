"""
高精度アンサンブルモデル: LightGBM + XGBoost + CatBoost + TabNet
feature_onehot.ipynbの特徴量エンジニアリングを利用
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

# 再現性のためのシード設定
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def feature_engineering(df, is_train=True):
    """特徴量エンジニアリング関数"""
    df = df.copy()

    # 数値特徴量の変換
    df['age_group'] = pd.cut(df['age'], bins=16).astype(str)
    df['age_group'] = df['age_group'].str.replace(r'[(),.\[\] ]', '_', regex=True)

    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)

    # 時系列特徴量
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['campaign_efficiency'] = df['duration'] / (df['campaign'] + 1)
    df['duration_log'] = np.log1p(df['duration'])

    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    # 月のマッピングと周期性エンコーディング
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

    # ローン関連の特徴量
    df['total_loans'] = (df['housing'] == 'yes').astype(int) + (df['loan'] == 'yes').astype(int)
    df['has_any_loan'] = (df['total_loans'] > 0).astype(int)

    # カテゴリカル特徴量の準備
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'age_group']

    # 相互作用特徴量
    df['job_education'] = df['job'].astype(str) + '_' + df['education'].astype(str)
    df['contact_month'] = df['contact'].astype(str) + '_' + df['month'].astype(str)

    interaction_cols = ['job_education', 'contact_month']
    categorical_cols.extend(interaction_cols)

    df = df.drop(columns=['month', 'month_numeric'])

    return df, categorical_cols

def prepare_data():
    """データの読み込みと前処理"""
    print("データの読み込みと前処理...")

    train_df = pd.read_csv("/home/user/bank/data/train.csv")
    test_df = pd.read_csv("/home/user/bank/data/test.csv")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Positive rate: {train_df['y'].mean():.4f}\n")

    # 特徴量エンジニアリング
    train_processed, categorical_cols = feature_engineering(train_df, is_train=True)
    test_processed, _ = feature_engineering(test_df, is_train=False)

    # ワンホットエンコーディング
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

    # ターゲットと特徴量の分離
    y = train_encoded['y']
    X = train_encoded.drop(columns=['id', 'y'])
    X_test = test_encoded.drop(columns=['id'])

    print(f"特徴量数: {X.shape[1]}\n")

    return X, y, X_test, test_df['id']

def train_lightgbm(X, y, X_test, cv, n_trials=15):
    """LightGBMの最適化と学習"""
    print("=" * 60)
    print("1. LightGBMモデルの最適化と学習")
    print("=" * 60)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 3000,
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced"
        }

        cv_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_valid_fold, y_valid_fold)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )

            preds = model.predict_proba(X_valid_fold)[:, 1]
            cv_scores.append(roc_auc_score(y_valid_fold, preds))

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV AUC: {study.best_value:.5f}")

    # 最良パラメータで学習
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 3000,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced"
    })

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        oof[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits

    oof_auc = roc_auc_score(y, oof)
    print(f"LightGBM OOF AUC: {oof_auc:.5f}\n")

    return oof, test_pred, oof_auc

def train_xgboost(X, y, X_test, cv, n_trials=15):
    """XGBoostの最適化と学習"""
    print("=" * 60)
    print("2. XGBoostモデルの最適化と学習")
    print("=" * 60)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 3000,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            "verbosity": 0
        }

        cv_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_valid_fold, y_valid_fold)],
                verbose=False
            )

            preds = model.predict_proba(X_valid_fold)[:, 1]
            cv_scores.append(roc_auc_score(y_valid_fold, preds))

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV AUC: {study.best_value:.5f}")

    # 最良パラメータで学習
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 3000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "verbosity": 0
    })

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            verbose=False
        )

        oof[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits

    oof_auc = roc_auc_score(y, oof)
    print(f"XGBoost OOF AUC: {oof_auc:.5f}\n")

    return oof, test_pred, oof_auc

def train_catboost(X, y, X_test, cv, n_trials=15):
    """CatBoostの最適化と学習"""
    print("=" * 60)
    print("3. CatBoostモデルの最適化と学習")
    print("=" * 60)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "iterations": 3000,
            "eval_metric": "AUC",
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "early_stopping_rounds": 100,
            "auto_class_weights": "Balanced"
        }

        cv_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            train_pool = Pool(X_train_fold, y_train_fold)
            valid_pool = Pool(X_valid_fold, y_valid_fold)

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool)

            preds = model.predict_proba(X_valid_fold)[:, 1]
            cv_scores.append(roc_auc_score(y_valid_fold, preds))

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV AUC: {study.best_value:.5f}")

    # 最良パラメータで学習
    best_params = study.best_params.copy()
    best_params.update({
        "iterations": 3000,
        "eval_metric": "AUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "early_stopping_rounds": 100,
        "auto_class_weights": "Balanced"
    })

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        train_pool = Pool(X_train_fold, y_train_fold)
        valid_pool = Pool(X_valid_fold, y_valid_fold)

        model = CatBoostClassifier(**best_params)
        model.fit(train_pool, eval_set=valid_pool)

        oof[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits

    oof_auc = roc_auc_score(y, oof)
    print(f"CatBoost OOF AUC: {oof_auc:.5f}\n")

    return oof, test_pred, oof_auc

def train_tabnet(X, y, X_test, cv):
    """TabNetの学習"""
    print("=" * 60)
    print("4. TabNetモデルの学習")
    print("=" * 60)

    # データの標準化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Fold {fold + 1}/{cv.n_splits}")

        X_train_fold = X_scaled.iloc[train_idx].values
        X_valid_fold = X_scaled.iloc[valid_idx].values
        y_train_fold = y.iloc[train_idx].values.reshape(-1, 1)
        y_valid_fold = y.iloc[valid_idx].values.reshape(-1, 1)

        model = TabNetClassifier(
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            seed=RANDOM_STATE,
            verbose=0
        )

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            eval_metric=['auc'],
            max_epochs=200,
            patience=20,
            batch_size=256,
            virtual_batch_size=128
        )

        oof[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_pred += model.predict_proba(X_test_scaled.values)[:, 1] / cv.n_splits

    oof_auc = roc_auc_score(y, oof)
    print(f"TabNet OOF AUC: {oof_auc:.5f}\n")

    return oof, test_pred, oof_auc

def ensemble_models(y, oof_predictions, test_predictions):
    """アンサンブルモデル"""
    print("=" * 60)
    print("5. アンサンブル")
    print("=" * 60)

    oof_lgb, oof_xgb, oof_cat, oof_tabnet = oof_predictions
    test_lgb, test_xgb, test_cat, test_tabnet = test_predictions

    # 単純平均
    oof_avg = (oof_lgb + oof_xgb + oof_cat + oof_tabnet) / 4
    test_avg = (test_lgb + test_xgb + test_cat + test_tabnet) / 4
    avg_auc = roc_auc_score(y, oof_avg)

    print(f"単純平均アンサンブル OOF AUC: {avg_auc:.5f}")

    # スタッキング
    meta_features = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_tabnet])
    meta_test = np.column_stack([test_lgb, test_xgb, test_cat, test_tabnet])

    meta_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    meta_model.fit(meta_features, y)

    oof_stacking = meta_model.predict_proba(meta_features)[:, 1]
    test_stacking = meta_model.predict_proba(meta_test)[:, 1]
    stacking_auc = roc_auc_score(y, oof_stacking)

    print(f"スタッキングアンサンブル OOF AUC: {stacking_auc:.5f}")
    print(f"\nメタモデルの重み:")
    print(f"  LightGBM: {meta_model.coef_[0][0]:.4f}")
    print(f"  XGBoost:  {meta_model.coef_[0][1]:.4f}")
    print(f"  CatBoost: {meta_model.coef_[0][2]:.4f}")
    print(f"  TabNet:   {meta_model.coef_[0][3]:.4f}\n")

    # 最良モデルの選択
    if stacking_auc > avg_auc:
        return oof_stacking, test_stacking, stacking_auc, "Stacking"
    else:
        return oof_avg, test_avg, avg_auc, "Average"

def find_best_threshold(y, oof_pred):
    """最適閾値の探索"""
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.3, 0.8, 0.01):
        pred_binary = (oof_pred > threshold).astype(int)
        f1 = f1_score(y, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

def main():
    print("高精度アンサンブルモデルの学習を開始します\n")

    # データ準備
    X, y, X_test, test_ids = prepare_data()

    # 交差検証の設定
    N_SPLITS = 5
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 各モデルの学習
    oof_lgb, test_lgb, lgb_auc = train_lightgbm(X, y, X_test, cv, n_trials=15)
    oof_xgb, test_xgb, xgb_auc = train_xgboost(X, y, X_test, cv, n_trials=15)
    oof_cat, test_cat, cat_auc = train_catboost(X, y, X_test, cv, n_trials=15)
    oof_tabnet, test_tabnet, tabnet_auc = train_tabnet(X, y, X_test, cv)

    # 各モデルのスコア表示
    print("=" * 60)
    print("各モデルのOOF AUCスコア")
    print("=" * 60)
    print(f"LightGBM: {lgb_auc:.5f}")
    print(f"XGBoost:  {xgb_auc:.5f}")
    print(f"CatBoost: {cat_auc:.5f}")
    print(f"TabNet:   {tabnet_auc:.5f}")
    print("=" * 60 + "\n")

    # アンサンブル
    oof_predictions = [oof_lgb, oof_xgb, oof_cat, oof_tabnet]
    test_predictions = [test_lgb, test_xgb, test_cat, test_tabnet]

    best_oof, best_test, best_auc, best_model = ensemble_models(
        y, oof_predictions, test_predictions
    )

    # 最適閾値の探索
    print("=" * 60)
    print("最終評価")
    print("=" * 60)
    best_threshold, best_f1 = find_best_threshold(y, best_oof)
    print(f"最良モデル: {best_model}")
    print(f"最良OOF AUC: {best_auc:.5f}")
    print(f"最適閾値: {best_threshold:.3f}")
    print(f"最適F1スコア: {best_f1:.5f}")

    # 最適閾値での精度
    oof_binary = (best_oof > best_threshold).astype(int)
    accuracy = accuracy_score(y, oof_binary)
    print(f"Accuracy: {accuracy:.5f}\n")

    # 提出ファイル作成
    test_pred_binary = (best_test > best_threshold).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'y': test_pred_binary
    })

    submission.to_csv('/home/user/bank/data/deep_learning_ensemble_submission.csv',
                      index=False, header=False)

    print("提出ファイルを作成しました: deep_learning_ensemble_submission.csv")
    print(f"予測分布:")
    print(submission['y'].value_counts())
    print(f"Positive予測率: {submission['y'].mean():.4f}")

    # 確率値も保存
    submission_proba = pd.DataFrame({
        'id': test_ids,
        'y_proba': best_test,
        'y_pred': test_pred_binary
    })

    submission_proba.to_csv('/home/user/bank/data/deep_learning_ensemble_submission_with_proba.csv',
                           index=False)

    print("\n学習完了!")

if __name__ == "__main__":
    main()
