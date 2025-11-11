"""
簡略版アンサンブルモデル: LightGBM + XGBoost
feature_onehot.ipynbの特徴量エンジニアリングを利用
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def feature_engineering(df, is_train=True):
    """特徴量エンジニアリング関数"""
    df = df.copy()

    df['age_group'] = pd.cut(df['age'], bins=16).astype(str)
    df['age_group'] = df['age_group'].str.replace(r'[(),.\[\] ]', '_', regex=True)

    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)

    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['campaign_efficiency'] = df['duration'] / (df['campaign'] + 1)
    df['duration_log'] = np.log1p(df['duration'])

    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

    df['total_loans'] = (df['housing'] == 'yes').astype(int) + (df['loan'] == 'yes').astype(int)
    df['has_any_loan'] = (df['total_loans'] > 0).astype(int)

    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'age_group']

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

    train_processed, categorical_cols = feature_engineering(train_df, is_train=True)
    test_processed, _ = feature_engineering(test_df, is_train=False)

    train_encoded = pd.get_dummies(train_processed, columns=categorical_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_processed, columns=categorical_cols, drop_first=True)

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

    return X, y, X_test, test_df['id']

def main():
    print("簡略版アンサンブルモデルの学習を開始します\n")

    X, y, X_test, test_ids = prepare_data()

    N_SPLITS = 5
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # LightGBM (最適パラメータを使用)
    print("=" * 60)
    print("LightGBMで学習中...")
    print("=" * 60)

    lgb_params = {
        "learning_rate": 0.0604359689285941,
        "num_leaves": 25,
        "max_depth": 10,
        "min_child_samples": 25,
        "subsample": 0.9192186345663692,
        "colsample_bytree": 0.6064071248556598,
        "reg_alpha": 6.548964434164017e-05,
        "reg_lambda": 1.8857546583877704e-08,
        "n_estimators": 3000,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced"
    }

    oof_lgb = np.zeros(len(X))
    test_pred_lgb = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Fold {fold + 1}/{N_SPLITS}")

        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)]
        )

        oof_lgb[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_pred_lgb += model.predict_proba(X_test)[:, 1] / N_SPLITS

    lgb_auc = roc_auc_score(y, oof_lgb)
    print(f"LightGBM OOF AUC: {lgb_auc:.5f}\n")

    # XGBoost (最適パラメータを使用)
    print("=" * 60)
    print("XGBoostで学習中...")
    print("=" * 60)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    xgb_params = {
        "learning_rate": 0.011956673284693823,
        "max_depth": 8,
        "min_child_weight": 4,
        "subsample": 0.9963988617094578,
        "colsample_bytree": 0.8069413619088408,
        "gamma": 0.9997189222676354,
        "reg_alpha": 7.747482126185368,
        "reg_lambda": 0.00010114471973055898,
        "n_estimators": 3000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "verbosity": 0
    }

    oof_xgb = np.zeros(len(X))
    test_pred_xgb = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Fold {fold + 1}/{N_SPLITS}")

        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            verbose=False
        )

        oof_xgb[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_pred_xgb += model.predict_proba(X_test)[:, 1] / N_SPLITS

    xgb_auc = roc_auc_score(y, oof_xgb)
    print(f"XGBoost OOF AUC: {xgb_auc:.5f}\n")

    # アンサンブル
    print("=" * 60)
    print("アンサンブル")
    print("=" * 60)

    # 単純平均
    oof_avg = (oof_lgb + oof_xgb) / 2
    test_pred_avg = (test_pred_lgb + test_pred_xgb) / 2
    avg_auc = roc_auc_score(y, oof_avg)
    print(f"単純平均アンサンブル OOF AUC: {avg_auc:.5f}")

    # スタッキング
    meta_features = np.column_stack([oof_lgb, oof_xgb])
    meta_test = np.column_stack([test_pred_lgb, test_pred_xgb])

    meta_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    meta_model.fit(meta_features, y)

    oof_stacking = meta_model.predict_proba(meta_features)[:, 1]
    test_pred_stacking = meta_model.predict_proba(meta_test)[:, 1]
    stacking_auc = roc_auc_score(y, oof_stacking)

    print(f"スタッキングアンサンブル OOF AUC: {stacking_auc:.5f}")
    print(f"\nメタモデルの重み:")
    print(f"  LightGBM: {meta_model.coef_[0][0]:.4f}")
    print(f"  XGBoost:  {meta_model.coef_[0][1]:.4f}\n")

    # 最良モデルの選択
    if stacking_auc > avg_auc:
        best_oof, best_test = oof_stacking, test_pred_stacking
        best_auc, best_name = stacking_auc, "Stacking"
    else:
        best_oof, best_test = oof_avg, test_pred_avg
        best_auc, best_name = avg_auc, "Average"

    print(f"最良モデル: {best_name}")
    print(f"最良OOF AUC: {best_auc:.5f}\n")

    # 最適閾値の探索
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.3, 0.8, 0.01):
        pred_binary = (best_oof > threshold).astype(int)
        f1 = f1_score(y, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"最適閾値: {best_threshold:.3f}")
    print(f"最適F1スコア: {best_f1:.5f}")

    oof_binary = (best_oof > best_threshold).astype(int)
    accuracy = accuracy_score(y, oof_binary)
    print(f"Accuracy: {accuracy:.5f}\n")

    # 提出ファイル作成
    test_pred_binary = (best_test > best_threshold).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'y': test_pred_binary
    })

    submission.to_csv('/home/user/bank/data/ensemble_submission.csv', index=False, header=False)

    print("提出ファイルを作成しました: ensemble_submission.csv")
    print(f"予測分布:")
    print(submission['y'].value_counts())
    print(f"Positive予測率: {submission['y'].mean():.4f}\n")

    print("学習完了!")

if __name__ == "__main__":
    main()
