"""
改善版LightGBM学習スクリプト
- ワンホットエンコーディング
- 交差検証（StratifiedKFold）
- ハイパーパラメータ最適化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import optuna
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 再現性のためのシード設定
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def feature_engineering(df, is_train=True):
    """
    特徴量エンジニアリング関数（ワンホットエンコーディング版）
    """
    df = df.copy()

    # ===== 1. 数値特徴量の変換 =====
    # 年齢グループ (スタージェンの公式に基づく最適ビン数: 16)
    # k = 1 + 3.322 * log10(n) = 1 + 3.322 * log10(27128) ≈ 16
    # 年齢範囲: 18-95歳を16グループに等幅分割
    df['age_group'] = pd.cut(df['age'], bins=16).astype(str)
    # LightGBMのエラー回避: 特殊文字を置換
    df['age_group'] = df['age_group'].str.replace(r'[(),.\[\] ]', '_', regex=True)

    # balance の対数変換
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)

    # ===== 2. 時系列特徴量 =====
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['campaign_efficiency'] = df['duration'] / (df['campaign'] + 1)
    df['duration_log'] = np.log1p(df['duration'])

    # previous関連
    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    # ===== 3. 月のマッピングと周期性エンコーディング =====
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

    # ===== 4. ローン関連の特徴量 =====
    df['total_loans'] = (df['housing'] == 'yes').astype(int) + (df['loan'] == 'yes').astype(int)
    df['has_any_loan'] = (df['total_loans'] > 0).astype(int)

    # ===== 5. バイナリ変数を数値化 =====
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # ワンホットエンコーディング対象のカテゴリカル変数
    categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'age_group']

    # ===== 6. 相互作用特徴量 =====
    df['job_education'] = df['job'].astype(str) + '_' + df['education'].astype(str)
    df['contact_month'] = df['contact'].astype(str) + '_' + df['month'].astype(str)

    interaction_cols = ['job_education', 'contact_month']
    categorical_cols.extend(interaction_cols)

    # monthは既に周期性エンコーディングしたので削除
    df = df.drop(columns=['month', 'month_numeric'])

    return df, categorical_cols


def objective_lgb_cv(trial, X, y):
    """
    LightGBMのハイパーパラメータ最適化（交差検証版）
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "n_estimators": 3000,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "boosting_type": "gbdt"
    }

    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
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
        auc = roc_auc_score(y_valid_fold, preds)
        cv_scores.append(auc)

    return np.mean(cv_scores)


def main():
    print("=" * 60)
    print("改善版LightGBM学習スクリプト")
    print("=" * 60)

    # データ読み込み
    print("\n[1/6] データ読み込み...")
    train_df = pd.read_csv("/home/user/bank/data/train.csv")
    test_df = pd.read_csv("/home/user/bank/data/test.csv")

    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"  Positive rate: {train_df['y'].mean():.4f}")

    # 特徴量エンジニアリング
    print("\n[2/6] 特徴量エンジニアリング...")
    train_processed, categorical_cols = feature_engineering(train_df, is_train=True)
    test_processed, _ = feature_engineering(test_df, is_train=False)
    print(f"  カテゴリカル変数数: {len(categorical_cols)}")

    # ワンホットエンコーディング
    print("\n[3/6] ワンホットエンコーディング...")
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

    print(f"  Train shape: {train_encoded.shape}")
    print(f"  Test shape: {test_encoded.shape}")
    print(f"  総特徴量数: {train_encoded.shape[1] - 2}")  # id, yを除く

    # ターゲットと特徴量の分離
    y = train_encoded['y']
    X = train_encoded.drop(columns=['id', 'y'])
    X_test = test_encoded.drop(columns=['id'])

    # ハイパーパラメータ最適化
    print("\n[4/6] ハイパーパラメータ最適化（交差検証）...")
    print("  これには時間がかかります（約10-20分）...")

    study = optuna.create_study(direction="maximize", study_name="lgbm_cv")
    study.optimize(lambda trial: objective_lgb_cv(trial, X, y),
                   n_trials=50,
                   show_progress_bar=True)

    print(f"\n  Best CV AUC: {study.best_value:.5f}")
    print(f"  Best params: {study.best_params}")

    # 最適パラメータで交差検証学習
    print("\n[5/6] 交差検証で最終モデル学習...")
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 3000,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "boosting_type": "gbdt"
    })

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    cv_scores = []
    models = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"  Fold {fold + 1}/5...", end=" ")

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

        oof_predictions[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        test_predictions += model.predict_proba(X_test)[:, 1] / 5

        fold_auc = roc_auc_score(y_valid_fold, oof_predictions[valid_idx])
        cv_scores.append(fold_auc)
        models.append(model)

        print(f"AUC: {fold_auc:.5f}")

    overall_auc = roc_auc_score(y, oof_predictions)
    print(f"\n  Overall OOF AUC: {overall_auc:.5f}")
    print(f"  Mean CV AUC: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")

    # 最適閾値探索
    print("\n  最適閾値を探索中...")
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.3, 0.8, 0.01):
        pred_binary = (oof_predictions > threshold).astype(int)
        f1 = f1_score(y, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"  最適閾値: {best_threshold:.3f}")
    print(f"  最適F1スコア: {best_f1:.5f}")

    # テストデータ予測
    print("\n[6/6] テストデータ予測と提出ファイル作成...")
    test_pred_binary = (test_predictions > best_threshold).astype(int)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'y': test_pred_binary
    })

    output_path = '/home/user/bank/data/improved_onehot_cv_submission.csv'
    submission.to_csv(output_path, index=False, header=False)

    print(f"  提出ファイル作成完了: {output_path}")
    print(f"  Positive予測率: {submission['y'].mean():.4f}")

    # 確率値も保存
    submission_proba = pd.DataFrame({
        'id': test_df['id'],
        'y_proba': test_predictions,
        'y_pred': test_pred_binary
    })

    proba_path = '/home/user/bank/data/improved_onehot_cv_submission_with_proba.csv'
    submission_proba.to_csv(proba_path, index=False)
    print(f"  確率値付きファイル作成完了: {proba_path}")

    print("\n" + "=" * 60)
    print("学習完了!")
    print("=" * 60)

    # 特徴量重要度のTop 20を表示
    print("\nTop 20 重要な特徴量:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': models[-1].feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:8.2f}")


if __name__ == "__main__":
    main()
