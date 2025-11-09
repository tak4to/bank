#!/usr/bin/env python3
"""
高精度LightGBMモデル学習スクリプト（目標: 99%精度）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)
import optuna
import lightgbm as lgb
import warnings
import pickle
warnings.filterwarnings('ignore')

# 再現性のためのシード設定
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("高精度LightGBMモデル学習開始")
print("=" * 60)

# ===== データ読み込み =====
print("\n[1/7] データ読み込み中...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTarget distribution:")
print(train_df['y'].value_counts())
print(f"Positive rate: {train_df['y'].mean():.4f}")


# ===== 特徴量エンジニアリング =====
def feature_engineering(df, is_train=True, encoders=None):
    """改良版特徴量エンジニアリング"""
    df = df.copy()

    # ===== 1. 数値特徴量の変換 =====

    # age関連
    df['age_group'] = pd.cut(df['age'],
                              bins=[0, 25, 35, 45, 55, 65, 100],
                              labels=['0-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    df['age_squared'] = df['age'] ** 2

    # balance関連
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)

    # ===== 2. DURATION関連特徴量（最重要） =====

    # 通話時間の対数変換
    df['duration_log'] = np.log1p(df['duration'])

    # 通話時間の2乗（非線形性を捉える）
    df['duration_squared'] = df['duration'] ** 2

    # 通話時間の平方根
    df['duration_sqrt'] = np.sqrt(df['duration'])

    # 通話時間のビン分割（カテゴリ化）
    df['duration_bin'] = pd.cut(df['duration'],
                                 bins=[-1, 100, 200, 300, 500, 1000, 10000],
                                 labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extremely_long'])

    # 1日あたりの通話時間
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)

    # キャンペーン効率（通話時間/キャンペーン回数）
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)

    # ===== 3. 月の特徴量 =====

    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)

    # 周期性エンコーディング
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

    # 四半期
    df['quarter'] = ((df['month_numeric'] - 1) // 3) + 1

    # ===== 4. pdays/previous関連特徴量 =====

    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['pdays_log'] = np.log1p(df['pdays'].replace(-1, 0))
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    # ===== 5. ローン関連特徴量 =====

    # yes/noを0/1に変換
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df['total_loans'] = df['housing'] + df['loan']
    df['has_any_loan'] = (df['total_loans'] > 0).astype(int)
    df['has_default'] = df['default']

    # ===== 6. 複合特徴量 =====

    # campaignとdurationの関係
    df['campaign_intensity'] = df['campaign'] * df['duration']

    # dayとdurationの関係
    df['day_duration_interaction'] = df['day'] * df['duration']

    # ===== 7. カテゴリカル変数のエンコーディング =====

    categorical_feats = ['job', 'marital', 'education', 'contact', 'poutcome',
                         'age_group', 'duration_bin']

    if is_train:
        target_encoders = {}
        label_encoders = {}
        freq_encoders = {}

        # Target Encoding
        for col in categorical_feats:
            if 'y' in df.columns:
                target_mean = df.groupby(col)['y'].mean()
                global_mean = df['y'].mean()
                counts = df.groupby(col).size()
                smoothing = 10
                smooth_target = (target_mean * counts + global_mean * smoothing) / (counts + smoothing)
                target_encoders[col] = smooth_target
                df[f'{col}_target_enc'] = df[col].map(smooth_target)

        # Frequency Encoding
        for col in categorical_feats:
            freq = df[col].value_counts(normalize=True)
            freq_encoders[col] = freq
            df[f'{col}_freq'] = df[col].map(freq)

        # Label Encoding
        for col in categorical_feats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        # monthとmonth_numericを削除（既に特徴量化済み）
        df = df.drop(columns=['month', 'month_numeric'])

        encoders = {
            'target_encoders': target_encoders,
            'label_encoders': label_encoders,
            'freq_encoders': freq_encoders
        }
        return df, encoders

    else:
        if encoders is None:
            raise ValueError("テストデータ処理時にはencodersを渡す必要があります")

        target_encoders = encoders['target_encoders']
        label_encoders = encoders['label_encoders']
        freq_encoders = encoders['freq_encoders']

        # Target Encoding
        for col in categorical_feats:
            if col in target_encoders:
                df[f'{col}_target_enc'] = df[col].map(target_encoders[col])
                df[f'{col}_target_enc'] = df[f'{col}_target_enc'].astype('float64')
                df[f'{col}_target_enc'].fillna(target_encoders[col].mean(), inplace=True)

        # Frequency Encoding
        for col in categorical_feats:
            if col in freq_encoders:
                df[f'{col}_freq'] = df[col].map(freq_encoders[col])
                df[f'{col}_freq'].fillna(freq_encoders[col].min(), inplace=True)

        # Label Encoding
        for col in categorical_feats:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # monthとmonth_numericを削除
        df = df.drop(columns=['month', 'month_numeric'])

        return df, None


print("\n[2/7] 特徴量エンジニアリング中...")
train_processed, encoders = feature_engineering(train_df, is_train=True)
print(f"Feature engineering completed! New train shape: {train_processed.shape}")


# ===== データ分割 =====
print("\n[3/7] データ分割中...")
y = train_processed['y']
exclude_cols = ['id', 'y']
X = train_processed.drop(columns=exclude_cols)

print(f"Total features: {X.shape[1]}")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train set: {X_train.shape}, y=1 rate: {y_train.mean():.4f}")
print(f"Valid set: {X_valid.shape}, y=1 rate: {y_valid.mean():.4f}")


# ===== ハイパーパラメータ最適化 =====
print("\n[4/7] ハイパーパラメータ最適化中...")

def objective_lgb(trial):
    """LightGBMのハイパーパラメータ最適化（accuracy最大化）"""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "n_estimators": 3000,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "boosting_type": "gbdt"
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )

    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

print(f"\nBest Accuracy: {study_lgb.best_value:.6f} ({study_lgb.best_value*100:.4f}%)")
print(f"Best params: {study_lgb.best_params}")


# ===== 最終モデルの学習 =====
print("\n[5/7] 最終モデル学習中...")

best_params_lgb = study_lgb.best_params
best_params_lgb.update({
    "n_estimators": 3000,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": 1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "boosting_type": "gbdt"
})

model_lgb = lgb.LGBMClassifier(**best_params_lgb)
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    callbacks=[lgb.early_stopping(150, verbose=True), lgb.log_evaluation(100)]
)

print("\nモデル学習完了！")


# ===== モデル評価 =====
print("\n[6/7] モデル評価中...")

y_pred_proba = model_lgb.predict_proba(X_valid)[:, 1]
y_pred = model_lgb.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
auc = roc_auc_score(y_valid, y_pred_proba)

print("\n" + "=" * 60)
print("検証データでの性能")
print("=" * 60)
print(f"Accuracy:  {accuracy:.6f} ({accuracy*100:.4f}%)")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")
print(f"AUC:       {auc:.6f}")
print("=" * 60)

print("\n混同行列:")
cm = confusion_matrix(y_valid, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_valid, y_pred))

# 特徴量重要度
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_lgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 重要な特徴量:")
print(feature_importance.head(20).to_string())


# モデル保存（テストデータ予測前に実行）
print("\n[7/8] モデルとエンコーダー保存中...")
with open('data/lgbm_model.pkl', 'wb') as f:
    pickle.dump(model_lgb, f)

with open('data/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("モデルとエンコーダーを保存しました")

# ===== テストデータへの予測 =====
print("\n[8/8] テストデータ予測中...")

try:
    test_processed, _ = feature_engineering(test_df, is_train=False, encoders=encoders)
    X_test = test_processed.drop(columns=['id'])

    test_pred = model_lgb.predict(X_test)

    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_df['id'],
        'y': test_pred
    })

    submission.to_csv('data/high_accuracy_submission.csv', index=False, header=False)
    print("提出ファイルを作成しました: data/high_accuracy_submission.csv")
    print(f"\n予測されたy=1の割合: {test_pred.mean():.4f}")
    print(f"予測されたy=1の数: {test_pred.sum()}")
    print(f"予測されたy=0の数: {(1-test_pred).sum()}")
except Exception as e:
    print(f"テストデータ予測中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
print("\n" + "=" * 60)
print("すべての処理が完了しました！")
print("=" * 60)
