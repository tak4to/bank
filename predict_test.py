#!/usr/bin/env python3
"""
テストデータの予測スクリプト
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("テストデータの予測を開始...")

# データ読み込み
test_df = pd.read_csv("data/test.csv")
print(f"Test shape: {test_df.shape}")

# エンコーダー読み込みの試行
try:
    with open('data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    print("エンコーダーを読み込みました")
except FileNotFoundError:
    print("エンコーダーファイルが見つかりません。モデルを再学習します。")
    # エンコーダーがない場合は終了
    exit(1)

# モデル読み込み
try:
    with open('data/lgbm_model.pkl', 'rb') as f:
        model_lgb = pickle.load(f)
    print("モデルを読み込みました")
except FileNotFoundError:
    print("モデルファイルが見つかりません。モデルを再学習してください。")
    exit(1)


# 特徴量エンジニアリング関数
def feature_engineering(df, is_train=True, encoders=None):
    """改良版特徴量エンジニアリング"""
    df = df.copy()

    # ===== 1. 数値特徴量の変換 =====
    df['age_group'] = pd.cut(df['age'],
                              bins=[0, 25, 35, 45, 55, 65, 100],
                              labels=['0-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    df['age_squared'] = df['age'] ** 2

    # balance関連
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)

    # ===== 2. DURATION関連特徴量 =====
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_squared'] = df['duration'] ** 2
    df['duration_sqrt'] = np.sqrt(df['duration'])
    df['duration_bin'] = pd.cut(df['duration'],
                                 bins=[-1, 100, 200, 300, 500, 1000, 10000],
                                 labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extremely_long'])
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)

    # ===== 3. 月の特徴量 =====
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)
    df['quarter'] = ((df['month_numeric'] - 1) // 3) + 1

    # ===== 4. pdays/previous関連特徴量 =====
    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['pdays_log'] = np.log1p(df['pdays'].replace(-1, 0))
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    # ===== 5. ローン関連特徴量 =====
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    df['total_loans'] = df['housing'] + df['loan']
    df['has_any_loan'] = (df['total_loans'] > 0).astype(int)
    df['has_default'] = df['default']

    # ===== 6. 複合特徴量 =====
    df['campaign_intensity'] = df['campaign'] * df['duration']
    df['day_duration_interaction'] = df['day'] * df['duration']

    # ===== 7. カテゴリカル変数のエンコーディング =====
    categorical_feats = ['job', 'marital', 'education', 'contact', 'poutcome',
                         'age_group', 'duration_bin']

    if not is_train:
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

    return df


# テストデータの特徴量エンジニアリング
print("特徴量エンジニアリング中...")
test_processed = feature_engineering(test_df, is_train=False, encoders=encoders)

# 特徴量の抽出
X_test = test_processed.drop(columns=['id'])

# すべての列をfloat型に変換（LightGBMのカテゴリカル特徴量エラーを回避）
for col in X_test.columns:
    if X_test[col].dtype.name == 'category':
        X_test[col] = X_test[col].cat.codes
    X_test[col] = X_test[col].astype('float64')

print(f"Test features shape: {X_test.shape}")
print(f"Data types: {X_test.dtypes.value_counts()}")

# 予測（numpy配列に変換してカテゴリカル特徴量の不一致を回避）
print("予測中...")
X_test_array = X_test.values  # numpy配列に変換
test_pred = model_lgb.predict(X_test_array)

# 提出ファイル作成
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_pred
})

submission.to_csv('data/high_accuracy_submission.csv', index=False, header=False)
print("\n提出ファイルを作成しました: data/high_accuracy_submission.csv")
print(f"予測されたy=1の割合: {test_pred.mean():.4f}")
print(f"予測されたy=1の数: {test_pred.sum()}")
print(f"予測されたy=0の数: {(1-test_pred).sum()}")

print("\n完了！")
