#!/usr/bin/env python3
"""
99%を目指すスタッキング + 超最適化モデル
- スタッキング（メタ学習）
- 徹底的なハイパーパラメータ最適化
- 閾値の最適化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import warnings
import pickle
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("99%を目指すスタッキング + 超最適化モデル")
print("=" * 80)

# データ読み込み
print("\n[1/7] データ読み込み...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 特徴量エンジニアリング関数定義
def advanced_feature_engineering(df, is_train=True, encoders=None):
    df = df.copy()
    high_conversion_months = ['mar', 'dec', 'sep', 'oct']
    df['is_high_conversion_month'] = df['month'].isin(high_conversion_months).astype(int)
    df['month_category'] = df['month'].map({
        'mar': 'very_high', 'dec': 'very_high', 'sep': 'very_high', 'oct': 'very_high',
        'apr': 'medium', 'feb': 'medium',
        'aug': 'low', 'jun': 'low', 'nov': 'low', 'jan': 'low', 'jul': 'low', 'may': 'low'
    })
    df['duration_very_long'] = (df['duration'] > 1000).astype(int)
    df['duration_long'] = ((df['duration'] > 500) & (df['duration'] <= 1000)).astype(int)
    df['duration_medium'] = ((df['duration'] > 200) & (df['duration'] <= 500)).astype(int)
    df['duration_short'] = (df['duration'] <= 200).astype(int)
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_squared'] = df['duration'] ** 2
    df['duration_sqrt'] = np.sqrt(df['duration'])
    df['poutcome_success'] = (df['poutcome'] == 'success').astype(int)
    df['poutcome_failure'] = (df['poutcome'] == 'failure').astype(int)
    df['poutcome_other'] = (df['poutcome'] == 'other').astype(int)
    df['poutcome_unknown'] = (df['poutcome'] == 'unknown').astype(int)
    df['contact_cellular'] = (df['contact'] == 'cellular').astype(int)
    df['contact_unknown'] = (df['contact'] == 'unknown').astype(int)
    df['job_high_conversion'] = df['job'].isin(['student', 'retired']).astype(int)
    df['job_low_conversion'] = df['job'].isin(['blue-collar', 'entrepreneur', 'housemaid']).astype(int)
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    df['no_housing'] = (df['housing'] == 0).astype(int)
    df['no_loan'] = (df['loan'] == 0).astype(int)
    df['total_loans'] = df['housing'] + df['loan']
    df['no_loans_at_all'] = ((df['housing'] == 0) & (df['loan'] == 0)).astype(int)
    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['pdays_log'] = np.log1p(df['pdays'].replace(-1, 0))
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)
    df['age_squared'] = df['age'] ** 2
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_high'] = (df['balance'] > 1500).astype(int)
    df['campaign_low'] = (df['campaign'] <= 2).astype(int)
    df['campaign_high'] = (df['campaign'] > 5).astype(int)
    df['duration_long_cellular'] = ((df['duration'] > 500) & (df['contact'] == 'cellular')).astype(int)
    df['success_with_previous'] = ((df['poutcome'] == 'success') & (df['pdays'] != -1)).astype(int)
    df['high_month_long_duration'] = (df['is_high_conversion_month'] & (df['duration'] > 300)).astype(int)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['marital_single'] = (df['marital'] == 'single').astype(int)
    df['education_tertiary'] = (df['education'] == 'tertiary').astype(int)
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)
    df['quarter'] = ((df['month_numeric'] - 1) // 3) + 1
    categorical_feats = ['job', 'marital', 'education', 'contact', 'poutcome', 'month_category']
    if is_train:
        target_encoders = {}
        label_encoders = {}
        for col in categorical_feats:
            if 'y' in df.columns:
                target_mean = df.groupby(col)['y'].mean()
                global_mean = df['y'].mean()
                counts = df.groupby(col).size()
                smoothing = 10
                smooth_target = (target_mean * counts + global_mean * smoothing) / (counts + smoothing)
                target_encoders[col] = smooth_target
                df[f'{col}_target_enc'] = df[col].map(smooth_target)
        for col in categorical_feats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        df = df.drop(columns=['month', 'month_numeric'])
        encoders = {'target_encoders': target_encoders, 'label_encoders': label_encoders}
        return df, encoders
    else:
        if encoders is None:
            raise ValueError("テストデータ処理時にはencodersを渡す必要があります")
        target_encoders = encoders['target_encoders']
        label_encoders = encoders['label_encoders']
        for col in categorical_feats:
            if col in target_encoders:
                df[f'{col}_target_enc'] = df[col].map(target_encoders[col])
                df[f'{col}_target_enc'] = df[f'{col}_target_enc'].astype('float64')
                df[f'{col}_target_enc'].fillna(target_encoders[col].mean(), inplace=True)
        for col in categorical_feats:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        df = df.drop(columns=['month', 'month_numeric'])
        return df, None

print("\n[2/7] 特徴量エンジニアリング...")
train_processed, encoders = advanced_feature_engineering(train_df, is_train=True)
print(f"Shape: {train_processed.shape}")

# データ分割
print("\n[3/7] データ分割...")
y = train_processed['y']
X = train_processed.drop(columns=['id', 'y'])
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")


# ===== Optuna でベースモデルを最適化 =====
print("\n[4/7] Optunaで最適化...")

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return accuracy_score(y_valid, pred)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=30, show_progress_bar=True)
print(f"\nBest RF Accuracy: {study_rf.best_value:.6f}")
print(f"Best RF params: {study_rf.best_params}")

best_rf_params = study_rf.best_params
best_rf_params['class_weight'] = 'balanced'
best_rf_params['random_state'] = RANDOM_STATE
best_rf_params['n_jobs'] = -1


# ===== スタッキング =====
print("\n[5/7] スタッキング（メタ学習）...")

# ベースモデル
base_models = [
    ('rf', RandomForestClassifier(**best_rf_params)),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.02, num_leaves=63, max_depth=17,
        min_child_samples=20, class_weight='balanced', random_state=RANDOM_STATE, verbosity=-1
    )),
    ('xgb', xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.02, max_depth=10,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        random_state=RANDOM_STATE, verbosity=0
    ))
]

# メタモデル
meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

print("スタッキングモデル学習中...")
stacking_model.fit(X_train, y_train)

stacking_pred = stacking_model.predict(X_valid)
stacking_acc = accuracy_score(y_valid, stacking_pred)
print(f"Stacking Accuracy: {stacking_acc:.6f} ({stacking_acc*100:.4f}%)")


# ===== 閾値の最適化 =====
print("\n[6/7] 閾値の最適化...")

stacking_pred_proba = stacking_model.predict_proba(X_valid)[:, 1]

best_threshold = 0.5
best_acc_threshold = stacking_acc

for threshold in np.arange(0.3, 0.8, 0.01):
    pred_threshold = (stacking_pred_proba >= threshold).astype(int)
    acc_threshold = accuracy_score(y_valid, pred_threshold)
    if acc_threshold > best_acc_threshold:
        best_acc_threshold = acc_threshold
        best_threshold = threshold

print(f"\n最適閾値: {best_threshold:.3f}")
print(f"閾値最適化後の精度: {best_acc_threshold:.6f} ({best_acc_threshold*100:.4f}%)")

final_pred = (stacking_pred_proba >= best_threshold).astype(int)

# 詳細評価
precision = precision_score(y_valid, final_pred)
recall = recall_score(y_valid, final_pred)
f1 = f1_score(y_valid, final_pred)
auc = roc_auc_score(y_valid, stacking_pred_proba)

print("\n" + "=" * 80)
print("最終スタッキングモデルの性能")
print("=" * 80)
print(f"Accuracy:  {best_acc_threshold:.6f} ({best_acc_threshold*100:.4f}%)")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")
print(f"AUC:       {auc:.6f}")
print(f"Threshold: {best_threshold:.3f}")
print("=" * 80)

print("\n混同行列:")
cm = confusion_matrix(y_valid, final_pred)
print(cm)

# モデル保存
with open('data/stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)
with open('data/stacking_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
with open('data/best_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

print("\nモデルとパラメータを保存しました")


# ===== テストデータ予測 =====
print("\n[7/7] テストデータ予測...")
test_processed, _ = advanced_feature_engineering(test_df, is_train=False, encoders=encoders)
X_test = test_processed.drop(columns=['id'])

for col in X_test.columns:
    if X_test[col].dtype.name == 'category':
        X_test[col] = X_test[col].cat.codes
    X_test[col] = X_test[col].astype('float64')

test_pred_proba = stacking_model.predict_proba(X_test.values)[:, 1]
test_pred = (test_pred_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_pred
})
submission.to_csv('data/stacking_submission.csv', index=False, header=False)

print(f"\n提出ファイル作成: data/stacking_submission.csv")
print(f"予測y=1の割合: {test_pred.mean():.4f} ({test_pred.mean()*100:.2f}%)")

print("\n" + "=" * 80)
print("完了！")
print("=" * 80)
