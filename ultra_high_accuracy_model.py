#!/usr/bin/env python3
"""
99%精度を目指す超高精度モデル
EDAの知見を活かした特徴量エンジニアリング + 複数モデル比較 + アンサンブル
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
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
print("99%精度を目指す超高精度モデル学習")
print("=" * 80)

# ===== データ読み込み =====
print("\n[1/9] データ読み込み...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")


# ===== EDAの知見を活かした超高度な特徴量エンジニアリング =====
def advanced_feature_engineering(df, is_train=True, encoders=None):
    """
    EDAで発見した重要なパターンを全て活用
    """
    df = df.copy()

    # ===== 1. 超重要特徴量: MONTH（50%以上の陽性率がある月） =====
    high_conversion_months = ['mar', 'dec', 'sep', 'oct']
    df['is_high_conversion_month'] = df['month'].isin(high_conversion_months).astype(int)

    # 月をカテゴリに分類
    df['month_category'] = df['month'].map({
        'mar': 'very_high', 'dec': 'very_high', 'sep': 'very_high', 'oct': 'very_high',
        'apr': 'medium', 'feb': 'medium',
        'aug': 'low', 'jun': 'low', 'nov': 'low', 'jan': 'low', 'jul': 'low', 'may': 'low'
    })

    # ===== 2. DURATION - 最重要特徴量 =====
    # EDAより: duration>1000で60%以上の陽性率
    df['duration_very_long'] = (df['duration'] > 1000).astype(int)
    df['duration_long'] = ((df['duration'] > 500) & (df['duration'] <= 1000)).astype(int)
    df['duration_medium'] = ((df['duration'] > 200) & (df['duration'] <= 500)).astype(int)
    df['duration_short'] = (df['duration'] <= 200).astype(int)

    # 対数変換
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_squared'] = df['duration'] ** 2
    df['duration_sqrt'] = np.sqrt(df['duration'])

    # ===== 3. POUTCOME - success=64.79%の陽性率 =====
    df['poutcome_success'] = (df['poutcome'] == 'success').astype(int)
    df['poutcome_failure'] = (df['poutcome'] == 'failure').astype(int)
    df['poutcome_other'] = (df['poutcome'] == 'other').astype(int)
    df['poutcome_unknown'] = (df['poutcome'] == 'unknown').astype(int)

    # ===== 4. CONTACT - cellular=14.95% vs unknown=4.01% =====
    df['contact_cellular'] = (df['contact'] == 'cellular').astype(int)
    df['contact_unknown'] = (df['contact'] == 'unknown').astype(int)

    # ===== 5. JOB - student=29.80%, retired=22.36% =====
    df['job_high_conversion'] = df['job'].isin(['student', 'retired']).astype(int)
    df['job_low_conversion'] = df['job'].isin(['blue-collar', 'entrepreneur', 'housemaid']).astype(int)

    # ===== 6. HOUSING & LOAN =====
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df['no_housing'] = (df['housing'] == 0).astype(int)  # housing=no は16.76%
    df['no_loan'] = (df['loan'] == 0).astype(int)
    df['total_loans'] = df['housing'] + df['loan']
    df['no_loans_at_all'] = ((df['housing'] == 0) & (df['loan'] == 0)).astype(int)

    # ===== 7. PREVIOUS CONTACT =====
    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['pdays_log'] = np.log1p(df['pdays'].replace(-1, 0))
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    # ===== 8. AGE & BALANCE =====
    df['age_squared'] = df['age'] ** 2
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_high'] = (df['balance'] > 1500).astype(int)

    # ===== 9. CAMPAIGN =====
    df['campaign_low'] = (df['campaign'] <= 2).astype(int)  # campaign少ない方が良い
    df['campaign_high'] = (df['campaign'] > 5).astype(int)

    # ===== 10. 複合特徴量（EDAから） =====
    # duration × contact の組み合わせ（cellular & 非常に長い = 62.5%）
    df['duration_long_cellular'] = ((df['duration'] > 500) & (df['contact'] == 'cellular')).astype(int)

    # poutcome=success × has_previous （64.8%の陽性率）
    df['success_with_previous'] = ((df['poutcome'] == 'success') & (df['pdays'] != -1)).astype(int)

    # high conversion month × long duration
    df['high_month_long_duration'] = (df['is_high_conversion_month'] & (df['duration'] > 300)).astype(int)

    # campaign効率
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)

    # ===== 11. MARITAL & EDUCATION =====
    df['marital_single'] = (df['marital'] == 'single').astype(int)  # single=14.92%
    df['education_tertiary'] = (df['education'] == 'tertiary').astype(int)  # tertiary=14.69%

    # ===== 12. 月の周期性 =====
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)
    df['quarter'] = ((df['month_numeric'] - 1) // 3) + 1

    # ===== 13. カテゴリカル変数のエンコーディング =====
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

        # 不要な列を削除
        df = df.drop(columns=['month', 'month_numeric'])

        encoders = {
            'target_encoders': target_encoders,
            'label_encoders': label_encoders
        }
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
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        df = df.drop(columns=['month', 'month_numeric'])

        return df, None


print("\n[2/9] 超高度な特徴量エンジニアリング...")
train_processed, encoders = advanced_feature_engineering(train_df, is_train=True)
print(f"New shape: {train_processed.shape}")

# ===== データ分割 =====
print("\n[3/9] データ分割...")
y = train_processed['y']
X = train_processed.drop(columns=['id', 'y'])
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
print(f"Total features: {X.shape[1]}")


# ===== 複数モデルの比較 =====
print("\n[4/9] 複数モデルの比較...")

models_to_compare = {}

# 1. LightGBM
print("\n--- LightGBM ---")
lgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 17,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'objective': 'binary',
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'verbosity': -1
}
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(150, verbose=False)])
lgb_pred = lgb_model.predict(X_valid)
lgb_acc = accuracy_score(y_valid, lgb_pred)
print(f"LightGBM Accuracy: {lgb_acc:.6f} ({lgb_acc*100:.4f}%)")
models_to_compare['LightGBM'] = (lgb_model, lgb_acc)

# 2. XGBoost
print("\n--- XGBoost ---")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 10,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
xgb_pred = xgb_model.predict(X_valid)
xgb_acc = accuracy_score(y_valid, xgb_pred)
print(f"XGBoost Accuracy: {xgb_acc:.6f} ({xgb_acc*100:.4f}%)")
models_to_compare['XGBoost'] = (xgb_model, xgb_acc)

# 3. CatBoost
print("\n--- CatBoost ---")
catboost_params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 10,
    'l2_leaf_reg': 3,
    'auto_class_weights': 'Balanced',
    'random_state': RANDOM_STATE,
    'verbose': 0
}
catboost_model = CatBoostClassifier(**catboost_params)
catboost_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=150, verbose=False)
catboost_pred = catboost_model.predict(X_valid)
catboost_acc = accuracy_score(y_valid, catboost_pred)
print(f"CatBoost Accuracy: {catboost_acc:.6f} ({catboost_acc*100:.4f}%)")
models_to_compare['CatBoost'] = (catboost_model, catboost_acc)

# 4. Random Forest
print("\n--- Random Forest ---")
rf_params = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_valid)
rf_acc = accuracy_score(y_valid, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.6f} ({rf_acc*100:.4f}%)")
models_to_compare['RandomForest'] = (rf_model, rf_acc)

# ベストモデルの選択
best_model_name = max(models_to_compare, key=lambda k: models_to_compare[k][1])
best_model, best_acc = models_to_compare[best_model_name]
print(f"\n最高精度モデル: {best_model_name} ({best_acc*100:.4f}%)")


# ===== アンサンブル（Voting） =====
print("\n[5/9] アンサンブル学習（Voting）...")
voting_model = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('catboost', catboost_model),
        ('rf', rf_model)
    ],
    voting='soft',
    n_jobs=-1
)
voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_valid)
voting_acc = accuracy_score(y_valid, voting_pred)
print(f"Voting Ensemble Accuracy: {voting_acc:.6f} ({voting_acc*100:.4f}%)")


# ===== 最終モデルの選択 =====
print("\n[6/9] 最終モデル選択...")
if voting_acc > best_acc:
    final_model = voting_model
    final_acc = voting_acc
    final_pred = voting_pred
    final_name = "VotingEnsemble"
else:
    final_model = best_model
    final_acc = best_acc
    final_pred = best_model.predict(X_valid)
    final_name = best_model_name

print(f"最終モデル: {final_name}")
print(f"最終精度: {final_acc:.6f} ({final_acc*100:.4f}%)")


# ===== 詳細評価 =====
print("\n[7/9] 詳細評価...")
precision = precision_score(y_valid, final_pred)
recall = recall_score(y_valid, final_pred)
f1 = f1_score(y_valid, final_pred)

if hasattr(final_model, 'predict_proba'):
    final_pred_proba = final_model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, final_pred_proba)
else:
    auc = 0.0

print("\n" + "=" * 80)
print("最終モデルの性能")
print("=" * 80)
print(f"Model:     {final_name}")
print(f"Accuracy:  {final_acc:.6f} ({final_acc*100:.4f}%)")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")
print(f"AUC:       {auc:.6f}")
print("=" * 80)

print("\n混同行列:")
cm = confusion_matrix(y_valid, final_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_valid, final_pred))


# ===== モデル保存 =====
print("\n[8/9] モデル保存...")
with open('data/ultra_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
with open('data/ultra_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("モデルとエンコーダーを保存しました")


# ===== テストデータ予測 =====
print("\n[9/9] テストデータ予測...")
test_processed, _ = advanced_feature_engineering(test_df, is_train=False, encoders=encoders)
X_test = test_processed.drop(columns=['id'])

# float型に変換
for col in X_test.columns:
    if X_test[col].dtype.name == 'category':
        X_test[col] = X_test[col].cat.codes
    X_test[col] = X_test[col].astype('float64')

test_pred = final_model.predict(X_test.values)

submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_pred
})
submission.to_csv('data/ultra_high_accuracy_submission.csv', index=False, header=False)

print(f"\n提出ファイル作成: data/ultra_high_accuracy_submission.csv")
print(f"予測y=1の割合: {test_pred.mean():.4f} ({test_pred.mean()*100:.2f}%)")
print(f"予測y=1の数: {test_pred.sum()}")

print("\n" + "=" * 80)
print("完了！")
print("=" * 80)
