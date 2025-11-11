"""
現在のモデルの詳細分析スクリプト
- 特徴量重要度
- 誤分類パターン
- データの統計分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import lightgbm as lgb
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

def main():
    print("=" * 80)
    print("モデル詳細分析")
    print("=" * 80)

    # データ読み込み
    train_df = pd.read_csv("/home/user/bank/data/train.csv")

    print("\n[1] データの基本統計")
    print("-" * 80)
    print(f"訓練データサイズ: {len(train_df)}")
    print(f"\nクラス分布:")
    print(train_df['y'].value_counts())
    neg = (train_df['y'] == 0).sum()
    pos = (train_df['y'] == 1).sum()
    print(f"不均衡比率: 1:{neg/pos:.2f}")

    # 特徴量エンジニアリング
    train_processed, categorical_cols = feature_engineering(train_df, is_train=True)
    train_encoded = pd.get_dummies(train_processed, columns=categorical_cols, drop_first=True)

    y = train_encoded['y']
    X = train_encoded.drop(columns=['id', 'y'])

    print(f"\n特徴量数: {X.shape[1]}")

    # 元のデータの統計
    print("\n[2] 元データの統計（正解ラベル別）")
    print("-" * 80)
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    for col in numerical_cols:
        if col in train_df.columns:
            print(f"\n{col}:")
            print(f"  全体: mean={train_df[col].mean():.2f}, std={train_df[col].std():.2f}")
            print(f"  y=0: mean={train_df[train_df['y']==0][col].mean():.2f}, std={train_df[train_df['y']==0][col].std():.2f}")
            print(f"  y=1: mean={train_df[train_df['y']==1][col].mean():.2f}, std={train_df[train_df['y']==1][col].std():.2f}")

    # モデル学習と特徴量重要度
    print("\n[3] LightGBMモデルの学習と特徴量重要度")
    print("-" * 80)

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X))
    feature_importances = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"Training Fold {fold + 1}/5...")

        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)]
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
        feature_importances.append(model.feature_importances_)

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nOOF AUC: {oof_auc:.5f}")

    # 特徴量重要度
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': np.mean(feature_importances, axis=0)
    }).sort_values('importance', ascending=False)

    print("\nTop 30 重要な特徴量:")
    print(feature_importance_df.head(30).to_string(index=False))

    # 誤分類分析
    print("\n[4] 誤分類パターン分析")
    print("-" * 80)

    # 最適閾値を見つける
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (oof_preds >= threshold).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"最適閾値: {best_threshold:.3f}")
    print(f"最適F1スコア: {best_f1:.5f}")

    # 最適閾値での予測
    oof_preds_binary = (oof_preds >= best_threshold).astype(int)

    # 混同行列
    cm = confusion_matrix(y, oof_preds_binary)
    print(f"\n混同行列:")
    print(f"              予測=0    予測=1")
    print(f"実際=0 (TN/FP)  {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"実際=1 (FN/TP)  {cm[1,0]:6d}    {cm[1,1]:6d}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negative:  {tn} ({tn/(tn+fp)*100:.2f}%)")
    print(f"False Positive: {fp} ({fp/(tn+fp)*100:.2f}%)")
    print(f"False Negative: {fn} ({fn/(fn+tp)*100:.2f}%)")
    print(f"True Positive:  {tp} ({tp/(fn+tp)*100:.2f}%)")

    # 詳細レポート
    print("\n分類レポート:")
    print(classification_report(y, oof_preds_binary, target_names=['Class 0', 'Class 1']))

    # 誤分類サンプルの分析
    print("\n[5] 誤分類サンプルの詳細分析")
    print("-" * 80)

    # False Positives (0を1と誤分類)
    fp_indices = ((y == 0) & (oof_preds_binary == 1))
    fp_preds_proba = oof_preds[fp_indices]
    print(f"\nFalse Positives: {fp_indices.sum()}件")
    print(f"予測確率の分布: mean={fp_preds_proba.mean():.3f}, std={fp_preds_proba.std():.3f}")
    print(f"予測確率範囲: [{fp_preds_proba.min():.3f}, {fp_preds_proba.max():.3f}]")

    # False Negatives (1を0と誤分類)
    fn_indices = ((y == 1) & (oof_preds_binary == 0))
    fn_preds_proba = oof_preds[fn_indices]
    print(f"\nFalse Negatives: {fn_indices.sum()}件")
    print(f"予測確率の分布: mean={fn_preds_proba.mean():.3f}, std={fn_preds_proba.std():.3f}")
    print(f"予測確率範囲: [{fn_preds_proba.min():.3f}, {fn_preds_proba.max():.3f}]")

    # 予測確率の分布
    print("\n[6] 予測確率の分布")
    print("-" * 80)
    print(f"y=0の予測確率: mean={oof_preds[y==0].mean():.3f}, std={oof_preds[y==0].std():.3f}")
    print(f"y=1の予測確率: mean={oof_preds[y==1].mean():.3f}, std={oof_preds[y==1].std():.3f}")

    # 改善提案
    print("\n" + "=" * 80)
    print("改善提案")
    print("=" * 80)

    print("\n1. 特徴量エンジニアリング:")
    print("   - duration（通話時間）が最重要特徴量")
    print("   - より多くの統計量特徴量（平均、中央値、分位数）を追加")
    print("   - より高次の交互作用特徴量を追加")
    print("   - Target Encodingを試す")

    print("\n2. モデルの改善:")
    print("   - より多様なモデルのアンサンブル（CatBoost, Neural Network, TabNet）")
    print("   - より長時間のハイパーパラメータ最適化（50-100 trials）")
    print("   - より深いニューラルネットワーク")

    print("\n3. データの改善:")
    print("   - 外れ値の処理")
    print("   - より高度な不均衡対策（SMOTE、focal lossなど）")
    print("   - Adversarial Validationでtrain/testの分布の違いを確認")

    print("\n4. アンサンブルの改善:")
    print("   - Weighted Blending（各モデルの重みを最適化）")
    print("   - Multi-Layer Stacking")
    print("   - Out-of-Fold予測の品質向上")

    # 結果を保存
    feature_importance_df.to_csv("/home/user/bank/data/feature_importance.csv", index=False)

    # 予測確率をDataFrameに保存
    analysis_df = train_df.copy()
    analysis_df['oof_pred_proba'] = oof_preds
    analysis_df['oof_pred_binary'] = oof_preds_binary
    analysis_df['is_fp'] = fp_indices.astype(int)
    analysis_df['is_fn'] = fn_indices.astype(int)
    analysis_df.to_csv("/home/user/bank/data/model_analysis.csv", index=False)

    print("\n分析結果を保存しました:")
    print("  - data/feature_importance.csv")
    print("  - data/model_analysis.csv")

if __name__ == "__main__":
    main()
