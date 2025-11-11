"""
特徴量分析スクリプト
- 相関分析
- 特徴量重要度分析
- 多重共線性チェック（VIF）
- 根拠に基づく特徴量選択
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# statsmodelsは必須ではない
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not found. VIF calculation will be skipped.")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def feature_engineering(df, is_train=True):
    """特徴量エンジニアリング"""
    df = df.copy()

    # 年齢グループ
    df['age_group'] = pd.cut(df['age'], bins=16).astype(str)
    df['age_group'] = df['age_group'].str.replace(r'[(),.\[\] ]', '_', regex=True)

    # balance関連
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)

    # 時系列特徴量
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['campaign_efficiency'] = df['duration'] / (df['campaign'] + 1)
    df['duration_log'] = np.log1p(df['duration'])

    # previous関連
    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['previous_per_pdays'] = df['previous'] / (df['pdays'].replace(-1, 1) + 1)

    # 月の周期性
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_numeric'] = df['month'].map(month_mapping)
    df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

    # ローン関連
    df['total_loans'] = (df['housing'] == 'yes').astype(int) + (df['loan'] == 'yes').astype(int)
    df['has_any_loan'] = (df['total_loans'] > 0).astype(int)

    # バイナリ変数を数値化
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # カテゴリカル変数
    categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'age_group']

    # 相互作用特徴量
    df['job_education'] = df['job'].astype(str) + '_' + df['education'].astype(str)
    df['contact_month'] = df['contact'].astype(str) + '_' + df['month'].astype(str)

    interaction_cols = ['job_education', 'contact_month']
    categorical_cols.extend(interaction_cols)

    df = df.drop(columns=['month', 'month_numeric'])

    return df, categorical_cols


def analyze_correlations(X, y):
    """相関分析"""
    print("\n" + "="*60)
    print("1. ターゲット変数との相関分析")
    print("="*60)

    # 数値特徴量のみ抽出
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # 相関係数を計算
    correlations = []
    for col in numeric_cols:
        corr = np.corrcoef(X[col].fillna(0), y)[0, 1]
        correlations.append({'feature': col, 'correlation': abs(corr), 'correlation_raw': corr})

    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

    print("\nTop 20 相関の高い特徴量:")
    print(corr_df.head(20).to_string(index=False))

    return corr_df


def calculate_vif(X):
    """VIF（多重共線性）計算"""
    print("\n" + "="*60)
    print("2. 多重共線性チェック（VIF）")
    print("="*60)

    if not HAS_STATSMODELS:
        print("statsmodelsがインストールされていないため、VIF計算をスキップします。")
        print("代わりに、相関行列を使用して多重共線性を簡易チェックします。")

        # 相関行列による簡易チェック
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].fillna(0)

        # 相関行列を計算
        corr_matrix = X_numeric.corr().abs()

        # 高い相関を持つペアを探す
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        if high_corr_pairs:
            print("\n高い相関を持つ特徴量ペア（|r| > 0.8）:")
            for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)[:10]:
                print(f"  {pair['feature1']:25s} <-> {pair['feature2']:25s}  r={pair['correlation']:.3f}")
        else:
            print("\n高い相関を持つペアは見つかりませんでした。")

        # ダミーのVIF DataFrameを返す（全てVIF=1とする）
        vif_df = pd.DataFrame([{'feature': col, 'VIF': 1.0} for col in numeric_cols])
        return vif_df

    print("VIF > 10: 多重共線性の問題あり")
    print("VIF > 5: 注意が必要")

    # 数値特徴量のみ抽出
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols].fillna(0)

    # VIF計算（計算量が多いので主要な特徴量のみ）
    vif_data = []
    for i, col in enumerate(numeric_cols[:30]):  # 最初の30個のみ
        try:
            vif = variance_inflation_factor(X_numeric.values, i)
            vif_data.append({'feature': col, 'VIF': vif})
        except:
            pass

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    print("\nVIFが高い特徴量（Top 20）:")
    print(vif_df.head(20).to_string(index=False))

    print("\n多重共線性の問題がある特徴量（VIF > 10）:")
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        print(high_vif.to_string(index=False))
    else:
        print("なし")

    return vif_df


def calculate_feature_importance(X, y):
    """特徴量重要度分析"""
    print("\n" + "="*60)
    print("3. 特徴量重要度分析（LightGBM）")
    print("="*60)

    # Train/Validationに分割
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # LightGBMで学習
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbosity=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # 特徴量重要度
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # AUC計算
    pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, pred)

    print(f"\nValidation AUC: {auc:.5f}")
    print("\nTop 30 重要な特徴量:")
    print(importance_df.head(30).to_string(index=False))

    return importance_df, model


def select_features(corr_df, vif_df, importance_df):
    """根拠に基づいた特徴量選択"""
    print("\n" + "="*60)
    print("4. 特徴量選択の根拠")
    print("="*60)

    # 重要度上位の特徴量
    top_important = set(importance_df.head(40)['feature'].tolist())

    # 相関が高い特徴量
    top_correlated = set(corr_df.head(30)['feature'].tolist())

    # VIFが低い特徴量（多重共線性が低い）
    if HAS_STATSMODELS:
        low_vif = set(vif_df[vif_df['VIF'] < 10]['feature'].tolist())
        vif_criterion = "VIF < 10（多重共線性が低い）"
    else:
        # VIFがない場合は全ての特徴量を許可（相関チェックは既に行っている）
        low_vif = set(vif_df['feature'].tolist())
        vif_criterion = "（VIF計算スキップ）"

    print("\n選択基準:")
    print("1. 特徴量重要度 Top 40")
    print("2. ターゲット変数との相関 Top 30")
    print(f"3. {vif_criterion}")

    # 重要度または相関が高く、かつVIFが低い特徴量を選択
    selected_numeric = (top_important | top_correlated) & low_vif

    print(f"\n選択された数値特徴量数: {len(selected_numeric)}")
    print("\n選択された特徴量:")
    for feat in sorted(selected_numeric):
        imp = importance_df[importance_df['feature'] == feat]['importance'].values
        imp_val = imp[0] if len(imp) > 0 else 0

        corr = corr_df[corr_df['feature'] == feat]['correlation'].values
        corr_val = corr[0] if len(corr) > 0 else 0

        vif = vif_df[vif_df['feature'] == feat]['VIF'].values
        vif_val = vif[0] if len(vif) > 0 else 0

        if HAS_STATSMODELS:
            print(f"  {feat:30s} | Importance: {imp_val:8.1f} | Corr: {corr_val:.4f} | VIF: {vif_val:.2f}")
        else:
            print(f"  {feat:30s} | Importance: {imp_val:8.1f} | Corr: {corr_val:.4f}")

    return list(selected_numeric)


def main():
    print("="*60)
    print("特徴量分析・選択スクリプト")
    print("="*60)

    # データ読み込み
    print("\nデータ読み込み...")
    train_df = pd.read_csv("/home/user/bank/data/train.csv")

    # 特徴量エンジニアリング
    print("特徴量エンジニアリング...")
    train_processed, categorical_cols = feature_engineering(train_df, is_train=True)

    # ワンホットエンコーディング
    print("ワンホットエンコーディング...")
    train_encoded = pd.get_dummies(train_processed, columns=categorical_cols, drop_first=True)

    y = train_encoded['y']
    X = train_encoded.drop(columns=['id', 'y'])

    print(f"総特徴量数: {X.shape[1]}")

    # 1. 相関分析
    corr_df = analyze_correlations(X, y)

    # 2. VIF計算
    vif_df = calculate_vif(X)

    # 3. 特徴量重要度
    importance_df, model = calculate_feature_importance(X, y)

    # 4. 特徴量選択
    selected_features = select_features(corr_df, vif_df, importance_df)

    # カテゴリカル特徴量も含める（ワンホット化されたもの）
    categorical_features = [col for col in X.columns if any(
        cat in col for cat in ['job_', 'marital_', 'education_', 'contact_',
                               'poutcome_', 'age_group_', 'job_education_', 'contact_month_']
    )]

    # 最終的な特徴量リスト
    final_features = selected_features + categorical_features

    print("\n" + "="*60)
    print("最終選択特徴量")
    print("="*60)
    print(f"数値特徴量: {len(selected_features)}")
    print(f"カテゴリカル特徴量（ワンホット化後）: {len(categorical_features)}")
    print(f"合計: {len(final_features)}")

    # 結果をファイルに保存
    with open('/home/user/bank/data/selected_features.txt', 'w') as f:
        f.write("# 選択された数値特徴量\n")
        for feat in sorted(selected_features):
            f.write(f"{feat}\n")

    print("\n選択された特徴量を保存しました: /home/user/bank/data/selected_features.txt")

    # 簡易的な性能評価
    print("\n" + "="*60)
    print("性能評価（選択後の特徴量）")
    print("="*60)

    X_selected = X[final_features]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_selected = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbosity=-1
    )

    model_selected.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    pred_selected = model_selected.predict_proba(X_valid)[:, 1]
    auc_selected = roc_auc_score(y_valid, pred_selected)

    print(f"選択前の特徴量数: {X.shape[1]}")
    print(f"選択後の特徴量数: {X_selected.shape[1]}")
    print(f"削減率: {(1 - X_selected.shape[1] / X.shape[1]) * 100:.1f}%")
    print(f"\nValidation AUC（全特徴量）: 基準モデル参照")
    print(f"Validation AUC（選択後）: {auc_selected:.5f}")


if __name__ == "__main__":
    main()
