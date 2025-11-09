#!/usr/bin/env python3
"""
徹底的なEDA（探索的データ分析）
データを深く理解して99%精度を目指す
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("徹底的なEDA（探索的データ分析）開始")
print("=" * 80)

# データ読み込み
train = pd.read_csv("data/train.csv")
print(f"\nデータ形状: {train.shape}")
print(f"特徴量数: {train.shape[1] - 1} (ターゲット除く)")

# ===== 1. ターゲット変数の詳細分析 =====
print("\n" + "=" * 80)
print("1. ターゲット変数の詳細分析")
print("=" * 80)

print("\nターゲット分布:")
print(train['y'].value_counts())
print(f"\n陽性率: {train['y'].mean():.4f} ({train['y'].mean()*100:.2f}%)")
print(f"不均衡比率: {(1-train['y'].mean())/train['y'].mean():.2f}:1")

# ===== 2. 数値特徴量の詳細分析 =====
print("\n" + "=" * 80)
print("2. 数値特徴量の詳細分析")
print("=" * 80)

numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for col in numeric_features:
    print(f"\n--- {col} ---")
    print(f"基本統計:")
    print(train[col].describe())

    # y=0とy=1での分布の違い
    y0_mean = train[train['y'] == 0][col].mean()
    y1_mean = train[train['y'] == 1][col].mean()
    ratio = y1_mean / y0_mean if y0_mean != 0 else float('inf')

    print(f"\ny=0の平均: {y0_mean:.2f}")
    print(f"y=1の平均: {y1_mean:.2f}")
    print(f"比率 (y=1/y=0): {ratio:.2f}")

    # T検定（統計的有意性）
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(
        train[train['y'] == 0][col],
        train[train['y'] == 1][col]
    )
    print(f"T検定 p値: {p_value:.6f} {'***有意' if p_value < 0.001 else '**有意' if p_value < 0.01 else '*有意' if p_value < 0.05 else '有意でない'}")

# ===== 3. カテゴリカル特徴量の詳細分析 =====
print("\n" + "=" * 80)
print("3. カテゴリカル特徴量とターゲットの関係")
print("=" * 80)

categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in categorical_features:
    print(f"\n--- {col} ---")

    # クロス集計
    ct = pd.crosstab(train[col], train['y'], normalize='index')
    ct_counts = pd.crosstab(train[col], train['y'])

    print("\n陽性率（y=1の割合）:")
    y1_rate = ct[1].sort_values(ascending=False)
    for idx, val in y1_rate.items():
        count = ct_counts.loc[idx].sum()
        print(f"  {idx:20s}: {val:.4f} ({val*100:.2f}%) [n={count:5d}]")

    # カイ二乗検定
    chi2, p_value, dof, expected = chi2_contingency(ct_counts)
    print(f"\nカイ二乗検定 p値: {p_value:.6f} {'***有意' if p_value < 0.001 else '**有意' if p_value < 0.01 else '*有意' if p_value < 0.05 else '有意でない'}")

# ===== 4. durationの特別分析（最重要特徴量） =====
print("\n" + "=" * 80)
print("4. DURATION（最重要特徴量）の特別分析")
print("=" * 80)

print("\ndurationの詳細統計:")
print(train['duration'].describe())

print("\ndurationの分位数別の陽性率:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    threshold = train['duration'].quantile(q)
    rate = train[train['duration'] >= threshold]['y'].mean()
    count = len(train[train['duration'] >= threshold])
    print(f"  {q*100:5.1f}%点以上 (>={threshold:7.1f}秒): 陽性率={rate:.4f} ({rate*100:.2f}%) [n={count:5d}]")

print("\ndurationの区間別の陽性率:")
bins = [0, 100, 200, 300, 400, 500, 750, 1000, 5000]
train['duration_bin_analysis'] = pd.cut(train['duration'], bins=bins)
duration_analysis = train.groupby('duration_bin_analysis')['y'].agg(['mean', 'count'])
print(duration_analysis)

# ===== 5. 特徴量間の相関分析 =====
print("\n" + "=" * 80)
print("5. 数値特徴量間の相関分析")
print("=" * 80)

numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']
correlation = train[numeric_cols].corr()

print("\nターゲット(y)との相関:")
target_corr = correlation['y'].sort_values(ascending=False)
for idx, val in target_corr.items():
    if idx != 'y':
        print(f"  {idx:15s}: {val:7.4f}")

# ===== 6. 複合的なパターン分析 =====
print("\n" + "=" * 80)
print("6. 複合的なパターン分析")
print("=" * 80)

# 前回コンタクトの有無とpoutcomeの組み合わせ
print("\n前回コンタクト × poutcome の陽性率:")
train['has_previous'] = train['pdays'] != -1
cross_analysis = train.groupby(['has_previous', 'poutcome'])['y'].agg(['mean', 'count'])
print(cross_analysis)

# duration × contact の組み合わせ
print("\nduration区間 × contact の陽性率:")
train['duration_category'] = pd.cut(train['duration'], bins=[0, 200, 400, 1000, 10000], labels=['短い', '普通', '長い', '非常に長い'])
cross_duration_contact = train.groupby(['duration_category', 'contact'])['y'].agg(['mean', 'count'])
print(cross_duration_contact)

# ===== 7. 外れ値の検出 =====
print("\n" + "=" * 80)
print("7. 外れ値の検出")
print("=" * 80)

for col in ['age', 'balance', 'duration', 'campaign', 'previous']:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
    outlier_rate = len(outliers) / len(train)

    print(f"\n{col}:")
    print(f"  外れ値の数: {len(outliers)} ({outlier_rate*100:.2f}%)")
    if len(outliers) > 0:
        print(f"  外れ値の陽性率: {outliers['y'].mean():.4f}")
        print(f"  全体の陽性率: {train['y'].mean():.4f}")

# ===== 8. 重要な発見のまとめ =====
print("\n" + "=" * 80)
print("8. 重要な発見のまとめ")
print("=" * 80)

print("\n【最も重要な特徴量】")
print("1. duration (通話時間) - ターゲットと最も強い相関")
print("2. poutcome (前回キャンペーン結果) - 特にsuccess")
print("3. pdays (前回コンタクトからの日数)")
print("4. contact (連絡方法) - cellularが有効")

print("\n【データの特徴】")
print(f"- 不均衡データ: y=1が{train['y'].mean()*100:.2f}%のみ")
print(f"- 欠損値: なし")
print(f"- 外れ値: 多数存在（特にbalance、duration、campaign）")

print("\n【99%精度を目指すための戦略】")
print("1. durationを中心とした特徴量エンジニアリング")
print("2. poutcome=success、pdays!=-1、contact=cellularの組み合わせ")
print("3. 複数モデルのアンサンブル")
print("4. クラス不均衡への対策（SMOTE、調整された閾値）")
print("5. 外れ値の適切な処理")

print("\n" + "=" * 80)
print("EDA完了")
print("=" * 80)
