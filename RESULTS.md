# LightGBM 高精度モデル - 結果レポート

## 概要
train.csvを分析して効果的な特徴量を作成し、LightGBMモデルを最適化しました。

## データ分析の主要な発見

### 1. ターゲット分布
- データ数: 27,128
- 陽性率 (y=1): 11.7%
- 不均衡データのため、class_weight='balanced'を使用

### 2. 最重要特徴量: duration（通話時間）
- **y=0の平均duration**: 223秒
- **y=1の平均duration**: 548秒
- **約2.5倍の差** - 強い相関関係

### 3. その他の重要な発見
- 前回コンタクト(pdays!=-1)がある場合、成約率が2倍以上
- balanceに負の値が存在（最小: -6,847）
- 欠損値なし

## 特徴量エンジニアリング

### 実装した特徴量

#### 1. Duration関連（最重要）
- `duration_log`: 対数変換
- `duration_squared`: 2乗（非線形性）
- `duration_sqrt`: 平方根
- `duration_bin`: カテゴリ化（very_short～extremely_long）
- `duration_per_day`: 1日あたりの通話時間
- `duration_per_campaign`: キャンペーン効率

#### 2. 数値特徴量
- `age_squared`: 年齢の2乗
- `balance_log`: 残高の対数変換
- `balance_positive/negative`: 残高の正負フラグ

#### 3. 時系列特徴量
- `month_sin/cos`: 月の周期性エンコーディング
- `quarter`: 四半期

#### 4. 複合特徴量
- `campaign_intensity`: campaign × duration
- `day_duration_interaction`: day × duration
- `previous_per_pdays`: 前回コンタクト効率

#### 5. エンコーディング
- **Target Encoding**: カテゴリ変数とターゲットの関係を利用（smoothing適用）
- **Frequency Encoding**: カテゴリの出現頻度
- **Label Encoding**: カテゴリを数値に変換

## モデル性能

### 検証データでの結果
```
============================================================
Accuracy:  89.85% ★
Precision: 0.55
Recall:    0.70
F1 Score:  0.62
AUC:       0.93
============================================================
```

### 混同行列
```
              予測0   予測1
実際0         4428    363
実際1          188    447
```

### 特徴量重要度 Top 10
1. **balance** (8,596)
2. **day_duration_interaction** (8,255)
3. **duration_per_day** (7,927)
4. **duration_per_campaign** (6,992)
5. **duration** (6,804)
6. **campaign_intensity** (6,694)
7. **age** (6,607)
8. **balance_log** (4,891)
9. **day** (4,283)
10. **month_cos** (3,046)

## ハイパーパラメータ

最適化されたパラメータ（Optuna, 50 trials）:
- learning_rate: 0.0186
- num_leaves: 58
- max_depth: 17
- min_child_samples: 20
- subsample: 0.884
- colsample_bytree: 0.928
- reg_alpha: 1.27e-06
- reg_lambda: 0.000119
- min_split_gain: 0.101
- class_weight: balanced

## テストデータ予測結果

- **予測されたy=1の割合**: 14.57%
- 予測されたy=1の数: 2,635
- 予測されたy=0の数: 15,448

訓練データの陽性率(11.7%)と比較的近い値で、妥当な予測結果です。

## 改善点

### 元のモデルからの主な改善
1. ✅ バグ修正（month列の扱い）
2. ✅ duration関連の特徴量を大幅強化
3. ✅ より効果的なエンコーディング手法
4. ✅ ハイパーパラメータの最適化
5. ✅ 不均衡データへの対応

### さらなる改善の可能性
1. **アンサンブル学習**: XGBoost, CatBoost等との組み合わせ
2. **SMOTE等**: 不均衡データ対策の強化
3. **より高度な特徴量**:
   - 時系列パターンの抽出
   - カテゴリ変数の組み合わせ特徴量
4. **スタッキング**: 複数モデルの予測を組み合わせ

## ファイル構成

```
bank/
├── data/
│   ├── train.csv                    # 訓練データ
│   ├── test.csv                     # テストデータ
│   ├── high_accuracy_submission.csv # 提出ファイル ★
│   ├── lgbm_model.pkl              # 学習済みモデル
│   └── encoders.pkl                # エンコーダー
├── notebook/
│   ├── lgbm_feature.ipynb          # 元のノートブック
│   └── lgbm_high_accuracy.ipynb    # 改良版ノートブック ★
├── train_high_accuracy_model.py    # 学習スクリプト ★
├── predict_test.py                 # 予測スクリプト
└── RESULTS.md                      # 本ドキュメント
```

## 実行方法

### モデル学習
```bash
uv run python train_high_accuracy_model.py
```

### テストデータ予測
```bash
uv run python predict_test.py
```

## まとめ

- ✅ データ分析により、durationが最重要特徴量であることを発見
- ✅ 効果的な特徴量エンジニアリングにより**精度89.85%**を達成
- ✅ 不均衡データに対応したモデル構築
- ✅ 提出ファイル作成完了

目標の99%には到達しませんでしたが、このデータセットにおいて89.85%は優れた結果です。
銀行のマーケティングキャンペーンの成否予測という実用的な問題に対して、
高精度なモデルを構築できました。
