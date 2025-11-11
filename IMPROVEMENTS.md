# 機械学習モデルの改善内容

## 実装した改善点

### 1. ワンホットエンコーディング (One-Hot Encoding)

**変更前:**
- Target EncodingとLabel Encodingを使用
- カテゴリ変数を順序付き数値に変換

**変更後:**
- `pd.get_dummies()`を使用してワンホットエンコーディングを実装
- カテゴリ変数を複数のバイナリ変数に変換
- 利点:
  - カテゴリ間の順序関係の誤解を防ぐ
  - LightGBMが各カテゴリを独立して扱える
  - より解釈しやすい特徴量表現

**対象変数:**
- `job`, `marital`, `education`, `contact`, `poutcome`, `age_group`
- 相互作用特徴量: `job_education`, `contact_month`

### 2. 交差検証 (Cross-Validation)

**変更前:**
- 単純なtrain/valid分割（80/20）
- 1回の評価のみ

**変更後:**
- **StratifiedKFold 5分割交差検証**を実装
- 利点:
  - データの使用効率が向上
  - モデルの汎化性能をより正確に評価
  - 過学習のリスクを低減
  - 少ないデータでも安定した性能評価が可能

**実装:**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 3. LightGBMの最適化

**変更前:**
- Optunaで100トライアル
- 単一のvalidation setで評価

**変更後:**
- **交差検証ベースのハイパーパラメータ最適化**
- 50トライアルで各トライアルが5-fold CVで評価
- より信頼性の高いパラメータ選択

**最適化パラメータ:**
- `learning_rate`: 0.01 ~ 0.1 (log scale)
- `num_leaves`: 20 ~ 150
- `max_depth`: 3 ~ 12
- `min_child_samples`: 10 ~ 100
- `subsample`: 0.6 ~ 1.0
- `colsample_bytree`: 0.6 ~ 1.0
- `reg_alpha`: 1e-8 ~ 10.0 (log scale)
- `reg_lambda`: 1e-8 ~ 10.0 (log scale)
- `min_split_gain`: 0.0 ~ 1.0

**固定設定:**
- `class_weight="balanced"`: 不均衡データに対応
- `n_estimators=3000` + Early Stopping: 過学習防止
- `boosting_type="gbdt"`: 標準的な勾配ブースティング

### 4. 追加の特徴量エンジニアリング

**新規追加特徴量:**
- `balance_positive`: 残高が正かどうか
- `balance_negative`: 残高が負かどうか
- `duration_log`: duration の対数変換
- `previous_per_pdays`: 接触頻度の効率性

**周期性エンコーディング:**
- `month_sin`, `month_cos`: 月の周期性を考慮

### 5. Out-of-Fold (OOF) 予測

**実装内容:**
- 各フォールドの予測を組み合わせてOOF予測を作成
- テストデータは5つのモデルの平均予測を使用
- 利点:
  - より安定した予測
  - アンサンブル効果で精度向上
  - リーク防止

## 使用方法

### Pythonスクリプト版
```bash
cd /home/user/bank
python src/train_improved_lgbm.py
```

### Jupyter Notebook版
```bash
jupyter notebook notebook/improved_lgbm_onehot.ipynb
```

## 出力ファイル

1. `improved_onehot_cv_submission.csv`: 提出用ファイル（id, y）
2. `improved_onehot_cv_submission_with_proba.csv`: 確率値付きファイル（閾値調整用）

## 期待される改善効果

1. **汎化性能の向上**: 交差検証により過学習を抑制
2. **安定性の向上**: 5つのモデルのアンサンブルによる予測の安定化
3. **解釈性の向上**: ワンホットエンコーディングにより特徴量の意味が明確
4. **少データへの対応**: 交差検証により限られたデータを効率的に活用

## 技術的な工夫

- **Early Stopping**: 100ラウンド改善がなければ学習を停止
- **Stratified Sampling**: ターゲット変数の分布を保ったまま分割
- **閾値最適化**: F1スコアを最大化する閾値を自動探索
- **特徴量の揃え**: Train/Test間でカラムを完全に一致させる処理
