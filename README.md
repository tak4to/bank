# Bank Marketing Machine Learning Project

銀行マーケティングキャンペーンの成約予測プロジェクト

---

## プロジェクト概要

このプロジェクトでは、銀行の電話マーケティングキャンペーンで顧客が定期預金に加入するかを予測します。

## 使用方法

### 方法1: 特徴量選択版（推奨）

```bash
# 特徴量を分析して選択
python src/feature_analysis.py

# 選択された特徴量で学習
python src/train_with_feature_selection.py
```

### 方法2: 全特徴量版

```bash
python src/train_improved_lgbm.py
```

### 方法3: Jupyter Notebook

```bash
jupyter notebook notebook/improved_lgbm_onehot.ipynb
```

## 実装した改善点

1. **ワンホットエンコーディング**: カテゴリ特徴量を適切に変換
2. **スタージェンの公式**: 年齢グループを最適化（16グループ）
3. **交差検証**: StratifiedKFold 5分割で汎化性能向上
4. **特徴量選択**: 相関分析・重要度・VIFで根拠のある選択
5. **ハイパーパラメータ最適化**: Optunaで最適化

詳細は以下を参照:
- [IMPROVEMENTS.md](IMPROVEMENTS.md): 改善内容の詳細
- [FEATURE_SELECTION.md](FEATURE_SELECTION.md): 特徴量選択の根拠

## ディレクトリ構成

```
bank/
├── data/                  # データディレクトリ
├── src/                   # ソースコード
│   ├── feature_analysis.py           # 特徴量分析
│   ├── train_with_feature_selection.py  # 特徴量選択版
│   └── train_improved_lgbm.py        # 全特徴量版
├── notebook/              # Jupyter Notebooks
└── FEATURE_SELECTION.md   # 特徴量選択の詳細
```

---

# 環境構築

# uvをインストール
uvを使ってPythonのパッケージの管理を行う。<br>
[uvのドキュメント](https://docs.astral.sh/uv/)
## uvの環境セットアップ
```
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uvでPythonのバージョンを管理する。以下のコマンドでPythonを任意のバージョンで管理する。
```
uv python install 3.10 3.11 3.12
```

インストール済みのPythonのバージョンは以下で確認することができる
```
uv python list
```

# uvを使った環境構築
以下のコマンドで、pyproject.tomlを使って仮想環境を構築する。
```
uv sync
```

初期化してやる場合は、以下のコマンドを実行する。
```
uv init hogehoge
```
初期化した後にuv sync をすると、仮想環境を作ることできる。

## 依存関係の管理
以下のコマンドでpyproject.tomlに依存関係を追加できます。
```
uv add pandas
```
--devオプションで、開発用の依存関係を追加できます。
```
uv add --dev pytest
```
以下のコマンドで依存関係を削除することできる。
```
uv remove pandas
```
