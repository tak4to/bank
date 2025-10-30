# smaple_dev

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
