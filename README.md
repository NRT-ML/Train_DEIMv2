# Train_DEIMv2

DEIMv2モデルの訓練プロジェクト

## 機能

- DEIMv2モデルの訓練
- カスタムデータセットのサポート
- マルチGPU訓練
- チェックポイントからの再開・ファインチューニング
- **訓練済みモデルのONNXエクスポート** ✨ New!

## 基本的な使用方法

### 訓練

```bash
python deim/train.py --config configs/config_.yaml
```

### 訓練後にONNXエクスポート

訓練時に`--export-onnx`を指定すると、訓練完了後に自動的にONNXモデルに変換されます。
設定ファイルも自動的に保存されるため、後で再変換する際も安全です。

```bash
python deim/train.py --config configs/config_.yaml --export-onnx
```

### 既存モデルをONNXに変換

既存のチェックポイントからONNX変換のみを実行する場合。
設定ファイルが自動保存されている場合は指定不要です：

```bash
python deim/train.py --export-onnx --resume outputs/deimv2_hgnetv2_atto_custom/best_stg1.pth --test-only
```

**注意**: 設定ファイルは訓練時に使用したものと同じである必要があります。

## ドキュメント

- [ONNX エクスポート機能の詳細](docs/ONNX_EXPORT.md)

## ディレクトリ構造

```
Train_DEIMv2/
├── configs/           # 設定ファイル
├── datasets/          # データセット
├── deim/              # 訓練スクリプト
├── outputs/           # 訓練結果・チェックポイント
├── weight/            # 事前学習済み重み
└── docs/              # ドキュメント
```
