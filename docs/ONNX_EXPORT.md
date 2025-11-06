# ONNX変換ガイド

訓練済みDEIMv2モデルをONNX形式に変換する方法を説明します。

## 目次

- [概要](#概要)
- [重要な注意事項](#重要な注意事項)
- [使用方法](#使用方法)
  - [方法1: 訓練後に自動変換](#方法1-訓練後に自動変換)
  - [方法2: 訓練済みモデルを後で変換](#方法2-訓練済みモデルを後で変換)
- [設定について](#設定について)
- [オプションパラメータ](#オプションパラメータ)
- [出力ファイル](#出力ファイル)
- [トラブルシューティング](#トラブルシューティング)

## 概要

DEIMv2モデルをONNX形式に変換することで、以下のメリットが得られます:

- **デプロイの簡易化**: ONNXランタイムやONNX対応フレームワークで実行可能
- **パフォーマンス最適化**: 推論時の高速化とメモリ効率の向上
- **プラットフォーム非依存**: 様々な環境（エッジデバイス、サーバー等）で動作
- **言語非依存**: Python以外の言語からも利用可能

## 重要な注意事項

⚠️ **訓練時の設定の保存**

ONNX変換では、**訓練時と完全に同じ設定**を使用する必要があります。訓練完了時、以下のファイルが自動的に保存されます:

```
outputs/
  └── your_model/
      ├── best_stg1.pth              # ベストモデル
      └── best_stg1_config.yaml      # 訓練時の最終設定（自動保存）
```

`best_stg1_config.yaml`には、訓練中に構築・調整された最終的な設定が含まれています:
- 画像サイズ
- バッチサイズ
- クラス数
- データセットパス
- エポック数
- 自動計算されたパラメータ（no_aug_epoch、warmup_iter等）
- その他すべての調整済み設定

ONNX変換時、このファイルが自動的に使用されるため、通常は`--config`オプションを指定する必要はありません。

## 使用方法

### 方法1: 訓練後に自動変換

訓練完了後、自動的にONNX形式に変換します。

```bash
python deim/train.py \
    --config configs/config.yaml \
    --export-onnx
```

このコマンドは以下を実行します:
1. モデルを訓練
2. 訓練完了時、最終設定を`best_stg1_config.yaml`として保存
3. 保存された設定を使用してONNX変換

### 方法2: 訓練済みモデルを後で変換

既に訓練済みのモデルをONNX形式に変換します。

```bash
# 基本的な変換（設定は自動検出）
python deim/train.py \
    --export-onnx \
    --resume outputs/deimv2_hgnetv2_atto_custom/best_stg1.pth \
    --test-only
```

訓練時に保存された`best_stg1_config.yaml`が自動的に使用されます。

#### 設定ファイルを明示的に指定する場合

```bash
python deim/train.py \
    --export-onnx \
    --resume outputs/deimv2_hgnetv2_atto_custom/best_stg1.pth \
    --config outputs/deimv2_hgnetv2_atto_custom/best_stg1_config.yaml \
    --test-only
```

⚠️ **注意**: 元の`configs/config.yaml`ではなく、訓練時に保存された`best_stg1_config.yaml`を使用してください。

## 設定について

### 設定ファイルの自動検出

ONNX変換時、以下の優先順位で設定ファイルを検索します:

1. `--config`オプションで指定されたパス
2. チェックポイントと同じディレクトリの`<checkpoint名>_config.yaml`
   - 例: `best_stg1.pth` → `best_stg1_config.yaml`
3. チェックポイントと同じディレクトリの`config.yaml`

### 訓練時の設定保存の仕組み

訓練完了時、`libs/checkpoint_utils.py`の`save_training_config()`関数が自動的に呼び出され、以下の処理が行われます:

1. 訓練中に構築された最終的な`cfg.yaml_cfg`を取得
2. YAML形式で`<checkpoint名>_config.yaml`として保存
3. ONNX変換時に自動的に使用可能になる

```python
# train.pyでの自動保存（ユーザー操作不要）
save_training_config(best_model_path, cfg.yaml_cfg)
```

### 元の設定ファイルとの違い

訓練時、`configs/config.yaml`の設定は、`libs/deimv2_config.py`の`create_deimv2_config()`関数によって大幅に書き換えられます:

**元の設定ファイル（configs/config.yaml）**:
```yaml
image_size: 640
batch_size: 2
num_classes: 5
epochs: 50
```

**訓練時に構築される最終設定（保存されるconfig.yaml）**:
```yaml
# 画像サイズとトランスフォーム設定が追加
eval_spatial_size: [640, 640]
transform:
  Resize:
    size: [640, 640]
  # ... 多数の設定が自動追加

# バッチサイズ
train_dataloader:
  batch_size: 2
val_dataloader:
  batch_size: 2

# クラス数
num_classes: 5

# エポック数と自動計算パラメータ
epoches: 50
no_aug_epoch: 5
flat_epoch: 10
warmup_iter: 100
# ... その他多数

# データセットパス
train_dataloader:
  dataset:
    ann_file: datasets/COCOstyle_small/annotations/train_annotations_fixed.json
    img_folder: datasets/COCOstyle_small/train
val_dataloader:
  dataset:
    ann_file: datasets/COCOstyle_small/annotations/val_annotations_fixed.json
    img_folder: datasets/COCOstyle_small/val

# オプティマイザ設定
optimizer:
  lr: 0.0001
  weight_decay: 0.0001

# その他数十項目の設定...
```

このため、ONNX変換には**保存された最終設定**を使用する必要があります。

## オプションパラメータ

### 基本オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--export-onnx` | False | ONNX変換を有効化 |
| `--resume` | None | 変換するチェックポイントのパス |
| `--config` | None | 設定ファイル（Noneの場合は自動検出） |
| `--test-only` | False | 訓練をスキップして変換のみ実行 |

### ONNX詳細オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--onnx-output` | None | 出力ファイル名（Noneの場合は自動生成） |
| `--onnx-opset` | 17 | ONNXのopsetバージョン |
| `--onnx-check` | True | エクスポート後にモデルを検証 |
| `--onnx-simplify` | True | onnx-simplifierで最適化 |
| `--onnx-batch-size` | 32 | エクスポート時のダミーバッチサイズ |

### 使用例

```bash
# バッチサイズ16でONNX変換
python deim/train.py \
    --export-onnx \
    --resume outputs/best_stg1.pth \
    --test-only \
    --onnx-batch-size 16

# 最適化なしで変換
python deim/train.py \
    --export-onnx \
    --resume outputs/best_stg1.pth \
    --test-only \
    --onnx-simplify False

# カスタム出力パス
python deim/train.py \
    --export-onnx \
    --resume outputs/best_stg1.pth \
    --test-only \
    --onnx-output my_custom_model.onnx
```

## 出力ファイル

### ファイル構成

訓練とONNX変換を実行すると、以下のファイルが生成されます:

```
outputs/
  └── deimv2_hgnetv2_atto_custom/
      ├── best_stg1.pth              # PyTorchチェックポイント
      ├── best_stg1_config.yaml      # 訓練時の最終設定（自動保存）
      ├── best_stg1.onnx             # ONNXモデル
      ├── last.pth                   # 最後のエポックの重み
      ├── checkpoint0004.pth         # 定期保存チェックポイント
      ├── ...
      ├── log.txt                    # 訓練ログ
      └── summary/                   # TensorBoard用ログ
```

### ONNXモデルの構造

エクスポートされたONNXモデルは以下の入出力を持ちます:

**入力**:
- `images`: `[batch_size, 3, height, width]` (NCHW形式)
- データ型: `float32`
- 動的バッチサイズ対応

**出力**:
- `labels`: クラスラベル `[batch_size, num_detections]`
- `boxes`: バウンディングボックス `[batch_size, num_detections, 4]` (XYXY形式)
- `scores`: 信頼度スコア `[batch_size, num_detections]`

## トラブルシューティング

### 問題1: 設定ファイルが見つからない

**エラーメッセージ**:
```
設定ファイルが見つかりません。config_pathを明示的に指定してください。
```

**解決方法**:
1. 訓練時に`--export-onnx`オプションを使用していたか確認
2. `best_stg1_config.yaml`が存在するか確認
3. 存在しない場合は、`--config`で明示的に指定:
   ```bash
   python deim/train.py \
       --export-onnx \
       --resume outputs/best_stg1.pth \
       --config outputs/best_stg1_config.yaml \
       --test-only
   ```

### 問題2: モデル構造の不一致

**エラーメッセージ**:
```
RuntimeError: Error(s) in loading state_dict for DEIM:
    size mismatch for ...
```

**原因**: 設定ファイルが訓練時と異なる（クラス数、画像サイズ等が不一致）

**解決方法**:
- 必ず訓練時に保存された`best_stg1_config.yaml`を使用
- 元の`configs/config.yaml`は使用しない

### 問題3: ONNXパッケージが見つからない

**エラーメッセージ**:
```
ModuleNotFoundError: No module named 'onnx'
```

**解決方法**:
```bash
pip install onnx onnx-simplifier
```

### 問題4: メモリ不足

**エラーメッセージ**:
```
RuntimeError: CUDA out of memory
```

**解決方法**:
- バッチサイズを小さくする:
  ```bash
  python deim/train.py --export-onnx --onnx-batch-size 8 ...
  ```
- CPUで変換する（環境変数設定）:
  ```bash
  set CUDA_VISIBLE_DEVICES=-1
  python deim/train.py --export-onnx ...
  ```

### 問題5: 訓練時の設定を確認したい

**方法**:
保存された設定ファイルを開いて確認:
```bash
cat outputs/deimv2_hgnetv2_atto_custom/best_stg1_config.yaml
```

主要な設定項目:
- `eval_spatial_size`: 画像サイズ
- `num_classes`: クラス数
- `epoches`: エポック数
- `train_dataloader.dataset`: 訓練データセット設定
- `val_dataloader.dataset`: 検証データセット設定

## 高度な使用方法

### カスタムエクスポート

`libs/onnx_export.py`の`export_to_onnx()`関数を直接使用することもできます:

```python
from libs import export_to_onnx

onnx_path = export_to_onnx(
    checkpoint_path='outputs/best_stg1.pth',
    config_path=None,  # 自動検出
    output_path='my_model.onnx',
    opset_version=17,
    check_model=True,
    simplify=True,
    batch_size=32
)
```

### ONNXモデルの検証

```python
import onnx

# モデルを読み込む
model = onnx.load('outputs/best_stg1.onnx')

# 構造を検証
onnx.checker.check_model(model)

# モデル情報を表示
print(onnx.helper.printable_graph(model.graph))
```

### ONNXランタイムでの推論

```python
import onnxruntime as ort
import numpy as np

# セッションを作成
session = ort.InferenceSession('outputs/best_stg1.onnx')

# ダミー入力を作成
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# 推論を実行
outputs = session.run(None, {'images': dummy_input})
labels, boxes, scores = outputs

print(f"検出されたオブジェクト数: {labels.shape[1]}")
```

## 参考資料

- [ONNX公式ドキュメント](https://onnx.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

## まとめ

✅ 訓練完了時、最終設定が自動的に保存される  
✅ ONNX変換時、設定が自動検出される  
✅ 訓練時と完全に同じ設定が使用される  
✅ 手動での設定管理は不要  

これにより、訓練とONNX変換の間で設定の不一致が発生するリスクが大幅に減少します。
