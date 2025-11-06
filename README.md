# Train DEIMv2

DEIMv2 (Efficient Instance Matching v2) モデルの訓練を簡単に実行するためのトレーニングフレームワークです。

## 📋 目次

- [Train DEIMv2](#train-deimv2)
  - [📋 目次](#-目次)
  - [概要](#概要)
    - [主な機能](#主な機能)
  - [ディレクトリ構成](#ディレクトリ構成)
  - [環境構築](#環境構築)
    - [1. リポジトリのクローン](#1-リポジトリのクローン)
    - [2. 依存パッケージのインストール](#2-依存パッケージのインストール)
  - [データセットの準備](#データセットの準備)
    - [COCOフォーマット](#cocoフォーマット)
    - [アノテーション形式](#アノテーション形式)
  - [モデルの訓練](#モデルの訓練)
    - [基本的な使い方](#基本的な使い方)
    - [設定ファイルの編集](#設定ファイルの編集)
    - [設定ファイルの主要パラメータ](#設定ファイルの主要パラメータ)
    - [ONNXエクスポート](#onnxエクスポート)
  - [利用可能なモデル](#利用可能なモデル)
    - [DEIMv2シリーズ (推奨)](#deimv2シリーズ-推奨)
    - [その他のモデル](#その他のモデル)
  - [出力ファイル](#出力ファイル)
  - [高度な使い方](#高度な使い方)
    - [カスタムデータ拡張](#カスタムデータ拡張)
    - [エポック設定の詳細](#エポック設定の詳細)
    - [GPUの指定](#gpuの指定)
  - [ライセンス](#ライセンス)
  - [参考リンク](#参考リンク)

## 概要

本リポジトリは、DEIMv2オブジェクト検出モデルのカスタムデータセットでの訓練を簡素化するために作成されました。複雑な設定ファイルの管理を自動化し、最小限の設定でモデル訓練を開始できます。

### 主な機能

- **簡易設定**: YAML設定ファイルから自動的に訓練用設定を生成
- **自動クラス数計算**: COCOフォーマットのアノテーションから自動でクラス数を取得
- **モデルサイズ自動調整**: モデルに応じた最適な入力サイズを自動設定
- **ONNXエクスポート**: 訓練後に自動でONNX形式に変換可能
- **エポック最適化**: データ拡張、ミックスアップなどのエポック設定を自動計算

## ディレクトリ構成

```
Train_DEIMv2/
├── train.py                    # メインの訓練スクリプト
├── configs/
│   └── config.yaml            # ユーザー設定ファイル (このファイルを編集)
│   └── train_config.yaml      # 自動生成される訓練用設定ファイル
├── libs/
│   ├── create_train_config.py # 訓練設定生成モジュール
│   └── export_to_onnx.py      # ONNX変換モジュール
├── DEIMv2/                     # DEIMv2のソースコード (サブモジュール)
├── datasets/                   # データセットを配置
│   └── COCOstyle_small/
│       ├── train/             # 訓練画像
│       ├── val/               # 検証画像
│       └── annotations/       # COCOフォーマットのアノテーション
├── outputs/                    # 訓練結果の出力先
└── weight/                     # 事前学習済み重みの配置
```

## 環境構築

### 1. リポジトリのクローン

```bash
git clone https://github.com/NRT-ML/Train_DEIMv2.git
cd Train_DEIMv2
```

### 2. 依存パッケージのインストール

```bash
pip install -r DEIMv2/requirements.txt
pip install onnx onnxsim  # ONNX変換を使用する場合
```

## データセットの準備

### COCOフォーマット

データセットはCOCO形式で準備してください。

```
datasets/
└── your_dataset/
    ├── train/                 # 訓練用画像
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── val/                   # 検証用画像
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── annotations/
        ├── train_annotations.json
        └── val_annotations.json
```

### アノテーション形式

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "class_name"
    }
  ]
}
```

## モデルの訓練

### 基本的な使い方

1. **設定ファイルの編集**: `configs/config.yaml`を編集
2. **訓練の実行**: 以下のコマンドを実行

```bash
python train.py -c configs/config.yaml
```

### 設定ファイルの編集

`configs/config.yaml`を開き、以下の項目を編集します:

```yaml
# モデルの選択
model: deimv2_hgnetv2_n_coco

# 出力ディレクトリ
output_dir: "./outputs/my_experiment"

# 訓練エポック数
epochs: 100

# オプティマイザ設定
optimizer:
  type: AdamW
  lr: 0.0004
  weight_decay: 0.0001

# 訓練データの設定
train_dataloader: 
  total_batch_size: 4
  dataset: 
    img_folder: datasets/your_dataset/train
    ann_file: datasets/your_dataset/annotations/train_annotations.json

# 検証データの設定
val_dataloader: 
  total_batch_size: 4
  dataset:
    img_folder: datasets/your_dataset/val
    ann_file: datasets/your_dataset/annotations/val_annotations.json
```

### 設定ファイルの主要パラメータ

| パラメータ | 説明 | 推奨値 |
|----------|------|--------|
| `model` | 使用するモデル名 | `deimv2_hgnetv2_s_coco`（バランス重視） |
| `epochs` | 訓練エポック数 | 100-200 |
| `optimizer.lr` | 学習率 | 0.0004 (バッチサイズ4の場合) |
| `total_batch_size` | バッチサイズ | GPUメモリに応じて調整 (4-16) |
| `remap_mscoco_category` | MS COCOカテゴリのリマップ | カスタムデータセットの場合は`False` |

### ONNXエクスポート

訓練後に自動的にONNX形式に変換する場合:

```bash
python train.py -c configs/config.yaml --export-onnx
```

この機能により、訓練完了後に最良の重み (`best_stg*.pth`) が自動的にONNX形式 (`best_stg*.onnx`) に変換されます。

## 利用可能なモデル

### DEIMv2シリーズ (推奨)

HGNetv2バックボーンを使用した効率的なモデル:

| モデル名 | パラメータ数 | FLOPs | AP (COCO) | 用途 |
|---------|------------|-------|-----------|------|
| `deimv2_hgnetv2_atto_coco` | 0.5M | 0.8G | 23.8 | 超軽量・エッジデバイス |
| `deimv2_hgnetv2_femto_coco` | 1.0M | 1.7G | 31.0 | 軽量デバイス |
| `deimv2_hgnetv2_pico_coco` | 1.5M | 5.2G | 38.5 | 小型デバイス |
| `deimv2_hgnetv2_n_coco` | 3.6M | 6.8G | 43.0 | バランス重視 |
| `deimv2_hgnetv2_s_coco` | 9.7M | 25.6G | 50.9 | **推奨・高精度** |

DINOv3バックボーンを使用した高性能モデル:

| モデル名 | パラメータ数 | FLOPs | AP (COCO) | 用途 |
|---------|------------|-------|-----------|------|
| `deimv2_dinov3_s_coco` | 9.7M | 25.6G | 50.9 | 高精度 |
| `deimv2_dinov3_m_coco` | 18.1M | 52.2G | 53.0 | より高精度 |
| `deimv2_dinov3_l_coco` | 32.2M | 96.7G | 56.0 | 最高精度 |
| `deimv2_dinov3_x_coco` | 50.3M | 151.6G | 57.8 | 最高精度・大規模 |

### その他のモデル

**DEIM-DFINEシリーズ**: `deim_hgnetv2_n_coco`, `deim_hgnetv2_s_coco`, etc.

**DFINEシリーズ**: `dfine_hgnetv2_n_coco`, `dfine_hgnetv2_s_coco`, etc.

**DEIM-RTDETRv2シリーズ**: `deim_r18vd_coco`, `deim_r50vd_coco`, etc.

詳細は`configs/config.yaml`のコメントを参照してください。

## 出力ファイル

訓練完了後、`output_dir`で指定したディレクトリに以下のファイルが生成されます:

```
outputs/
└── my_experiment/
    ├── best_stg1.pth          # Stage 1の最良モデル
    ├── best_stg1.onnx         # ONNX変換済みモデル (--export-onnx使用時)
    ├── last.pth               # 最終エポックのモデル
    ├── checkpoint00XX.pth     # 定期的なチェックポイント
    ├── log.txt                # 訓練ログ
    ├── eval/                  # 評価結果
    └── summary/               # TensorBoard用サマリー
```

## 高度な使い方

### カスタムデータ拡張

`config.yaml`の`train_dataloader.dataset.transforms.ops`セクションで、データ拡張を細かく制御できます:

```yaml
transforms:
  ops:
    - type: Mosaic              # モザイク拡張
      probability: 1.0
      rotation_range: 10
      scaling_range: [0.5, 1.5]
    - type: RandomHorizontalFlip
    - type: RandomPhotometricDistort
      p: 0.5
```

### エポック設定の詳細

スクリプトは以下のエポック設定を自動計算します:

- **flat_epoch**: 学習率が平坦化するエポック (総エポック × 0.5)
- **no_aug_epoch**: データ拡張を停止するエポック (総エポック × 0.1)
- **mixup_epochs**: Mixup適用期間 ([エポック × 0.04, flat_epoch])

これらは自動設定されますが、必要に応じて`libs/create_train_config.py`で調整可能です。

### GPUの指定

デフォルトではGPU 0が使用されます。別のGPUを使用する場合は、`train.py`内の環境変数を変更:

```python
env["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1を使用
```

または、コマンドラインから:

```bash
$env:CUDA_VISIBLE_DEVICES="1"; python train.py -c configs/config.yaml
```

## ライセンス

このプロジェクトは、DEIMv2のライセンスに準拠します。詳細は`DEIMv2/LICENSE`を参照してください。

## 参考リンク

- [DEIMv2 公式リポジトリ](https://github.com/Intellindust-AI-Lab/DEIMv2)