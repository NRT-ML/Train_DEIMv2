# Train DEIMv2

DEIMv2モデルの訓練を簡単に実行するためのトレーニングフレームワークです。

## 📋 目次

- [概要](#概要)
- [ディレクトリ構成](#ディレクトリ構成)
- [環境構築](#環境構築)
- [データセットの準備](#データセットの準備)
- [モデルの訓練](#モデルの訓練)
- [利用可能なモデル](#利用可能なモデル)
- [出力ファイル](#出力ファイル)
- [高度な使い方](#高度な使い方)
- [ライセンス](#ライセンス)
- [参考リンク](#参考リンク)

## 概要

本リポジトリは、DEIMv2 オブジェクト検出モデルをカスタムデータセットで訓練する手間を大幅に削減することを目的に作成されました。難解な設定ファイルの管理を自動化し、最小限の設定で訓練を開始できます。

### 主な機能

- **簡易設定**: YAML 設定から訓練用設定を自動生成
- **自動クラス数計算**: COCO 形式アノテーションからクラス数を自動取得
- **モデル入力サイズ調整**: モデルに応じた最適入力サイズを自動設定
- **ONNX エクスポート**: 訓練後に ONNX 形式へ自動変換可能
- **ファインチューニング**: 学習済みモデルを自動ダウンロードして再訓練
- **バックボーン自動ダウンロード**: DINOv3（S/M）バックボーンを自動取得

## ディレクトリ構成

```text
Train_DEIMv2/
├── requirements.txt            # 依存パッケージ
├── train.py                    # メイン訓練スクリプト
│
├── configs/                    # 設定ファイル群
│   ├── config.yaml             # ユーザー設定ファイル (サンプル1)
│   └── config_no_aug.yaml      # ユーザー設定ファイル (サンプル2)
│
├── libs/                       # 訓練支援モジュール
│
├── DEIMv2/                     # 元の DEIMv2 ソース
│   ├── ckpts/                  # DINOv3 バックボーン配置先（自動DL）
│   └── ...
│
├── datasets/                   # データセット配置ディレクトリ
│   └── your_dataset/         # coco形式
│
├── pretrained/                 # 学習済みモデル配置先（自動DL）
│
├── outputs/                    # 訓練結果出力先
│   └── {experiment_name}/
│
└── weight/                     # hgnetv2 学習済み重み配置先(自動DL)
```

## 環境構築

### 1. リポジトリのクローン

```bash
git clone --recursive https://github.com/NRT-ML/Train_DEIMv2.git
cd Train_DEIMv2
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## データセットの準備

### COCOフォーマット

```text
datasets/
└── your_dataset/
    ├── train/                 # 訓練画像
    ├── val/                   # 検証画像
    └── annotations/
        ├── train_annotations.json
        └── val_annotations.json
```

### アノテーション形式（抜粋）

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

1. `configs/config.yaml` を編集（モデル名やデータセットパスを指定）
2. 以下のコマンドいずれかを実行

```bash
# ランダム初期化での訓練
python train.py -c configs/config.yaml

# 学習済み重みを使ったファインチューニング
python train.py -c configs/config.yaml -t

# 訓練後に ONNX を自動エクスポート
python train.py -c configs/config.yaml -e

# ファインチューニング + ONNX エクスポート
python train.py -c configs/config.yaml -t -e
```

### コマンドライン引数

| 引数 | 短縮形 | 内容 | 必須 |
|------|--------|------|------|
| `--config` | `-c` | 設定ファイル（YAML）のパス | ✓ |
| `--tuning` | `-t` | 学習済みモデルをダウンロードしてファインチューニング |  |
| `--export-onnx` | `-e` | 訓練後に最良モデルを ONNX 形式で保存 |  |

### 自動ダウンロード機能

#### バックボーン

※ hgnetv2はDEIMv2がデフォルトで自動ダウンロード。

| モデル | バックボーン | ファイル名 | 自動DL |
|--------|--------------|-----------|--------|
| `deimv2_dinov3_s_coco` | ViT-Tiny | `vitt_distill.pt` | ✅ |
| `deimv2_dinov3_m_coco` | ViT-Tiny+ | `vittplus_distill.pt` | ✅ |
| `deimv2_dinov3_l_coco` | DINOv3 ViT-S/16 | - | ❌ 対応予定なし |
| `deimv2_dinov3_x_coco` | DINOv3 ViT-S/16+ | - | ❌ 対応予定なし |

ダウンロード先: `DEIMv2/ckpts/`

#### 学習済みモデル（ファインチューニング用）

`-t` オプションを指定すると、モデルに応じた COCO 学習済み重みを Google Drive から自動ダウンロードします。

ダウンロード先: `pretrained/`

### 設定ファイルの編集

```yaml
model: deimv2_hgnetv2_n_coco
output_dir: "./outputs/my_experiment"
epochs: 100

optimizer:
  type: AdamW
  lr: 0.0004
  weight_decay: 0.0001

train_dataloader: 
  total_batch_size: 4
  dataset: 
    img_folder: datasets/your_dataset/train
    ann_file: datasets/your_dataset/annotations/train_annotations.json
    transforms:
      ops:                                          # データ拡張設定(任意。DEIMv2のConfig参照)
        ...

val_dataloader: 
  total_batch_size: 4
  dataset:
    img_folder: datasets/your_dataset/val
    ann_file: datasets/your_dataset/annotations/val_annotations.json
    transforms:
      ops:                                          # データ拡張設定(任意。DEIMv2のConfig参照)
        ...
```

## 利用可能なモデル

### DEIMv2シリーズ (推奨)

#### HGNetv2 バックボーン

| モデル名 | Param | FLOPs | AP | 入力サイズ | 用途 |
|----------|-------|-------|----|------------|------|
| `deimv2_hgnetv2_atto_coco` | 0.5M | 0.8G | 23.8 | 320×320 | 超軽量・エッジ用途 |
| `deimv2_hgnetv2_femto_coco`| 1.0M | 1.7G | 31.0 | 416×416 | 軽量デバイス |
| `deimv2_hgnetv2_pico_coco` | 1.5M | 5.2G | 38.5 | 640×640 | 小型デバイス |
| `deimv2_hgnetv2_n_coco`    | 3.6M | 6.8G | 43.0 | 640×640 | バランス重視 |
| `deimv2_hgnetv2_s_coco`    | 9.7M | 25.6G| 50.9 | 640×640 | 高精度（推奨） |

#### DINOv3 バックボーン

| モデル名 | バックボーン | Param | 入力サイズ | 対応状況 |
|----------|-------------|-------|------------|----------|
| `deimv2_dinov3_s_coco` | ViT-Tiny (蒸留版) | 9.7M | 640×640 | ✅ 対応 |
| `deimv2_dinov3_m_coco` | ViT-Tiny+ (蒸留版) | 18.1M | 640×640 | ✅ 対応 |

### ファインチューニング対応モデル

| モデル名 | 自動ダウンロード | 備考 |
|----------|-----------------|------|
| `deimv2_hgnetv2_atto_coco` | ✅ | HGNetv2 系 |
| `deimv2_hgnetv2_femto_coco`| ✅ | 〃 |
| `deimv2_hgnetv2_pico_coco` | ✅ | 〃 |
| `deimv2_hgnetv2_n_coco`    | ✅ | 〃 |
| `deimv2_dinov3_s_coco`     | ✅ | DINOv3 S |
| `deimv2_dinov3_m_coco`     | ✅ | DINOv3 M |
| `deimv2_dinov3_l_coco`     | ❌ | サポート対象外 |
| `deimv2_dinov3_x_coco`     | ❌ | サポート対象外 |

## 出力ファイル

```text
outputs/
└── my_experiment/
    ├── best_stg*.pth          # Stage1 最良モデル
    ├── best_stg*.onnx         # ONNX （-e 指定時）
    ├── last.pth               # 最終エポック
    ├── checkpoint000X.pth     # 定期保存
    ├── log.txt                # 訓練ログ
    ├── eval/                  # 評価結果
    └── summary/               # TensorBoard ログ
```

## 高度な使い方

### カスタムデータ拡張

`config.yaml`の`train_dataloader.dataset.transforms.ops`セクションで、データ拡張を細かく制御できます:

```yaml
transforms:
  ops: # Sample
    - {type: Resize}  # Resizeのsize, Mosaicのoutput_sizeはimage_sizeから自動設定
    - {type: ConvertPILImage, dtype: 'float32', scale: true}
    - {type: ConvertBoxes, fmt: 'cxcywh', normalize: true}
```

### エポック設定の詳細

`libs/create_train_config.py` にて以下を自動計算しています。

- `flat_epoch`: 総エポックの 50%
- `no_aug_epoch`: 総エポックの 10%
- `mixup_epochs`: 総エポックの 4% 〜 50%

必要に応じてコード側で調整ください。

## ライセンス

このプロジェクトは DEIMv2 のライセンスに準拠します。詳細は `DEIMv2/LICENSE` を参照してください。

## 参考リンク

- [DEIMv2 公式リポジトリ](https://github.com/Intellindust-AI-Lab/DEIMv2)
