# DEIMv2 訓練スクリプト

このディレクトリには、DEIMv2モデルを訓練するための`train.py`スクリプトが含まれています。

## 📋 概要

`train.py`は、以下の特徴を持つDEIMv2訓練スクリプトです:

- ✅ DEIMv2の本来の訓練フローに従った実装
- ✅ DEIMKitのTrainerクラスを参考にした設計
- ✅ `configs/config.yaml`から設定を読み込み
- ✅ 単一GPU/マルチGPU訓練対応
- ✅ チェックポイントからの再開・ファインチューニング対応
- ✅ 自動混合精度(AMP)サポート
- ✅ モジュール化された構造で保守性が高い

## 🧩 モジュール構成

### `train.py`
メインスクリプト。コマンドライン引数の解析、訓練の実行、DEIMv2環境の初期化を担当します。

### `libs/config_loader.py`
YAML設定ファイルの読み込みを担当するモジュールです。

**主な関数:**
- `load_config_from_yaml(config_path: str) -> dict`: YAMLファイルを読み込んで辞書として返します

### `libs/deimv2_config.py`
DEIMv2のYAMLConfig互換オブジェクトを作成するモジュールです。ユーザー設定を元に、以下を自動設定します:

**主な関数:**
- `create_deimv2_config(user_config: dict, args: argparse.Namespace) -> YAMLConfig`: DEIMv2設定オブジェクトを作成

**自動設定される項目:**
- データセットパス、画像サイズ、バッチサイズ
- `no_aug_epoch` (総エポックの13%)
- `flat_epoch` (総エポックの50%)
- `warmup_iter` (総イテレーションの5%)
- `ema_warmups` (総イテレーションの5%)
- `mixup_epochs` ([4%, 50%]のエポック)
- `stop_epoch` (総エポックの90%)
- `data_aug_epochs` ([4%, 50%, 90%]のエポック)
- `matcher_change_epoch` (総エポックの90%)

## 📁 ディレクトリ構造

このスクリプトを使用する前に、以下のディレクトリ構造になっていることを確認してください：

```
train_deim/
├── configs/
│   └── config.yaml          # 訓練設定ファイル
├── deim/
│   ├── train.py             # メイン訓練スクリプト
│   ├── README.md            # このファイル
│   ├── libs/                # サポートモジュール
│   │   ├── __init__.py      # パッケージ初期化
│   │   ├── config_loader.py # YAML設定読み込み
│   │   └── deimv2_config.py # DEIMv2設定作成
│   └── DEIMv2/              # DEIMv2リポジトリをここに配置
│       ├── configs/
│       │   ├── deimv2/
│       │   ├── deim_dfine/
│       │   └── deim_rtdetrv2/
│       ├── engine/
│       ├── train.py
│       └── ...
└── outputs/                 # 訓練結果の出力先
```

**重要**: DEIMv2リポジトリを`deim/DEIMv2`に配置してください。

```bash
# DEIMv2のクローン例
cd deim
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git
```

## �🚀 使い方

### 1. 設定ファイルの準備

`configs/config.yaml`を編集して、データセットパスやハイパーパラメータを設定します。

```yaml
# モデル選択
model: deimv2_hgnetv2_s_coco

# データセットパス
train_ann_file: "dataset/train/_annotations.coco.json"
train_img_folder: "dataset/train"
val_ann_file: "dataset/val/_annotations.coco.json"
val_img_folder: "dataset/val"

# 画像サイズ
image_size: [640, 640]

# バッチサイズ
train_batch_size: 16
val_batch_size: 8

# クラス数（背景クラスを含む）
num_classes: 81  # COCOの場合: 80 + 1

# その他の設定...
```

### 2. 訓練の実行

#### 単一GPU訓練

```bash
python deim/train.py
```

#### マルチGPU訓練（4 GPU の例）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 deim/train.py
```

#### カスタム設定ファイルを使用

```bash
python deim/train.py --config path/to/custom_config.yaml
```

#### AMP（自動混合精度）を有効化

```bash
python deim/train.py --use-amp
```

### 3. チェックポイントからの再開

訓練を中断した場合、チェックポイントから再開できます:

```bash
python deim/train.py --resume outputs/checkpoint0099.pth
```

### 4. ファインチューニング

事前学習済みモデルからファインチューニング:

```bash
python deim/train.py --tuning pretrained/model.pth
```

### 5. 評価のみ

訓練せずに評価のみ実行:

```bash
python deim/train.py --test-only --resume outputs/best.pth
```

## 📖 コマンドライン引数

```
python deim/train.py --help
```

主要な引数:

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `-c, --config` | 設定ファイルのパス | `configs/config.yaml` |
| `-r, --resume` | チェックポイントから再開 | None |
| `-t, --tuning` | ファインチューニング | None |
| `-d, --device` | 使用デバイス | None (自動検出) |
| `--seed` | 乱数シード | 0 |
| `--use-amp` | AMP有効化 | False |
| `--output-dir` | 出力ディレクトリ | None |
| `--test-only` | 評価のみ | False |

## 🎯 利用可能なモデル

DEIMv2リポジトリには、複数のモデルシリーズが含まれています:

### DEIMv2シリーズ（DINOv3/HGNetv2バックボーン）

最新のDEIMv2モデル群。優れた性能と効率のバランス。

| モデル名 | サイズ | パラメータ数 | GFLOPs | COCO AP |
|---------|--------|------------|--------|---------|
| `deimv2_hgnetv2_atto_coco` | Atto | 0.5M | 0.8 | 23.8 |
| `deimv2_hgnetv2_femto_coco` | Femto | 1.0M | 1.7 | 31.0 |
| `deimv2_hgnetv2_pico_coco` | Pico | 1.5M | 5.2 | 38.5 |
| `deimv2_hgnetv2_n_coco` | N | 3.6M | 6.8 | 43.0 |
| `deimv2_hgnetv2_s_coco` | **S** | **9.7M** | **25.6** | **50.9** ⭐ |
| `deimv2_dinov3_s_coco` | S+ | 9.7M | 25.6 | 50.9 |
| `deimv2_dinov3_m_coco` | M | 18.1M | 52.2 | 53.0 |
| `deimv2_dinov3_l_coco` | L | 32.2M | 96.7 | 56.0 |
| `deimv2_dinov3_x_coco` | X | 50.3M | 151.6 | 57.8 |

### DEIM-DFINEシリーズ（HGNetv2バックボーン）

DEIMとD-FINEを組み合わせたモデル群。

| モデル名 | サイズ | パラメータ数 | GFLOPs |
|---------|--------|------------|--------|
| `deim_hgnetv2_n_coco` | N | 3.1M | 5.9 |
| `deim_hgnetv2_s_coco` | S | 9.7M | 25.7 |
| `deim_hgnetv2_m_coco` | M | 18.8M | 56.6 |
| `deim_hgnetv2_l_coco` | L | 32.8M | 107.3 |
| `deim_hgnetv2_x_coco` | X | 57.8M | 202.9 |

### DFINEシリーズ（HGNetv2バックボーン）

オリジナルのD-FINEモデル群。

| モデル名 | サイズ | パラメータ数 | GFLOPs |
|---------|--------|------------|--------|
| `dfine_hgnetv2_n_coco` | N | 3.0M | 5.8 |
| `dfine_hgnetv2_s_coco` | S | 9.6M | 25.6 |
| `dfine_hgnetv2_m_coco` | M | 18.7M | 56.5 |
| `dfine_hgnetv2_l_coco` | L | 32.7M | 107.2 |
| `dfine_hgnetv2_x_coco` | X | 57.7M | 202.8 |

### DEIM-RTDETRv2シリーズ（ResNet/Swinバックボーン）

DEIMとRT-DETRv2を組み合わせたモデル群。

| モデル名 | バックボーン | 説明 |
|---------|------------|------|
| `deim_r18vd_coco` | ResNet-18 | 軽量版 |
| `deim_r34vd_coco` | ResNet-34 | 中型版 |
| `deim_r50vd_coco` | ResNet-50 | 標準版 |
| `deim_r101vd_coco` | ResNet-101 | 大型版 |
| `deim_swin_t_coco` | Swin-Tiny | Transformer軽量版 |
| `deim_swin_s_coco` | Swin-Small | Transformer標準版 |

⭐ **推奨**: 最初は`deimv2_hgnetv2_s_coco`から始めることをお勧めします（バランスの良い性能）

## 📁 データセット形式

データセットはCOCO形式で準備する必要があります:

```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
└── val/
    ├── image1.jpg
    ├── image2.jpg
    └── _annotations.coco.json
```

### カスタムデータセットの注意点

1. **クラスインデックス**: 0から始める必要があります
2. **`num_classes`**: 実際のクラス数 + 1（背景クラス）
3. **`remap_mscoco`**: カスタムデータセットの場合は`false`に設定

## 🔧 カスタマイズ

### バッチサイズの変更

GPUメモリに応じてバッチサイズを調整:

- 単一GPU (16GB VRAM): `train_batch_size: 8-16`
- 単一GPU (24GB VRAM): `train_batch_size: 16-32`
- マルチGPU (4x16GB): `train_batch_size: 64-128`

### 入力サイズの変更

```yaml
image_size: [320, 320]  # より小さい画像サイズ
```

Mosaicの出力サイズは自動的に `input_size / 2` に設定されます。

### エポック数の調整

```yaml
epochs: 50  # 50エポック訓練
```

**自動設定されるパラメータ:**

`epochs`を設定すると、以下のパラメータが自動的に計算されます（DEIMKit/src/trainer.pyのロジックを参考）:

| パラメータ | 計算式 | epochs=50の場合 | 説明 |
|-----------|--------|----------------|------|
| `no_aug_epoch` | `epochs × 0.13` | 6 | データ拡張なしのエポック数 |
| `flat_epoch` | `epochs × 0.5` | 25 | 学習率フラットのエポック数 |
| `warmup_iter` | `(total_iters × 0.05)` | 自動計算 | 学習率ウォームアップのイテレーション数 |
| `ema_warmups` | `(total_iters × 0.05)` | 自動計算 | EMAウォームアップのイテレーション数 |
| `mixup_epochs` | `[epochs × 0.04, epochs × 0.5]` | [2, 25] | Mixup拡張の開始・終了エポック |
| `stop_epoch` | `epochs × 0.9` | 45 | マルチスケール訓練の停止エポック |
| `data_aug_epochs` | `[epochs × 0.04, × 0.5, × 0.9]` | [2, 25, 45] | データ拡張の段階的縮小エポック |
| `matcher_change_epoch` | `epochs × 0.9` | 45 | マッチャー変更エポック |

**手動で設定したい場合:**

```yaml
epochs: 50
no_aug_epoch: 10      # 手動で設定（自動計算を上書き）
flat_epoch: 30        # 手動で設定
warmup_iter: 1000     # 手動で設定
```

手動で設定した値は、自動計算された値よりも優先されます。

## 📊 出力

訓練結果は`output_dir`に保存されます:

```
outputs/deimv2_hgnetv2_s_custom/
├── best.pth              # ベストモデル
├── checkpoint0099.pth    # 定期チェックポイント
├── checkpoint_final.pth  # 最終チェックポイント
├── config.yml            # 使用した設定
└── tensorboard/          # TensorBoardログ
```

### TensorBoardで進捗確認

```bash
tensorboard --logdir outputs/deimv2_hgnetv2_s_custom
```

ブラウザで http://localhost:6006/ にアクセス

## 🐛 トラブルシューティング

### CUDA Out of Memory

バッチサイズを減らしてください:

```yaml
train_batch_size: 8  # または 4
```

### インポートエラー

DEIMv2ディレクトリが正しい位置にあることを確認:

```
deim/
├── train.py
└── DEIMv2/          # ここに DEIMv2 リポジトリ
    ├── engine/
    ├── configs/
    └── ...
```

配置されていない場合:

```bash
cd deim
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git
```

### 分散訓練が動作しない

torchrunを使用していることを確認:

```bash
torchrun --master_port=7777 --nproc_per_node=4 deim/train.py
```

## 📚 参考資料

- **DEIMv2 公式リポジトリ**: https://github.com/Intellindust-AI-Lab/DEIMv2
- **DEIMKit**: https://github.com/dnth/DEIMKit
- **論文**: https://arxiv.org/abs/2509.20787

## 💡 ヒント

1. **最初は小さいモデルで試す**: `deimv2_hgnetv2_n_coco`や`deimv2_hgnetv2_s_coco`から始めることをお勧めします
2. **AMPを使う**: `--use-amp`フラグでメモリ使用量を削減し、訓練を高速化
3. **学習率の調整**: バッチサイズを変更した場合、学習率も線形にスケーリングします
4. **データ拡張**: カスタムデータセットでは、データ拡張パラメータの調整が重要です

## ⚠️ 注意事項

- 訓練前に`configs/config.yaml`の全ての必須パラメータを設定してください
- COCOデータセットでない場合、`remap_mscoco: false`に設定してください
- マルチGPU訓練時は、総バッチサイズが全GPUで分割されます

---

ご質問や問題がありましたら、Issueを作成してください!
