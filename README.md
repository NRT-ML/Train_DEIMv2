English | [日本語](/README_jp.md)

# Train DEIMv2

A training framework that makes it easy to run training for the [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) model.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Available Models](#available-models)
- [Output Files](#output-files)
- [Advanced Usage](#advanced-usage)
- [License](#license)
- [References](#references)

## Overview

This repository was created to drastically reduce the effort needed to train the DEIMv2 object detection model on your custom dataset. It automates the management of complex configuration files so you can start training with minimal setup.

### Key Features

- **Simple setup**: Automatically generates the training configuration from a YAML definition
- **Automatic class counting**: Reads the number of classes from COCO-format annotations
- **Input size adjustment**: Sets the optimal input size for each model
- **ONNX export**: Optionally converts the trained model to ONNX right after training
- **Fine-tuning**: Downloads pretrained weights and retrains them automatically
- **Automatic backbone download**: Fetches DINOv3 (S/M) backbones as needed

## Directory Structure

```text
Train_DEIMv2/
├── requirements.txt            # Dependency list
├── train.py                    # Main training script
│
├── configs/                    # Configuration files
│   ├── config.yaml             # Sample user configuration (with augmentation)
│   └── config_no_aug.yaml      # Sample user configuration (no augmentation)
│
├── libs/                       # Helper modules for training
│
├── DEIMv2/                     # Original DEIMv2 source
│   ├── ckpts/                  # DINOv3 backbones (downloaded automatically)
│   └── ...
│
├── datasets/                   # Dataset directory
│   └── your_dataset/           # COCO format dataset
│
├── pretrained/                 # Pretrained weights (downloaded automatically)
│
├── outputs/                    # Training outputs
│   └── {experiment_name}/
│
└── weight/                     # Pretrained hgnetv2 weights (downloaded automatically)
```

## Environment Setup

### 1. Clone the repository

```bash
git clone --recursive https://github.com/NRT-ML/Train_DEIMv2.git
cd Train_DEIMv2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### COCO format

```text
datasets/
└── your_dataset/
    ├── train/                 # Training images
    ├── val/                   # Validation images
    └── annotations/
        ├── train_annotations.json
        └── val_annotations.json
```

### Annotation format (excerpt)

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

## Training the Model

### Basic workflow

1. Edit `configs/config.yaml` (set the model name, dataset paths, etc.)
2. Run one of the commands below

```bash
# Train from random initialization
python train.py -c configs/config.yaml

# Fine-tune with pretrained weights
python train.py -c configs/config.yaml -t

# Export to ONNX after training
python train.py -c configs/config.yaml -e

# Fine-tuning + ONNX export
python train.py -c configs/config.yaml -t -e
```

### Command-line arguments

| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--config` | `-c` | Path to the YAML configuration file | ✓ |
| `--tuning` | `-t` | Download pretrained weights and fine-tune |  |
| `--export-onnx` | `-e` | Export the best model to ONNX after training |  |

### Automatic downloads

#### Backbones

Note: hgnetv2 weights are downloaded automatically by DEIMv2 by default.

| Model | Backbone | File name | Auto download |
|-------|----------|-----------|----------------|
| `deimv2_dinov3_s_coco` | ViT-Tiny | `vitt_distill.pt` | ✅ |
| `deimv2_dinov3_m_coco` | ViT-Tiny+ | `vittplus_distill.pt` | ✅ |
| `deimv2_dinov3_l_coco` | DINOv3 ViT-S/16 | - | ❌ Not supported |
| `deimv2_dinov3_x_coco` | DINOv3 ViT-S/16+ | - | ❌ Not supported |

Download destination: `DEIMv2/ckpts/`

#### Pretrained models (for fine-tuning)

When the `-t` option is provided, the appropriate COCO-pretrained weights are downloaded from Google Drive for the selected model.

Download destination: `pretrained/`

### Editing the configuration file

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
      ops:                                          # Optional data augmentation; see DEIMv2 configs
        ...

val_dataloader:
  total_batch_size: 4
  dataset:
    img_folder: datasets/your_dataset/val
    ann_file: datasets/your_dataset/annotations/val_annotations.json
    transforms:
      ops:                                          # Optional data augmentation; see DEIMv2 configs
        ...
```

## Available Models

### DEIMv2 series (recommended)

#### HGNetv2 backbone

| Model name | Param | FLOPs | AP | Input size | Use case |
|------------|-------|-------|----|------------|----------|
| `deimv2_hgnetv2_atto_coco` | 0.5M | 0.8G | 23.8 | 320×320 | Ultra lightweight, edge devices |
| `deimv2_hgnetv2_femto_coco` | 1.0M | 1.7G | 31.0 | 416×416 | Lightweight devices |
| `deimv2_hgnetv2_pico_coco` | 1.5M | 5.2G | 38.5 | 640×640 | Compact devices |
| `deimv2_hgnetv2_n_coco` | 3.6M | 6.8G | 43.0 | 640×640 | Balanced choice |
| `deimv2_hgnetv2_s_coco` | 9.7M | 25.6G | 50.9 | 640×640 | High accuracy (recommended) |

#### DINOv3 backbone

| Model name | Backbone | Param | Input size | Status |
|------------|----------|-------|------------|--------|
| `deimv2_dinov3_s_coco` | ViT-Tiny (distilled) | 9.7M | 640×640 | ✅ Supported |
| `deimv2_dinov3_m_coco` | ViT-Tiny+ (distilled) | 18.1M | 640×640 | ✅ Supported |

### Fine-tuning ready models

| Model name | Auto download | Notes |
|------------|----------------|-------|
| `deimv2_hgnetv2_atto_coco` | ✅ | HGNetv2 family |
| `deimv2_hgnetv2_femto_coco` | ✅ | HGNetv2 family |
| `deimv2_hgnetv2_pico_coco` | ✅ | HGNetv2 family |
| `deimv2_hgnetv2_n_coco` | ✅ | HGNetv2 family |
| `deimv2_dinov3_s_coco` | ✅ | DINOv3 S |
| `deimv2_dinov3_m_coco` | ✅ | DINOv3 M |
| `deimv2_dinov3_l_coco` | ❌ | Not supported |
| `deimv2_dinov3_x_coco` | ❌ | Not supported |

## Output Files

```text
outputs/
└── my_experiment/
    ├── best_stg*.pth          # Best stage model
    ├── best_stg*.onnx         # ONNX export (when -e is specified)
    ├── last.pth               # Final epoch checkpoint
    ├── checkpoint000X.pth     # Periodic checkpoints
    ├── log.txt                # Training log
    ├── eval/                  # Evaluation results
    └── summary/               # TensorBoard logs
```

## Advanced Usage

### Custom data augmentation

Control data augmentation via the `train_dataloader.dataset.transforms.ops` section in `config.yaml`:

```yaml
transforms:
  ops: # Sample
    - {type: Resize}  # Resize size and Mosaic output_size are derived from image_size automatically
    - {type: ConvertPILImage, dtype: 'float32', scale: true}
    - {type: ConvertBoxes, fmt: 'cxcywh', normalize: true}
```

### Epoch scheduling details

`libs/create_train_config.py` automatically computes the following values:

- `flat_epoch`: 50% of total epochs
- `no_aug_epoch`: 10% of total epochs
- `mixup_epochs`: 4% to 50% of total epochs

Adjust the code if you need different behavior.

## License

This project follows the DEIMv2 license. See `DEIMv2/LICENSE` for details.

## References

- [DEIMv2 official repository](https://github.com/Intellindust-AI-Lab/DEIMv2)
