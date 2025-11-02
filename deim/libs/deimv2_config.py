"""
DEIMv2設定作成モジュール

ユーザー設定からDEIMv2用のYAMLConfig互換オブジェクトを作成します。
"""

import argparse
import logging
import sys
from pathlib import Path

# DEIMv2のモジュールをインポートできるようにパスを追加
# DEIMv2はdeim/DEIMv2に配置されている前提
_module_dir = Path(__file__).parent.absolute()
_deim_dir = _module_dir.parent.absolute()
_deimv2_dir = _deim_dir / 'DEIMv2'
sys.path.insert(0, str(_deimv2_dir))

from engine.core import YAMLConfig

logger = logging.getLogger(__name__)

def create_deimv2_config(user_config: dict, args: argparse.Namespace):
    """
    ユーザー設定からDEIMv2用のYAMLConfig互換オブジェクトを作成します。
    
    Args:
        user_config: configs/config.yamlから読み込んだ設定
        args: コマンドライン引数
        
    Returns:
        YAMLConfig: DEIMv2のYAMLConfigオブジェクト
        
    Raises:
        FileNotFoundError: モデル設定ファイルが見つからない場合
    """
    
    # ベースとなるモデル設定ファイルのパスを決定
    model_name = user_config.get('model', 'deimv2_hgnetv2_s_coco')
    
    # DEIMv2はdeim/DEIMv2に配置されている前提
    script_dir = Path(__file__).parent.parent.absolute()  # deim/libs -> deim
    deimv2_configs_dir = script_dir / 'DEIMv2' / 'configs'
    
    # モデル設定ファイルのパスを構築
    model_config_path = _find_model_config(model_name, deimv2_configs_dir)
    
    logger.info(f"モデル設定ファイル: {model_config_path}")
    
    # YAMLConfigを作成
    cfg = YAMLConfig(str(model_config_path))
    
    # ユーザー設定で上書き
    _apply_dataset_config(cfg, user_config)
    _apply_image_size_config(cfg, user_config)
    _apply_batch_size_config(cfg, user_config)
    _apply_class_config(cfg, user_config)
    _apply_epoch_config(cfg, user_config)
    _apply_optimizer_config(cfg, user_config)
    _apply_dataloader_config(cfg, user_config)
    # output_dirは最後に適用して、モデル設定ファイルの値を確実に上書き
    _apply_output_config(cfg, user_config, args)
    _apply_cli_args(cfg, args)
    _apply_postprocessor_config(cfg, user_config)
    
    return cfg


def _find_model_config(model_name: str, configs_dir: Path) -> Path:
    """
    モデル名に応じた設定ファイルを検索します。
    
    Args:
        model_name: モデル名
        configs_dir: DEIMv2設定ディレクトリ
        
    Returns:
        Path: 設定ファイルのパス
        
    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
    """
    # モデルタイプに応じて適切なディレクトリを選択
    possible_dirs = []
    
    if model_name.startswith('deimv2_'):
        possible_dirs = ['deimv2']
    elif model_name.startswith('deim_hgnetv2_') or model_name.startswith('dfine_hgnetv2_'):
        possible_dirs = ['deim_dfine']
    elif model_name.startswith('deim_r') or model_name.startswith('deim_swin'):
        possible_dirs = ['deim_rtdetrv2']
    else:
        possible_dirs = ['deimv2', 'deim_dfine', 'deim_rtdetrv2']
    
    # 指定されたディレクトリからモデル設定ファイルを検索
    for dir_name in possible_dirs:
        candidate_path = configs_dir / dir_name / f'{model_name}.yml'
        if candidate_path.exists():
            logger.info(f"モデルタイプ: {dir_name}")
            return candidate_path
    
    raise FileNotFoundError(
        f"モデル設定ファイルが見つかりません: {model_name}\n"
        f"検索したディレクトリ: {possible_dirs}\n"
        f"利用可能なモデルは以下のいずれかです:\n"
        f"  - deimv2_*: DEIMv2シリーズ\n"
        f"  - deim_hgnetv2_*: DEIM-DFINEシリーズ\n"
        f"  - dfine_hgnetv2_*: DFINEシリーズ\n"
        f"  - deim_r*/deim_swin*: DEIM-RTDETRv2シリーズ"
    )


def _apply_dataset_config(cfg, user_config: dict):
    """データセット設定を適用します。"""
    if 'train_ann_file' in user_config:
        cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'] = user_config['train_ann_file']
    if 'train_img_folder' in user_config:
        cfg.yaml_cfg['train_dataloader']['dataset']['img_folder'] = user_config['train_img_folder']
    if 'val_ann_file' in user_config:
        cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = user_config['val_ann_file']
    if 'val_img_folder' in user_config:
        cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = user_config['val_img_folder']


def _apply_image_size_config(cfg, user_config: dict):
    """画像サイズ設定を適用します。"""
    if 'image_size' not in user_config:
        return
    
    image_size = user_config['image_size']
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return
    
    # eval_spatial_size を設定
    cfg.yaml_cfg['eval_spatial_size'] = list(image_size)
    
    # Resizeトランスフォームを更新
    _update_transforms_size(cfg.yaml_cfg['train_dataloader']['dataset']['transforms'], image_size, True)
    _update_transforms_size(cfg.yaml_cfg['val_dataloader']['dataset']['transforms'], image_size, False)
    
    mosaic_size = [s // 2 for s in image_size]
    logger.info(f"画像サイズを設定: {image_size}, Mosaicサイズ: {mosaic_size[0]}")


def _update_transforms_size(transforms: dict, image_size: list, is_train: bool):
    """トランスフォームのサイズを更新します。"""
    if 'ops' not in transforms:
        return
    
    mosaic_size = [s // 2 for s in image_size]
    for transform in transforms['ops']:
        if transform.get('type') == 'Resize':
            transform['size'] = list(image_size)
        elif is_train and transform.get('type') == 'Mosaic':
            transform['output_size'] = mosaic_size[0]


def _apply_batch_size_config(cfg, user_config: dict):
    """バッチサイズ設定を適用します。"""
    if 'train_batch_size' in user_config:
        cfg.yaml_cfg['train_dataloader']['total_batch_size'] = user_config['train_batch_size']
        logger.info(f"訓練バッチサイズ: {user_config['train_batch_size']}")
    
    if 'val_batch_size' in user_config:
        # val_dataloaderもtotal_batch_sizeを使用
        cfg.yaml_cfg['val_dataloader']['total_batch_size'] = user_config['val_batch_size']
        logger.info(f"検証バッチサイズ: {user_config['val_batch_size']}")
    else:
        # デフォルトのバッチサイズを設定（指定されていない場合）
        if 'total_batch_size' not in cfg.yaml_cfg.get('val_dataloader', {}):
            cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 8
            logger.info("検証バッチサイズ: 8 (デフォルト)")
    
    # Windowsでのマルチプロセッシング問題を回避するため、num_workersを0に設定
    cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 0
    logger.info("DataLoaderのnum_workersを0に設定（Windowsの互換性のため）")


def _apply_class_config(cfg, user_config: dict):
    """クラス数設定を適用します。"""
    if 'num_classes' in user_config:
        cfg.yaml_cfg['num_classes'] = user_config['num_classes']
        logger.info(f"クラス数: {user_config['num_classes']}")
    
    if 'remap_mscoco' in user_config:
        cfg.yaml_cfg['remap_mscoco_category'] = user_config['remap_mscoco']
        logger.info(f"COCO カテゴリリマップ: {user_config['remap_mscoco']}")
    
    # カテゴリIDを0ベースにリマップする設定
    if user_config.get('remap_to_zero_based', False):
        # カスタムデータセットでカテゴリIDが1から始まる場合、
        # DEIMv2は0ベースのインデックスを期待するため、-1する必要がある
        # これはDEIMv2のCOCODatasetクラスが内部で処理する
        cfg.yaml_cfg['remap_mscoco_category'] = False  # MSCOCOリマップは無効
        logger.info("カテゴリIDを0ベースにリマップ（カスタムデータセット用）")


def _apply_output_config(cfg, user_config: dict, args: argparse.Namespace):
    """出力ディレクトリ設定を適用します。"""
    if 'output_dir' in user_config:
        cfg.yaml_cfg['output_dir'] = user_config['output_dir']
        # YAMLConfigのoutput_dirプロパティも直接更新する必要がある
        cfg.output_dir = user_config['output_dir']
        logger.info(f"出力ディレクトリ: {user_config['output_dir']}")
    elif args.output_dir:
        cfg.yaml_cfg['output_dir'] = args.output_dir
        cfg.output_dir = args.output_dir
        logger.info(f"出力ディレクトリ (CLI引数): {args.output_dir}")


def _apply_epoch_config(cfg, user_config: dict):
    """エポック関連の設定を適用します（自動計算含む）。"""
    if 'epochs' not in user_config:
        return
    
    num_epochs = user_config['epochs']
    cfg.yaml_cfg['epoches'] = num_epochs
    # YAMLConfigのepochesプロパティも直接更新する必要がある
    cfg.epoches = num_epochs
    logger.info(f"エポック数: {num_epochs}")
    
    # 訓練画像数とイテレーション数を計算
    train_img_folder = cfg.yaml_cfg['train_dataloader']['dataset'].get('img_folder', '')
    total_batch_size = cfg.yaml_cfg['train_dataloader'].get('total_batch_size', 16)
    
    num_images = _count_images(train_img_folder)
    iter_per_epoch = max(1, num_images / total_batch_size) if num_images > 0 else 100
    
    # 各パラメータの自動設定
    _set_no_aug_epoch(cfg, user_config, num_epochs)
    _set_flat_epoch(cfg, user_config, num_epochs)
    _set_warmup_iter(cfg, user_config, num_epochs, iter_per_epoch)
    _set_ema_warmups(cfg, user_config, num_epochs, iter_per_epoch)
    _set_mixup_epochs(cfg, user_config, num_epochs)
    _set_stop_epoch(cfg, user_config, num_epochs)
    _set_data_aug_epochs(cfg, user_config, num_epochs)
    _set_matcher_change_epoch(cfg, user_config, num_epochs)


def _count_images(folder: str) -> int:
    """画像数をカウントします。"""
    if not folder or not Path(folder).exists():
        return 0
    
    try:
        num_images = len([f for f in Path(folder).iterdir() if f.is_file()])
        logger.info(f"訓練画像数: {num_images}")
        return num_images
    except Exception as e:
        logger.warning(f"画像数の取得に失敗: {e}")
        return 0


def _set_no_aug_epoch(cfg, user_config: dict, num_epochs: int):
    """no_aug_epochを設定します。"""
    if 'no_aug_epoch' in user_config:
        cfg.yaml_cfg['no_aug_epoch'] = user_config['no_aug_epoch']
        logger.info(f"no_aug_epochを使用: {user_config['no_aug_epoch']}")
    else:
        no_aug_epoch = max(1, int(num_epochs * 0.13))
        cfg.yaml_cfg['no_aug_epoch'] = no_aug_epoch
        logger.info(f"no_aug_epochを自動設定: {no_aug_epoch} (総エポックの13%)")


def _set_flat_epoch(cfg, user_config: dict, num_epochs: int):
    """flat_epochを設定します。"""
    if 'flat_epoch' in user_config:
        cfg.yaml_cfg['flat_epoch'] = user_config['flat_epoch']
        logger.info(f"flat_epochを使用: {user_config['flat_epoch']}")
    else:
        flat_epoch = max(1, int(num_epochs * 0.5))
        cfg.yaml_cfg['flat_epoch'] = flat_epoch
        logger.info(f"flat_epochを自動設定: {flat_epoch} (総エポックの50%)")


def _set_warmup_iter(cfg, user_config: dict, num_epochs: int, iter_per_epoch: float):
    """warmup_iterを設定します。"""
    if 'warmup_iter' in user_config:
        warmup_iter = user_config['warmup_iter']
        logger.info(f"warmup_iterを使用: {warmup_iter}")
    else:
        warmup_iter = int(iter_per_epoch * num_epochs * 0.05)
        warmup_iter = max(warmup_iter, int(iter_per_epoch))
        logger.info(f"warmup_iterを自動設定: {warmup_iter} ({warmup_iter / iter_per_epoch:.1f}エポック)")
    
    cfg.yaml_cfg['lr_warmup_scheduler'] = cfg.yaml_cfg.get('lr_warmup_scheduler', {})
    cfg.yaml_cfg['lr_warmup_scheduler']['warmup_duration'] = warmup_iter


def _set_ema_warmups(cfg, user_config: dict, num_epochs: int, iter_per_epoch: float):
    """ema_warmupsを設定します。"""
    if 'ema_warmups' in user_config:
        ema_warmups = user_config['ema_warmups']
        logger.info(f"ema_warmupsを使用: {ema_warmups}")
    else:
        ema_warmups = int(iter_per_epoch * num_epochs * 0.05)
        ema_warmups = max(ema_warmups, int(iter_per_epoch))
        logger.info(f"ema_warmupsを自動設定: {ema_warmups} ({ema_warmups / iter_per_epoch:.1f}エポック)")
    
    cfg.yaml_cfg['ema'] = cfg.yaml_cfg.get('ema', {})
    cfg.yaml_cfg['ema']['warmups'] = ema_warmups


def _set_mixup_epochs(cfg, user_config: dict, num_epochs: int):
    """mixup_epochsを設定します。"""
    if 'mixup_epochs' in user_config or 'collate_fn' not in cfg.yaml_cfg.get('train_dataloader', {}):
        return
    
    start_mixup = max(1, int(num_epochs * 0.04))
    end_mixup = int(num_epochs * 0.5)
    mixup_epochs = [start_mixup, end_mixup]
    cfg.yaml_cfg['train_dataloader']['collate_fn']['mixup_epochs'] = mixup_epochs
    logger.info(f"mixup_epochsを自動設定: {mixup_epochs}")


def _set_stop_epoch(cfg, user_config: dict, num_epochs: int):
    """stop_epochを設定します。"""
    if 'stop_epoch' in user_config or 'collate_fn' not in cfg.yaml_cfg.get('train_dataloader', {}):
        return
    
    stop_epoch = int(num_epochs * 0.9)
    cfg.yaml_cfg['train_dataloader']['collate_fn']['stop_epoch'] = stop_epoch
    logger.info(f"stop_epochを自動設定: {stop_epoch} (総エポックの90%)")


def _set_data_aug_epochs(cfg, user_config: dict, num_epochs: int):
    """data augmentation epochsを設定します。"""
    transforms = cfg.yaml_cfg.get('train_dataloader', {}).get('dataset', {}).get('transforms', {})
    if 'policy' not in transforms:
        return
    
    user_policy = user_config.get('train_dataloader', {}).get('dataset', {}).get('transforms', {}).get('policy', {})
    if user_policy is not None and 'epoch' in user_policy:
        return
    
    data_aug_1 = max(1, int(num_epochs * 0.04))
    data_aug_2 = int(num_epochs * 0.5)
    data_aug_3 = int(num_epochs * 0.9)
    cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'] = [data_aug_1, data_aug_2, data_aug_3]
    logger.info(f"data augmentation epochsを自動設定: [{data_aug_1}, {data_aug_2}, {data_aug_3}]")


def _set_matcher_change_epoch(cfg, user_config: dict, num_epochs: int):
    """matcher_change_epochを設定します。"""
    if 'DEIMCriterion' not in cfg.yaml_cfg or 'matcher' not in cfg.yaml_cfg['DEIMCriterion']:
        return
    
    user_criterion = user_config.get('DEIMCriterion', {}).get('matcher', {})
    if 'matcher_change_epoch' in user_criterion:
        return
    
    matcher_change_epoch = int(num_epochs * 0.9)
    cfg.yaml_cfg['DEIMCriterion']['matcher']['matcher_change_epoch'] = matcher_change_epoch
    logger.info(f"matcher_change_epochを自動設定: {matcher_change_epoch} (総エポックの90%)")


def _apply_optimizer_config(cfg, user_config: dict):
    """オプティマイザ設定を適用します。"""
    if 'optimizer' not in user_config:
        return
    
    optimizer_config = user_config['optimizer']
    if 'lr' in optimizer_config:
        cfg.yaml_cfg['optimizer']['lr'] = optimizer_config['lr']
        logger.info(f"学習率: {optimizer_config['lr']}")
    if 'weight_decay' in optimizer_config:
        cfg.yaml_cfg['optimizer']['weight_decay'] = optimizer_config['weight_decay']
        logger.info(f"重み減衰: {optimizer_config['weight_decay']}")
    if 'betas' in optimizer_config:
        cfg.yaml_cfg['optimizer']['betas'] = optimizer_config['betas']


def _apply_dataloader_config(cfg, user_config: dict):
    """データローダー設定を適用します。"""
    _apply_train_dataloader_config(cfg, user_config)
    _apply_eval_dataloader_config(cfg, user_config)


def _apply_train_dataloader_config(cfg, user_config: dict):
    """訓練データローダー設定を適用します。"""
    if 'train_dataloader' not in user_config:
        return
    
    train_dl_config = user_config['train_dataloader']
    
    if 'total_batch_size' in train_dl_config:
        cfg.yaml_cfg['train_dataloader']['total_batch_size'] = train_dl_config['total_batch_size']
    
    if 'dataset' not in train_dl_config or 'transforms' not in train_dl_config['dataset']:
        return
    
    transforms_config = train_dl_config['dataset']['transforms']
    
    if 'policy' in transforms_config:
        cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['policy'] = transforms_config['policy']
    
    if 'ops' in transforms_config:
        _apply_user_transforms(cfg, user_config, transforms_config['ops'], is_train=True)


def _apply_eval_dataloader_config(cfg, user_config: dict):
    """評価データローダー設定を適用します。"""
    if 'eval_dataloader' not in user_config:
        return
    
    eval_dl_config = user_config['eval_dataloader']
    
    if 'dataset' not in eval_dl_config or 'transforms' not in eval_dl_config['dataset']:
        return
    
    transforms_config = eval_dl_config['dataset']['transforms']
    
    if 'policy' in transforms_config:
        cfg.yaml_cfg['val_dataloader']['dataset']['transforms']['policy'] = transforms_config['policy']
    
    if 'ops' in transforms_config:
        _apply_user_transforms(cfg, user_config, transforms_config['ops'], is_train=False)


def _apply_user_transforms(cfg, user_config: dict, user_ops: list, is_train: bool):
    """ユーザー定義のトランスフォームを適用します。"""
    # image_sizeを各transformに反映
    if 'image_size' in user_config:
        image_size = user_config['image_size']
        mosaic_size = [s // 2 for s in image_size]
        
        for transform in user_ops:
            if transform.get('type') == 'Resize':
                transform['size'] = list(image_size)
            elif is_train and transform.get('type') == 'Mosaic':
                transform['output_size'] = mosaic_size[0]
    
    dataloader_key = 'train_dataloader' if is_train else 'val_dataloader'
    cfg.yaml_cfg[dataloader_key]['dataset']['transforms']['ops'] = user_ops
    logger.info(f"ユーザー定義の{'train' if is_train else 'eval'} transforms.opsを適用")


def _apply_cli_args(cfg, args: argparse.Namespace):
    """コマンドライン引数を適用します。"""
    if args.device:
        cfg.yaml_cfg['device'] = args.device
    
    if args.use_amp:
        cfg.yaml_cfg['use_amp'] = True
        logger.info("自動混合精度(AMP)を有効化")
    
    if args.resume:
        cfg.yaml_cfg['resume'] = args.resume
        logger.info(f"チェックポイントから再開: {args.resume}")
    
    if args.tuning:
        cfg.yaml_cfg['tuning'] = args.tuning
        logger.info(f"チェックポイントからファインチューニング: {args.tuning}")
    
    if args.seed is not None:
        cfg.yaml_cfg['seed'] = args.seed
        logger.info(f"乱数シード: {args.seed}")


def _apply_postprocessor_config(cfg, user_config: dict):
    """ポストプロセッサ設定を適用します。"""
    if 'num_top_queries' in user_config:
        # PostProcessorの設定を更新
        if 'PostProcessor' not in cfg.yaml_cfg:
            cfg.yaml_cfg['PostProcessor'] = {}
        cfg.yaml_cfg['PostProcessor']['num_top_queries'] = user_config['num_top_queries']
        logger.info(f"num_top_queries: {user_config['num_top_queries']}")
