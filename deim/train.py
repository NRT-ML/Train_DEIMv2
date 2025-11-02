"""
DEIMv2モデルの訓練スクリプト

このスクリプトは、DEIMv2モデルを訓練するためのものです。
DEIMKitのTrainerクラスを利用し、configs/config.yamlから設定を読み込みます。

Usage:
    # 単一GPU訓練
    python deim/train.py

    # マルチGPU訓練（4 GPU の例）
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 deim/train.py

    # カスタム設定ファイルを指定
    python deim/train.py --config path/to/custom_config.yaml

    # 学習済みチェックポイントから再開
    python deim/train.py --resume path/to/checkpoint.pth

    # ファインチューニング
    python deim/train.py --tuning path/to/pretrained.pth

    # 評価のみ
    python deim/train.py --test-only --resume path/to/checkpoint.pth

References:
    - DEIMv2: https://github.com/Intellindust-AI-Lab/DEIMv2
    - DEIMKit: https://github.com/dnth/DEIMKit
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

script_dir = Path(__file__).parent.absolute()
deimv2_dir = script_dir / 'DEIMv2'
sys.path.insert(0, str(deimv2_dir))

# # ================================
# # torchvision v2互換性パッチ
# # ================================
# # DEIMv2のカスタムトランスフォームは_transform()メソッドを実装していますが、
# # torchvision v2の新しいAPIではtransform()メソッドが必要です。
# # このパッチは、DEIMv2を変更せずにtorchvision v2との互換性を確保します。
# import torchvision.transforms.v2 as T

# # 元のTransformクラスのtransform()メソッドを保存
# _original_transform = T.Transform.transform

# def _patched_transform(self, inpt, params):
#     """
#     torchvision v2互換性のためのパッチ。
    
#     DEIMv2のカスタムトランスフォームは_transform()を実装しているため、
#     NotImplementedErrorの代わりに_transform()を呼び出します。
#     """
#     if hasattr(self, '_transform'):
#         # DEIMv2のカスタムトランスフォームの場合、_transform()を使用
#         return self._transform(inpt, params)
#     else:
#         # 通常のtorchvision transformsの場合、元の動作を維持
#         return _original_transform(self, inpt, params)

# # transform()メソッドをパッチ
# T.Transform.transform = _patched_transform

from libs import load_config_from_yaml, create_deimv2_config

# DEIMv2のengineモジュールをインポート
from engine.misc import dist_utils
from engine.solver import TASKS

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

def main(args: argparse.Namespace):
    """
    メイン関数: DEIMv2モデルの訓練を実行します。
    
    Args:
        args: コマンドライン引数
    """
    logger.info("=" * 80)
    logger.info("DEIMv2 訓練スクリプト")
    logger.info("=" * 80)
    
    # 分散訓練のセットアップ
    # 単一GPUの場合でも、torch.distributedを初期化する必要がある
    # （DEIMv2のバックボーンコードがget_rank()を呼び出すため）
    dist_utils.setup_distributed(
        print_rank=args.print_rank,
        print_method=args.print_method,
        seed=args.seed
    )
    
    # 分散訓練が初期化されていない場合（単一GPU）、
    # ダミーの分散環境を初期化する
    if not torch.distributed.is_initialized():
        logger.info("単一GPUモード: torch.distributedを初期化")
        torch.distributed.init_process_group(
            backend='gloo',  # Windowsでも動作するバックエンド
            init_method='tcp://127.0.0.1:23456',
            world_size=1,
            rank=0
        )
    
    # 設定ファイルを読み込み
    user_config = load_config_from_yaml(args.config)
    
    # DEIMv2用の設定を作成
    cfg = create_deimv2_config(user_config, args)
    
    # 再開とチューニングの両方は同時にサポートしない
    assert not all([args.tuning, args.resume]), \
        '再開(--resume)とチューニング(--tuning)は同時に指定できません'
    
    # 再開またはチューニングの場合、事前学習済み重みの読み込みを無効化
    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            logger.info("HGNetv2の事前学習済み重みの読み込みを無効化")
    
    # 設定を表示
    logger.info("=" * 80)
    logger.info("設定内容:")
    logger.info("=" * 80)
    logger.info(f"モデル: {user_config.get('model', 'deimv2_hgnetv2_s_coco')}")
    logger.info(f"出力ディレクトリ: {cfg.yaml_cfg.get('output_dir', './outputs')}")
    logger.info(f"エポック数: {cfg.yaml_cfg.get('epoches', 50)}")
    logger.info(f"バッチサイズ: {cfg.yaml_cfg['train_dataloader'].get('total_batch_size', 'N/A')}")
    logger.info(f"学習率: {cfg.yaml_cfg['optimizer'].get('lr', 'N/A')}")
    logger.info(f"画像サイズ: {cfg.yaml_cfg.get('eval_spatial_size', 'N/A')}")
    logger.info(f"クラス数: {cfg.yaml_cfg.get('num_classes', 'N/A')}")
    logger.info(f"デバイス: {cfg.yaml_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"AMP: {cfg.yaml_cfg.get('use_amp', False)}")
    logger.info("=" * 80)
    
    # タスクに応じたソルバーを作成
    task = cfg.yaml_cfg.get('task', 'detection')
    logger.info(f"タスク: {task}")
    
    solver = TASKS[task](cfg)
    
    # 評価のみの場合
    if args.test_only:
        logger.info("評価モードで実行中...")
        solver.val()
    else:
        # 訓練を実行
        logger.info("訓練を開始します...")
        solver.fit()
    
    # クリーンアップ
    dist_utils.cleanup()
    
    logger.info("=" * 80)
    logger.info("訓練/評価が完了しました!")
    logger.info("=" * 80)


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数をパースします。
    
    Returns:
        argparse.Namespace: パースされた引数
    """
    parser = argparse.ArgumentParser(
        description='DEIMv2モデルの訓練スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な訓練
  python deim/train.py

  # カスタム設定ファイルを使用
  python deim/train.py --config path/to/config.yaml

  # マルチGPU訓練（4 GPU）
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 deim/train.py

  # チェックポイントから再開
  python deim/train.py --resume outputs/checkpoint.pth

  # ファインチューニング
  python deim/train.py --tuning pretrained/model.pth

  # 評価のみ
  python deim/train.py --test-only --resume outputs/best.pth

  # AMP（自動混合精度）を有効化
  python deim/train.py --use-amp
        """
    )
    
    # 優先度0: 基本設定
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='configs/config.yaml',
        help='設定ファイルのパス（デフォルト: configs/config.yaml）'
    )
    
    parser.add_argument(
        '-r', '--resume',
        type=str,
        default=None,
        help='チェックポイントから訓練を再開'
    )
    
    parser.add_argument(
        '-t', '--tuning',
        type=str,
        default=None,
        help='チェックポイントからファインチューニング'
    )
    
    parser.add_argument(
        '-d', '--device',
        type=str,
        default=None,
        help='使用するデバイス（例: cuda, cpu, cuda:0）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='乱数シード（再現性のため）'
    )
    
    parser.add_argument(
        '--use-amp',
        action='store_true',
        help='自動混合精度（AMP）訓練を有効化'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='出力ディレクトリ'
    )
    
    parser.add_argument(
        '--summary-dir',
        type=str,
        default=None,
        help='TensorBoardサマリーディレクトリ'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        default=False,
        help='評価のみ実行（訓練しない）'
    )
    
    # 優先度1: 詳細設定
    parser.add_argument(
        '-u', '--update',
        nargs='+',
        default=None,
        help='YAML設定を更新（例: --update train_dataloader.batch_size=16）'
    )
    
    # 環境設定
    parser.add_argument(
        '--print-method',
        type=str,
        default='builtin',
        help='print メソッド（builtin または loguru）'
    )
    
    parser.add_argument(
        '--print-rank',
        type=int,
        default=0,
        help='出力するランクID（分散訓練時）'
    )
    
    parser.add_argument(
        '--local-rank',
        type=int,
        default=None,
        help='ローカルランクID（torchrun用）'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
