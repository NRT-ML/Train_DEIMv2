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

    # 訓練後に自動的にONNX形式にエクスポート
    python deim/train.py --export-onnx

    # スタンドアロンでONNX変換のみ実行
    python deim/train.py --export-onnx --resume outputs/best_stg1.pth --test-only

    # ONNXエクスポートの詳細オプション
    python deim/train.py --export-onnx --onnx-output model.onnx --onnx-opset 16 --onnx-batch-size 16

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

# Windows環境での分散処理の問題を回避するための環境変数設定
# libuvサポートなしのPyTorchで発生するエラーを防ぐ
if 'USE_LIBUV' not in os.environ:
    os.environ['USE_LIBUV'] = '0'

script_dir = Path(__file__).parent.absolute()
deimv2_dir = script_dir / 'DEIMv2'
sys.path.insert(0, str(deimv2_dir))

from libs import load_config_from_yaml, create_deimv2_config, export_to_onnx, save_training_config

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
    
    # 分散訓練が初期化されていない場合(単一GPU)、
    # Windows環境ではlibuvサポートの問題があるため、
    # 単一GPUモードでは分散処理を初期化しない
    if not torch.distributed.is_initialized():
        logger.info("単一GPUモード: 分散処理を使用しません")
        # 分散処理を使用しない場合は、環境変数を設定
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
    
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
        
        # 訓練完了後、設定を保存
        output_dir = cfg.yaml_cfg.get('output_dir', './outputs')
        best_model_path = os.path.join(output_dir, 'best_stg1.pth')
        if os.path.exists(best_model_path):
            logger.info("訓練時の最終設定を保存...")
            save_training_config(best_model_path, cfg.yaml_cfg)
        
        # 訓練完了後、ONNXエクスポートを実行
        if args.export_onnx:
            logger.info("\n訓練が完了しました。ONNXエクスポートを開始します...")
            
            # ベストモデルが存在しない場合は最後のモデルを使用
            if not os.path.exists(best_model_path):
                logger.warning(f"ベストモデルが見つかりません: {best_model_path}")
                best_model_path = os.path.join(output_dir, 'last.pth')
                
                if not os.path.exists(best_model_path):
                    logger.error(f"モデルファイルが見つかりません: {best_model_path}")
                    logger.error("ONNXエクスポートをスキップします")
                else:
                    logger.info(f"最後のモデルを使用します: {best_model_path}")
            else:
                logger.info(f"ベストモデルを使用します: {best_model_path}")
            
            # ONNXエクスポートを実行
            if os.path.exists(best_model_path):
                onnx_path = export_to_onnx(
                    checkpoint_path=best_model_path,
                    config_path=None,  # 保存された設定を自動検出
                    output_path=args.onnx_output,
                    opset_version=args.onnx_opset,
                    check_model=args.onnx_check,
                    simplify=args.onnx_simplify,
                    batch_size=args.onnx_batch_size
                )
                
                if onnx_path:
                    logger.info(f"✓ ONNXモデルが正常に保存されました: {onnx_path}")

    
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

  # 訓練後にONNX変換
  python deim/train.py --export-onnx

  # スタンドアロンでONNX変換のみ実行
  python deim/train.py --export-onnx --resume outputs/best_stg1.pth --test-only

  # ONNX変換の詳細オプション
  python deim/train.py --export-onnx --onnx-output model.onnx --onnx-opset 16
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
    
    # ONNXエクスポート関連のオプション
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        help='訓練完了後にONNX形式にエクスポート'
    )
    
    parser.add_argument(
        '--onnx-output',
        type=str,
        default=None,
        help='ONNXファイルの出力パス（デフォルト: モデルと同じディレクトリ）'
    )
    
    parser.add_argument(
        '--onnx-opset',
        type=int,
        default=17,
        help='ONNXのopsetバージョン（デフォルト: 17）'
    )
    
    parser.add_argument(
        '--onnx-check',
        action='store_true',
        default=True,
        help='エクスポート後にONNXモデルを検証'
    )
    
    parser.add_argument(
        '--onnx-simplify',
        action='store_true',
        default=True,
        help='onnx-simplifierで最適化（デフォルト: 有効）'
    )
    
    parser.add_argument(
        '--onnx-batch-size',
        type=int,
        default=32,
        help='ONNXエクスポート時のバッチサイズ（デフォルト: 32）'
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
    
    # スタンドアロンでONNX変換のみを実行する場合
    # --export-onnx --resume <checkpoint> --test-only のように指定
    if args.export_onnx and args.resume and args.test_only:
        logger.info("スタンドアロンONNX変換モード")
        
        onnx_path = export_to_onnx(
            checkpoint_path=args.resume,
            config_path=args.config,  # Noneの場合は自動検出
            output_path=args.onnx_output,
            opset_version=args.onnx_opset,
            check_model=args.onnx_check,
            simplify=args.onnx_simplify,
            batch_size=args.onnx_batch_size
        )
        if onnx_path:
            logger.info(f"✓ ONNXモデルが正常に保存されました: {onnx_path}")
    else:
        # 通常の訓練/評価を実行
        main(args)


