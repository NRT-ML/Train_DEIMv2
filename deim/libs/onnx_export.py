"""
ONNX エクスポート機能

訓練済みDEIMv2モデルをONNX形式にエクスポートするためのユーティリティ関数を提供します。

Example:
    >>> from libs.onnx_export import export_to_onnx
    >>> onnx_path = export_to_onnx(
    ...     checkpoint_path='outputs/best_stg1.pth',
    ...     config_path='configs/config.yaml'
    ... )
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

try:
    import onnx
    import onnxsim
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from engine.core import YAMLConfig
    DEIMV2_AVAILABLE = True
except ImportError:
    DEIMV2_AVAILABLE = False

from .checkpoint_utils import find_config_for_checkpoint

# ロガーの取得
logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str,
    config_path: str = None,
    output_path: str = None,
    opset_version: int = 17,
    check_model: bool = True,
    simplify: bool = True,
    batch_size: int = 32
):
    """
    訓練済みモデルをONNX形式にエクスポートします。
    
    この関数は、DEIMv2モデルのチェックポイント(.pth)を読み込み、
    ONNX形式に変換します。EMAモデルが存在する場合は自動的に優先して使用され、
    オプションでモデルの検証と最適化も行います。
    
    重要: config_pathは訓練時に使用した設定ファイルと同じものを指定してください。
    チェックポイントには設定情報が含まれていないため、モデルアーキテクチャ、
    画像サイズ、クラス数などが訓練時と一致する必要があります。
    
    Args:
        checkpoint_path (str): チェックポイントファイル(.pth)のパス
        config_path (str, optional): 設定ファイル(.yaml)のパス。
            訓練時に使用した設定ファイルと同じものを指定してください。
            Noneの場合、チェックポイントと同じディレクトリのconfig.yamlを探索します。
        output_path (str, optional): 出力ファイルパス。Noneの場合は自動生成
            （checkpoint_pathの.pthを.onnxに置換）
        opset_version (int): ONNXのopsetバージョン。デフォルトは17
        check_model (bool): エクスポート後にモデルをチェックするかどうか。
            デフォルトはTrue
        simplify (bool): onnx-simplifierで最適化するかどうか。
            デフォルトはTrue
        batch_size (int): エクスポート時のダミーデータのバッチサイズ。
            デフォルトは32
    
    Returns:
        str: 出力されたONNXファイルのパス。エラーが発生した場合はNone
    
    Raises:
        ImportError: 必要なモジュール（engine.core.YAMLConfig）が
            インポートできない場合
        FileNotFoundError: checkpoint_pathまたはconfig_pathが存在しない場合
    
    Example:
        >>> # 基本的な使用（訓練時の設定ファイルを指定）
        >>> onnx_path = export_to_onnx(
        ...     checkpoint_path='outputs/best_stg1.pth',
        ...     config_path='configs/config.yaml'  # 訓練時と同じ設定
        ... )
        
        >>> # カスタムオプション付き
        >>> onnx_path = export_to_onnx(
        ...     checkpoint_path='outputs/best_stg1.pth',
        ...     config_path='configs/config.yaml',  # 訓練時と同じ設定
        ...     output_path='my_model.onnx',
        ...     opset_version=16,
        ...     batch_size=16
        ... )
    
    Warning:
        config_pathには必ず訓練時に使用した設定ファイルを指定してください。
        異なる設定を使用すると、モデルの構造が一致せず、エラーが発生したり、
        正しく変換できない可能性があります。
    
    Note:
        - ONNXエクスポート機能を使用するには、onnxパッケージが必要です
        - 最適化機能を使用するには、onnx-simplifierパッケージが必要です
        - エクスポートされるモデルは、動的バッチサイズをサポートします
    """
    logger.info("=" * 80)
    logger.info("ONNX変換を開始します")
    logger.info("=" * 80)
    logger.warning("⚠ 重要: config_pathには訓練時に使用した設定ファイルを指定してください")
    
    # DEIMv2のインポート確認
    if not DEIMV2_AVAILABLE:
        logger.error("engine.core.YAMLConfigのインポートに失敗しました")
        logger.error("DEIMv2のengineモジュールが正しくインストールされているか確認してください")
        return None
    
    # チェックポイントファイルの存在確認
    if not os.path.exists(checkpoint_path):
        logger.error(f"チェックポイントファイルが見つかりません: {checkpoint_path}")
        return None
    
    # 設定ファイルのパスを解決
    if config_path is None:
        logger.info("設定ファイルが指定されていません。自動検索を試みます...")
        config_path = find_config_for_checkpoint(checkpoint_path)
        
        if config_path is None:
            logger.error("設定ファイルが見つかりません。config_pathを明示的に指定してください。")
            logger.error("設定ファイルは訓練時に使用したものと同じである必要があります。")
            logger.info("\nヒント:")
            logger.info("  1. 訓練時の設定ファイルパスを--onnx-config-pathで指定")
            logger.info("  2. または、訓練完了時に自動的に保存された設定を使用")
            return None
        else:
            logger.info(f"✓ 訓練時の設定を自動検出: {config_path}")
    else:
        logger.info(f"指定された設定ファイル: {config_path}")
    
    # 設定ファイルの存在確認
    if not os.path.exists(config_path):
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return None
    
    logger.info(f"✓ チェックポイント: {checkpoint_path}")
    logger.info(f"✓ 設定ファイル: {config_path}")
    
    # チェックポイントを読み込む
    logger.info("チェックポイントを読み込み中...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # チェックポイントの内容を確認
    logger.info(f"チェックポイントに含まれるキー: {list(checkpoint.keys())}")
    
    # EMAモデルまたは通常のモデルの状態を取得
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        logger.info("✓ EMAモデルの重みを使用します（より高精度）")
    else:
        state = checkpoint['model']
        logger.info("✓ 通常のモデルの重みを使用します")
    
    # 設定を読み込む
    logger.info(f"設定ファイルを読み込み中: {config_path}")
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    # 事前学習済み重みの読み込みを無効化
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        logger.info("HGNetv2の事前学習済み重みの読み込みを無効化")
    
    # チェックポイントに訓練情報がある場合は表示
    if 'last_epoch' in checkpoint:
        logger.info(f"  - 訓練エポック: {checkpoint['last_epoch']}")
    if 'date' in checkpoint:
        logger.info(f"  - 保存日時: {checkpoint['date']}")
    
    logger.warning("⚠ 上記の設定が訓練時と一致しているか確認してください")
    logger.info("=" * 80)
    
    # モデルに重みをロード
    cfg.model.load_state_dict(state)
    logger.info("✓ モデルの重みをロードしました")
    
    # デプロイ用のモデルを作成
    class ONNXExportModel(nn.Module):
        """
        ONNX変換用のモデルラッパー
        
        このクラスは、DEIMv2モデルの推論とポストプロセスを一つにまとめ、
        ONNX形式にエクスポート可能な形式に変換します。
        
        Attributes:
            model: デプロイモードのDEIMv2モデル
            postprocessor: デプロイモードのポストプロセッサ
        """
        def __init__(self, model, postprocessor):
            """
            ONNXエクスポート用モデルを初期化します。
            
            Args:
                model: DEIMv2モデル
                postprocessor: ポストプロセッサ
            """
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            """
            順伝播処理を実行します。
            
            Args:
                images (torch.Tensor): 入力画像 [N, 3, H, W]
                    - N: バッチサイズ（動的）
                    - 3: RGBチャンネル
                    - H, W: 画像の高さと幅
                orig_target_sizes (torch.Tensor): 元の画像サイズ [N, 2]
                    - 各画像の元のサイズ (height, width)
            
            Returns:
                tuple: (labels, boxes, scores)
                    - labels: クラスラベル
                    - boxes: バウンディングボックス座標 [x1, y1, x2, y2]
                    - scores: 信頼度スコア
            """
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = ONNXExportModel(cfg.model, cfg.postprocessor)
    model.eval()
    logger.info("✓ デプロイ用モデルを作成しました")
    
    # ダミーデータを作成
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    dummy_images = torch.rand(batch_size, 3, *img_size)
    dummy_sizes = torch.tensor([img_size])
    
    logger.info(f"入力サイズ: images={dummy_images.shape}, orig_target_sizes={dummy_sizes.shape}")
    
    # テスト実行
    logger.info("モデルのテスト実行中...")
    with torch.no_grad():
        _ = model(dummy_images, dummy_sizes)
    logger.info("✓ モデルのテスト実行が成功しました")
    
    # 動的軸の設定（バッチサイズを可変にする）
    dynamic_axes = {
        'images': {0: 'N'},
        'orig_target_sizes': {0: 'N'}
    }
    
    # 出力ファイルパスを決定
    if output_path is None:
        output_path = checkpoint_path.replace('.pth', '.onnx')
    
    logger.info(f"ONNX形式でエクスポート中: {output_path}")
    logger.info(f"  - Opsetバージョン: {opset_version}")
    logger.info(f"  - 動的バッチサイズ: 有効")
    
    # ONNXにエクスポート
    torch.onnx.export(
        model,
        (dummy_images, dummy_sizes),
        output_path,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        verbose=False,
        do_constant_folding=True,
    )
    
    logger.info(f"✓ ONNXエクスポート完了: {output_path}")
    
    # モデルのチェック
    if check_model:
        if not ONNX_AVAILABLE:
            logger.warning("⚠ onnxパッケージがインストールされていないため、検証をスキップします")
            logger.info("  インストール: pip install onnx")
        else:
            try:
                logger.info("ONNXモデルの検証中...")
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info("✓ ONNXモデルの検証が成功しました")
            except Exception as e:
                logger.error(f"✗ ONNXモデルの検証に失敗しました: {e}")
                logger.warning("  モデルは保存されていますが、正常に動作しない可能性があります")
    
    # モデルの最適化
    if simplify:
        if not ONNX_AVAILABLE:
            logger.warning("⚠ onnx-simplifierがインストールされていないため、最適化をスキップします")
            logger.info("  最適化を有効にするには: pip install onnx-simplifier")
        else:
            try:
                logger.info("ONNX-Simplifierで最適化中...")
                
                input_shapes = {
                    'images': list(dummy_images.shape),
                    'orig_target_sizes': list(dummy_sizes.shape)
                }
                
                onnx_model_simplify, check = onnxsim.simplify(
                    output_path,
                    test_input_shapes=input_shapes
                )
                
                if check:
                    onnx.save(onnx_model_simplify, output_path)
                    logger.info("✓ ONNX最適化が成功しました")
                    logger.info("  推論速度の向上が期待できます")
                else:
                    logger.warning("⚠ ONNX最適化に失敗しましたが、元のモデルは保存されています")
                    
            except Exception as e:
                logger.error(f"✗ ONNX最適化に失敗しました: {e}")
                logger.info("  元のONNXモデルは保存されています")
    
    logger.info("=" * 80)
    logger.info(f"ONNX変換が完了しました: {output_path}")
    logger.info("=" * 80)
    
    return output_path
