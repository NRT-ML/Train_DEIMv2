"""
チェックポイントユーティリティ

チェックポイントに設定情報を保存・復元するための関数を提供します。
"""

import logging
import os
import shutil
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def save_training_config(checkpoint_path: str, config_dict: dict):
    """
    訓練時の最終的な設定（cfg.yaml_cfg）をYAMLファイルとして保存します。
    
    この関数は、訓練中に構築・調整された最終的な設定を保存します。
    ONNX変換時には、この保存された設定を使用することで、
    訓練時と完全に同じ設定でモデルを構築できます。
    
    Args:
        checkpoint_path (str): チェックポイントファイルのパス
        config_dict (dict): 保存する設定辞書（cfg.yaml_cfg）
    
    Example:
        >>> # 訓練完了時
        >>> save_training_config(
        ...     checkpoint_path='outputs/best_stg1.pth',
        ...     config_dict=cfg.yaml_cfg
        ... )
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"チェックポイントファイルが見つかりません: {checkpoint_path}")
            return
        
        # チェックポイントと同じディレクトリに設定を保存
        output_config_path = checkpoint_path.parent / f"{checkpoint_path.stem}_config.yaml"
        
        # __include__キーを除外してコピー
        # YAMLConfigの内部で使用される特殊キーで、保存には不要
        config_to_save = {k: v for k, v in config_dict.items() if k != '__include__'}
        
        # YAMLとして保存
        with open(output_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"✓ 訓練時の設定を保存: {output_config_path}")
        
    except Exception as e:
        logger.error(f"設定ファイルの保存に失敗しました: {e}")


def save_config_with_checkpoint(checkpoint_path: str, config_path: str):
    """
    チェックポイントと一緒に設定ファイルをコピーして保存します。
    
    注意: この関数は後方互換性のために残していますが、
    save_training_config()を使用することを推奨します。
    
    訓練完了後、チェックポイントと同じディレクトリに設定ファイルをコピーすることで、
    後でONNX変換などを行う際に、正しい設定を使用できるようにします。
    
    Args:
        checkpoint_path (str): チェックポイントファイルのパス
        config_path (str): 設定ファイルのパス
    
    Example:
        >>> save_config_with_checkpoint(
        ...     checkpoint_path='outputs/best_stg1.pth',
        ...     config_path='configs/config.yaml'
        ... )
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        config_path = Path(config_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"チェックポイントファイルが見つかりません: {checkpoint_path}")
            return
        
        if not config_path.exists():
            logger.warning(f"設定ファイルが見つかりません: {config_path}")
            return
        
        # チェックポイントと同じディレクトリに設定をコピー
        output_config_path = checkpoint_path.parent / f"{checkpoint_path.stem}_config.yaml"
        
        shutil.copy2(config_path, output_config_path)
        logger.info(f"✓ 設定ファイルを保存: {output_config_path}")
        
    except Exception as e:
        logger.error(f"設定ファイルの保存に失敗しました: {e}")


def find_config_for_checkpoint(checkpoint_path: str) -> str:
    """
    チェックポイントに対応する設定ファイルを探します。
    
    以下の優先順位で設定ファイルを探索します:
    1. <checkpoint>_config.yaml (save_config_with_checkpointで保存されたもの)
    2. チェックポイントと同じディレクトリのconfig.yaml/config.yml
    3. チェックポイントの親ディレクトリのconfig.yaml/config.yml
    
    Args:
        checkpoint_path (str): チェックポイントファイルのパス
    
    Returns:
        str: 見つかった設定ファイルのパス。見つからない場合はNone
    
    Example:
        >>> config_path = find_config_for_checkpoint('outputs/best_stg1.pth')
        >>> if config_path:
        ...     print(f"設定ファイル: {config_path}")
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.warning(f"チェックポイントファイルが見つかりません: {checkpoint_path}")
        return None
    
    # 1. <checkpoint>_config.yamlを探す
    saved_config = checkpoint_path.parent / f"{checkpoint_path.stem}_config.yaml"
    if saved_config.exists():
        logger.info(f"✓ 保存された設定ファイルを検出: {saved_config}")
        return str(saved_config)
    
    # 2. 同じディレクトリのconfig.yaml/config.ymlを探す
    for config_name in ['config.yaml', 'config.yml']:
        config_path = checkpoint_path.parent / config_name
        if config_path.exists():
            logger.info(f"✓ 設定ファイルを検出: {config_path}")
            logger.warning("⚠ この設定ファイルが訓練時と同じか確認してください")
            return str(config_path)
    
    # 3. 親ディレクトリのconfig.yaml/config.ymlを探す
    for config_name in ['config.yaml', 'config.yml']:
        config_path = checkpoint_path.parent.parent / config_name
        if config_path.exists():
            logger.info(f"✓ 設定ファイルを検出: {config_path}")
            logger.warning("⚠ この設定ファイルが訓練時と同じか確認してください")
            return str(config_path)
    
    logger.warning("設定ファイルが見つかりませんでした")
    return None
