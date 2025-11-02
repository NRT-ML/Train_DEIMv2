"""
設定ファイル読み込みモジュール

YAMLファイルから訓練設定を読み込むための関数を提供します。
"""

import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def load_config_from_yaml(config_path: str) -> dict:
    """
    YAMLファイルから設定を読み込みます。
    
    Args:
        config_path: YAMLファイルのパス
        
    Returns:
        dict: 読み込んだ設定の辞書
        
    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        yaml.YAMLError: YAMLファイルのパースに失敗した場合
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    logger.info(f"設定ファイルを読み込み中: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML解析エラー: {e}")
        raise
