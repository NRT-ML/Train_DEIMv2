import os, sys
import subprocess
from pathlib import Path
from libs.create_train_config import CreateTrainConfig
import argparse


def main(cfg_path: str):
    """設定ファイルから訓練用設定を生成し、訓練を実行"""
    # 訓練用設定ファイルの生成
    ctc = CreateTrainConfig(cfg_path)
    train_cfg_path = ctc.train_cfg_path
    
    # 環境変数の設定
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # コマンドの実行（uv仮想環境のPythonを使用）
    subprocess.run(
        [sys.executable, "DEIMv2/train.py", "-c", str(train_cfg_path), "--use-amp", "--seed=0"],
        env=env
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    
    main(args.config)