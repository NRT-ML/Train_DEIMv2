import os, sys
import subprocess
from pathlib import Path
import argparse

from libs.create_train_config import CreateTrainConfig
from libs.export_to_onnx import export_to_onnx

def main(cfg_path: str):
    """設定ファイルから訓練用設定を生成し、訓練を実行"""
    # 訓練用設定ファイルの生成
    ctc = CreateTrainConfig(cfg_path)
    train_cfg_path = ctc.train_cfg_path
    
    # 環境変数の設定
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # コマンドの実行（仮想環境のPythonを使用）
    subprocess.run(
        [sys.executable, "DEIMv2/train.py", "-c", str(train_cfg_path), "--use-amp", "--seed=0"],
        env=env
    )

    resume = list(Path(ctc.cfg["output_dir"]).glob("**/best_stg*.pth"))[-1]

    # onnxエクスポート
    if args.export_onnx:
        if resume:
            # ONNXエクスポートを実行
            onnx_path = export_to_onnx(
                config_path=str(train_cfg_path),
                resume=str(resume),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the config YAML file")
    parser.add_argument('--export-onnx', action='store_true', help="Export the trained model to ONNX format after training")
    args = parser.parse_args()
    
    main(args.config)