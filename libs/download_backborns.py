import os
from pathlib import Path
import gdown

BACKBONE_DIR = Path(__file__).absolute().parent.parent / "ckpts"
MODEL_BACKBONE_MAP = {
    "deimv2_dinov3_s_coco": "vitt_distill",
    "deimv2_dinov3_m_coco": "vittplus_distill",
}
BACKBONE_ID_MAP = {
    "vitt_distill": "1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs",      # ViT-Tiny
    "vittplus_distill": "1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt",  # ViT-Tiny+
}

def download_backbone(model_name):
    """
    Google Driveからバックボーンモデルをダウンロードする関数
    
    Args:
        model_name (str): モデル名
    
    Returns:
        bool: ダウンロード成功時はTrue、スキップ時はFalse。
    
    """
    if model_name not in MODEL_BACKBONE_MAP:
        return False
    
    backbone_name = MODEL_BACKBONE_MAP[model_name]
    
    # 保存先ディレクトリを作成
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    
    # バックボーンダウンロード
    output_path = str(BACKBONE_DIR / f"{backbone_name}.pt")

    # Google DriveのダウンロードURL
    url = f"https://drive.google.com/uc?id={BACKBONE_ID_MAP[backbone_name]}"

    # gdownを使用してファイルをダウンロード
    gdown.download(url, str(output_path), quiet=False)
    
    print(f"\nバックボーンダウンロード完了: {output_path}")
    
    return True