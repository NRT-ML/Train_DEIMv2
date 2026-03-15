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
    Function to download backbone model from Google Drive.
    
    Args:
        model_name (str): Model name
    
    Returns:
        bool: True on successful download, False if skipped.
    
    """
    if model_name not in MODEL_BACKBONE_MAP:
        return False
    
    backbone_name = MODEL_BACKBONE_MAP[model_name]
    
    # Create output directory
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download backbone
    output_path = str(BACKBONE_DIR / f"{backbone_name}.pt")

    # Google Drive download URL
    url = f"https://drive.google.com/uc?id={BACKBONE_ID_MAP[backbone_name]}"

    # Download file using gdown
    gdown.download(url, str(output_path), quiet=False)
    
    print(f"\nBackbone download complete: {output_path}")
    
    return True