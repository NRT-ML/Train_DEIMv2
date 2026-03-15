import os
from pathlib import Path
import gdown

PRETRAINED_DIR = Path(__file__).absolute().parent.parent / "pretrained"
PRETRAINED_ID_MAP = {
    # HGNetv2-based models
    "deimv2_hgnetv2_atto_coco": "18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc",
    "deimv2_hgnetv2_femto_coco": "16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds",
    "deimv2_hgnetv2_pico_coco": "1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT",
    "deimv2_hgnetv2_n_coco": "1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K",
    
    # DINOv3-based models
    "deimv2_dinov3_s_coco": "1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL",
    "deimv2_dinov3_m_coco": "1nPKDHrotusQ748O1cQXJfi5wdShq6bKp",
}

def download_pretrained(model):
    """
    Function to download files from Google Drive.
    
    Args:
        model (str): Pretrained model name
    
    Returns:
        output_path (str): Downloaded file path.
    
    """
    if model not in PRETRAINED_ID_MAP:
        raise ValueError(f"Pretrained model not found: {model}")
    
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    file_id = PRETRAINED_ID_MAP[model]
    output_path = PRETRAINED_DIR / f"{model}.pth"

    # Google Drive download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Starting download of pretrained model: {model}")
    
    # Download file using gdown
    gdown.download(url, str(output_path), quiet=False)
    print(f"\nDownload complete: {output_path}")

    return output_path
    