import numpy as np
from PIL import Image
from pathlib import Path
from .config import SUPPORT_FORMATS


def read_gray_image(path: Path):
    try:
        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None


def save_gray_image(path: Path, img: np.ndarray):
    Image.fromarray(img).save(path)


def get_all_image_files(folder_path: Path):
    return [
        p for p in folder_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORT_FORMATS
        and p.stat().st_size > 1024
    ]


def parse_score_from_filename(img_path: Path):
    filename = img_path.name
    for part in filename.split("_"):
        if "人工" in part:
            score_str = part.replace("人工", "").split(".")[0]
            return float(score_str)
    raise ValueError(f"Cannot parse score from filename: {filename}")