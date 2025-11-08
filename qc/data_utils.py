from pathlib import Path
from typing import List
import os

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists(): return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)
