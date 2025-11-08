# qc/dip.py (robust)
from typing import Tuple, Dict, Any
import cv2
import numpy as np

def _get(cfg: Dict[str, Any], key: str, default):
    return cfg.get(key, default)

def preprocess(img, cfg: Dict[str, Any]):
    h, w = img.shape[:2]
    max_side = max(h, w)
    resize_max = _get(cfg, "resize_max", 1024)
    if max_side > resize_max:
        scale = resize_max / max_side
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe_clip = _get(cfg, "clahe_clip", 2.0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Bilateral (edge-preserving denoise)
    bilateral_d = int(_get(cfg, "bilateral_d", 5))
    # ใช้ sigma ค่าคงที่ตามโค้ดเดิม
    gray = cv2.bilateralFilter(gray, bilateral_d, 75, 75)

    # Sharpen
    sharpen_amount = float(_get(cfg, "sharpen_amount", 0.6))
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1 + sharpen_amount, blur, -sharpen_amount, 0)
    return sharp

def segment_defects(pre, cfg: Dict[str, Any]):
    # Adaptive threshold (Gaussian)
    blk = int(_get(cfg, "adaptive_block", 31))
    if blk < 3: blk = 3
    if blk % 2 == 0: blk += 1
    C = int(_get(cfg, "adaptive_C", -5))

    th = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blk, C)

    # Morphology
    mo = int(_get(cfg, "morph_open", 3))
    mc = int(_get(cfg, "morph_close", 5))
    mo = max(1, mo)
    mc = max(1, mc)
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mo,mo))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mc,mc))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_o)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_c)

    # Remove tiny blobs
    min_area = int(_get(cfg, "min_area_px", 200))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask = np.zeros_like(th)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels==i] = 255
    return mask

def mask_to_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        boxes.append((x,y,w,h,area,perim,c))
    return boxes
