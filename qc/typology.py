# qc/typology.py
from typing import List, Tuple

Box = Tuple[int, int, int, int]
TypedBox = Tuple[int, int, int, int, str]

def rough_type_from_boxes(boxes: List[Tuple]) -> List[TypedBox]:
    """
    รับ list ของกล่องที่อาจเป็น 4 ค่า (x,y,w,h) หรือ 5 ค่า (x,y,w,h,t)
    แล้วคืนเป็น (x,y,w,h,t) เสมอ
    """
    typed: List[TypedBox] = []
    for b in boxes:
        # รองรับทั้ง 4 และ 5 ค่า (หรือมากกว่า)
        if len(b) >= 5:
            x, y, w, h = b[:4]
        else:
            x, y, w, h = b  # 4 ค่า

        if w <= 0 or h <= 0:
            continue

        long_edge  = max(w, h)
        short_edge = min(w, h)
        ar = long_edge / (short_edge + 1e-6)
        area = w * h

        # heuristics — เน้นให้เส้นยาวเป็น "scratch"
        if ar >= 2.2 and (short_edge <= 0.45 * long_edge):
            t = "scratch"
        elif ar >= 3.0:
            t = "scratch"
        elif ar < 1.6 and area <= 1200:
            t = "scratch"
        else:
            t = "scratch"

        typed.append((x, y, w, h, t))

    return typed
