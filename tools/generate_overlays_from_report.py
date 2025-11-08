# tools/generate_overlays_from_report.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import cv2
import numpy as np

from qc.dip import preprocess, segment_defects, mask_to_boxes
from qc.report import overlay_and_save
from qc.cli import load_cfg


def best_by_score(cands):
    """รับลิสต์ (score, payload) แล้วคืน payload ที่ score สูงสุด"""
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]


# ---------------- Hough วงกลม (ใช้ในโหมด --circle แบบ ROI) ----------------
def pick_best_circle(gray, min_r, max_r, dp, p1, p2):
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=12,
        param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r
    )
    cands = []
    if circles is not None:
        h, w = gray.shape
        for cx, cy, r in np.uint16(np.around(circles[0, :])):
            if cx - r < 6 or cy - r < 6 or cx + r > w - 6 or cy + r > h - 6:
                continue

            rr = max(int(r), 3)
            inner = gray[cy - rr // 2:cy + rr // 2, cx - rr // 2:cx + rr // 2]
            outer = gray[max(0, cy - 2 * rr):min(h, cy + 2 * rr),
                         max(0, cx - 2 * rr):min(w, cx + 2 * rr)]
            if inner.size == 0 or outer.size == 0:
                continue

            score = float(abs(outer.mean() - inner.mean()))

            pad = max(2, rr // 2)
            x1, y1 = cx - rr - pad, cy - rr - pad
            x2, y2 = cx + rr + pad, cy + rr + pad

            x1 = max(0, min(x1, w - 2)); x2 = max(1, min(x2, w - 1))
            y1 = max(0, min(y1, h - 2)); y2 = max(1, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            cands.append((score, (int(x1), int(y1), int(x2), int(y2), "spot")))
    return best_by_score(cands)


# ---------------- Hough วงกลมแบบ Global (โหมด --global-circle) ----------------
def pick_best_circle_global(gray, min_r, max_r, dp=1.0, p1=150, p2=16):
    """สแกนทั้งภาพ → ให้คะแนนด้วย ring_score → เลือกวงที่ 'เป็นวง defect' ที่สุดเพียงวงเดียว"""
    H, W = gray.shape
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=8,
        param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return None

    def ring_score(cx, cy, r):
        r = int(r)
        if r < 3:
            return -1e9
        pad = max(2, r // 2)
        x1, y1 = max(0, cx - r - pad), max(0, cy - r - pad)
        x2, y2 = min(W - 1, cx + r + pad), min(H - 1, cy + r + pad)
        if x2 <= x1 or y2 <= y1:
            return -1e9
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return -1e9

        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        ccx, ccy = cx - x1, cy - y1
        rr = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2)

        inner = roi[(rr <= 0.5 * r)]
        ring = roi[(rr >= 0.7 * r) & (rr <= 1.2 * r)]
        outer = roi[(rr >= 1.5 * r) & (rr <= 2.0 * r)]
        if inner.size == 0 or ring.size == 0 or outer.size == 0:
            return -1e9

        c1 = float(outer.mean() - ring.mean())  # ring มืดกว่า outer
        c2 = float(inner.mean() - ring.mean())  # inner ควรสว่างกว่า ring
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, 3)
        edge = float((gx * gx + gy * gy)[(rr >= 0.8 * r) & (rr <= 1.2 * r)].mean()) \
            if np.any((rr >= 0.8 * r) & (rr <= 1.2 * r)) else 0.0

        # รูปแบบให้ weight กับความต่างค่า + ขอบ
        return 0.65 * c1 + 0.20 * c2 + 0.15 * edge

    best = None
    for cx, cy, r in np.uint16(np.around(circles[0, :])):
        s = ring_score(int(cx), int(cy), int(r))
        if best is None or s > best[0]:
            best = (s, int(cx), int(cy), int(r))

        if best is None or best[0] <= -1e8:
            return None

        _, cx, cy, r = best
        # ---- กล่องแบบ "แน่นรอบวง" ----
        shrink = 1.05                      # 1.0–1.2 ยิ่งต่ำยิ่งแน่น
        pad    = max(1, int(round(0.10*r)))  # เดิม 0.20*r → 0.10*r
        rx     = int(round(shrink * r))
        ry     = int(round(shrink * r))
        x1, y1 = cx - rx - pad, cy - ry - pad
        x2, y2 = cx + rx + pad, cy + ry + pad

        x1 = max(0, min(x1, W - 2)); x2 = max(1, min(x2, W - 1))
        y1 = max(0, min(y1, H - 2)); y2 = max(1, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return (int(x1), int(y1), int(x2), int(y2), "spot")



# ---------------- SimpleBlobDetector: วงดำเล็ก (โหมด --ring) ----------------
def pick_best_ring_by_blob(gray, min_area, max_area,
                           min_circ, min_inertia, min_convex, center_w=0.2):
    # หา candidate จาก blob "ดำและกลม"
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor = True;      p.blobColor = 0
    p.filterByArea = True;       p.minArea = float(min_area); p.maxArea = float(max_area)
    p.filterByCircularity = True; p.minCircularity = float(min_circ)
    p.filterByInertia = True;    p.minInertiaRatio = float(min_inertia)
    p.filterByConvexity = True;  p.minConvexity = float(min_convex)
    p.minThreshold = 10; p.maxThreshold = 220; p.thresholdStep = 10

    det = cv2.SimpleBlobDetector_create(p)
    kps = det.detect(gray)
    if not kps:
        return None

    H, W = gray.shape
    cx0, cy0 = W / 2.0, H / 2.0

    def ring_score(cx, cy, r):
        r = int(r)
        if r < 3:
            return -1e9
        pad = max(2, r // 2)
        x1, y1 = max(0, cx - r - pad), max(0, cy - r - pad)
        x2, y2 = min(W - 1, cx + r + pad), min(H - 1, cy + r + pad)
        if x2 <= x1 or y2 <= y1:
            return -1e9

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return -1e9

        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        ccx, ccy = cx - x1, cy - y1
        rr = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2)

        inner = roi[(rr <= 0.5 * r)]
        ring = roi[(rr >= 0.7 * r) & (rr <= 1.2 * r)]
        outer = roi[(rr >= 1.5 * r) & (rr <= 2.0 * r)]
        if inner.size == 0 or ring.size == 0 or outer.size == 0:
            return -1e9

        c1 = float(outer.mean() - ring.mean())
        c2 = float(inner.mean() - ring.mean())
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, 3)
        edge = float((gx * gx + gy * gy)[(rr >= 0.8 * r) & (rr <= 1.2 * r)].mean()) \
            if np.any((rr >= 0.8 * r) & (rr <= 1.2 * r)) else 0.0

        # เพิ่มเล็กน้อยให้ใกล้กลางภาพ
        center_pen = 1.0 - (np.hypot(cx - cx0, cy - cy0) / np.hypot(cx0, cy0))
        return 0.6 * c1 + 0.2 * c2 + 0.15 * edge + 0.05 * center_pen

    cands = []
    for k in kps:
        cx = int(round(k.pt[0])); cy = int(round(k.pt[1]))
        r0 = int(max(3, min(15, round(k.size / 2))))

        # refine รอบๆ จุดด้วย Hough เพื่อกันลวงจาก texture
        win = max(24, 4 * r0)
        x1, y1 = max(0, cx - win), max(0, cy - win)
        x2, y2 = min(W - 1, cx + win), min(H - 1, cy + win)
        roi = gray[y1:y2, x1:x2]
        best_local = None
        if roi.size > 0:
            circles = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT, dp=1.0, minDist=8,
                param1=140, param2=16, minRadius=max(3, r0 - 2), maxRadius=min(16, r0 + 4)
            )
            if circles is not None:
                for px, py, rr in np.uint16(np.around(circles[0, :])):
                    cxr, cyr, r = int(px + x1), int(py + y1), int(rr)
                    s = ring_score(cxr, cyr, r)
                    if best_local is None or s > best_local[0]:
                        best_local = (s, cxr, cyr, r)

        if best_local is None:
            s = ring_score(cx, cy, r0)
            best_local = (s, cx, cy, r0)

        s, bx, by, br = best_local
        if s <= -1e8:
            continue

        # ---- กล่องแบบ "แน่นรอบวง" ----
        shrink = 1.10
        pad    = max(1, int(round(0.20*br)))
        rx     = int(round(shrink * br))
        ry     = int(round(shrink * br))
        xx1, yy1 = bx - rx - pad, by - ry - pad
        xx2, yy2 = bx + rx + pad, by + ry + pad

        xx1 = max(0, min(xx1, W - 2)); xx2 = max(1, min(xx2, W - 1))
        yy1 = max(0, min(yy1, H - 2)); yy2 = max(1, min(yy2, H - 1))
        if xx2 <= xx1 or yy2 <= yy1:
            continue

        cands.append((s, (int(xx1), int(yy1), int(xx2), int(yy2), "spot")))


    return best_by_score(cands)

def _scale_boxes_to_original(img_color, ref_gray, typed):
    """
    typed: [(x1,y1,x2,y2,label), ...] พิกัดบน ref_gray
    คืนเป็นพิกัดบน img_color (ภาพต้นฉบับ)
    """
    H0, W0 = img_color.shape[:2]
    Hr, Wr = ref_gray.shape[:2]
    if (H0, W0) == (Hr, Wr):
        return typed

    sx = W0 / float(Wr); sy = H0 / float(Hr)
    out = []
    for t in typed:
        if t is None: 
            continue
        x1, y1, x2, y2, lab = t[:5]
        X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
        X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
        X1 = max(0, min(X1, W0 - 2)); X2 = max(1, min(X2, W0 - 1))
        Y1 = max(0, min(Y1, H0 - 2)); Y2 = max(1, min(Y2, H0 - 1))
        if X2 > X1 and Y2 > Y1:
            out.append((X1, Y1, X2, Y2, lab))
    return out




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out_class9")
    ap.add_argument("--cfg", default="qc.cfg.yaml")

    # โหมดเลือก
    ap.add_argument("--global-circle", action="store_true",
                    help="ฮัฟทั้งภาพ เลือกวงที่เป็น 'ring' จริงที่สุด (แนะนำสำหรับ Class9)")
    ap.add_argument("--ring", action="store_true",
                    help="หา 'วงดำเล็ก' ด้วย SimpleBlobDetector + refine")
    ap.add_argument("--circle", action="store_true",
                    help="Hough circle ธรรมดา")

    # พารามิเตอร์ blob (รัศมี ~4–10 px)
    ap.add_argument("--blob_min_area", type=float, default=40)
    ap.add_argument("--blob_max_area", type=float, default=300)
    ap.add_argument("--blob_min_circularity", type=float, default=0.80)
    ap.add_argument("--blob_min_inertia", type=float, default=0.40)
    ap.add_argument("--blob_min_convexity", type=float, default=0.80)
    ap.add_argument("--blob_center_weight", type=float, default=0.20)

    # พารามิเตอร์ Hough (สำรอง)
    ap.add_argument("--hough_min_r", type=int, default=4)
    ap.add_argument("--hough_max_r", type=int, default=12)
    ap.add_argument("--hough_dp", type=float, default=1.0)
    ap.add_argument("--hough_p1", type=int, default=140)
    ap.add_argument("--hough_p2", type=int, default=24)

    # กล่อง DIP ธรรมดา (fallback)
    ap.add_argument("--min_area_px", type=int, default=None)
    ap.add_argument("--border", type=int, default=24)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rep = out_dir / "report.csv"
    if not rep.exists():
        raise SystemExit(f"not found: {rep}")

    cfg = load_cfg(args.cfg)
    dip_cfg = dict(cfg.get("dip", {}))
    if args.min_area_px is not None:
        dip_cfg["min_area_px"] = int(args.min_area_px)

    df = pd.read_csv(rep)
    fails = df[df["pass"].astype(int) == 0]
    if fails.empty:
        print("No failed images in report.csv")
        return

    ov_dir = out_dir / "overlay"
    ov_dir.mkdir(parents=True, exist_ok=True)

    found = made = 0
    for _, r in fails.iterrows():
        name = r["image"]
        img_path = out_dir / "fail" / name
        if not img_path.exists():
            continue
        found += 1

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gray = preprocess(img, dip_cfg)  # grayscale + enhance

        if args.global_circle:
            best = pick_best_circle_global(gray, min_r=4, max_r=14, dp=1.0, p1=150, p2=16)
            typed = [best] if best is not None else []

        elif args.ring:
            best = pick_best_ring_by_blob(
                gray,
                min_area=args.blob_min_area, max_area=args.blob_max_area,
                min_circ=args.blob_min_circularity,
                min_inertia=args.blob_min_inertia,
                min_convex=args.blob_min_convexity,
                center_w=args.blob_center_weight
            )
            typed = [best] if best is not None else []

        elif args.circle:
            best = pick_best_circle(
                gray,
                min_r=args.hough_min_r, max_r=args.hough_max_r,
                dp=args.hough_dp, p1=args.hough_p1, p2=args.hough_p2
            )
            typed = [best] if best is not None else []

        else:
            # กล่องจาก DIP (ใช้เมื่ออยากดูสัน-แพตช์กว้าง ๆ)
            seg = segment_defects(gray, dip_cfg)
            raw = mask_to_boxes(seg)
            boxes = []
            H, W = gray.shape
            for rb in raw:
                if len(rb) >= 6:
                    x1, y1, x2, y2, lab, _ = rb[:6]
                else:
                    x1, y1, x2, y2, lab = rb
                if x1 <= args.border or y1 <= args.border or x2 >= W - args.border or y2 >= H - args.border:
                    continue
                if (x2 - x1) * (y2 - y1) < dip_cfg.get("min_area_px", 30):
                    continue
                boxes.append((int(x1), int(y1), int(x2), int(y2), lab))
            typed = boxes

        if typed:
            typed = [t for t in typed if t is not None]
            typed_on_orig = _scale_boxes_to_original(img, gray, typed)
            if typed_on_orig:
                overlay_and_save(img, typed_on_orig, str(ov_dir / f"{Path(name).stem}_overlay.png"))
                made += 1

    print(f"[overlay] fails: {len(fails)} | found: {found} | overlays: {made}")
    print(f"[overlay] saved to: {ov_dir}")


if __name__ == "__main__":
    main()
