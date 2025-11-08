# qc/cli.py — ResNet18 + kNN (hybrid / label-only) + DIP (optional on unlabeled) + eval
import argparse, os, csv, shutil, sys
from pathlib import Path

import yaml
import cv2
import numpy as np
from tqdm import tqdm

from .data_utils import list_images, ensure_dir
from .dip import preprocess, segment_defects, mask_to_boxes
from .typology import rough_type_from_boxes
from .report import overlay_and_save, write_csv
from .anomaly import _build_backbone, extract_feature, AnomalyKNN
import joblib
import json as _json


def load_cfg(path="qc.cfg.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def pick_paths(cfg: dict):
    p = cfg.get("paths", {})
    tn  = p.get("train_normal_Test", p.get("train_normal"))
    inf = p.get("infer_input_Test",  p.get("infer_input"))
    out = p.get("out_dir_Test",      p.get("out_dir"))
    lbl = p.get("label_lookup_root_Test", p.get("label_lookup_root", ""))

    from pathlib import Path
    tn  = Path(tn) if tn else None
    inf = Path(inf) if inf else None
    out = Path(out) if out else None
    lbl = Path(lbl) if (lbl and str(lbl).strip()) else None
    return tn, inf, out, lbl


# ============================== BUILD ==============================

def build(args):
    cfg = load_cfg(args.cfg)
    mcfg = cfg["model"]
    train_dir = Path(cfg["paths"]["train_normal"])
    out_dir = Path(cfg["paths"]["out_dir"])
    model_dir = out_dir / "models"
    ensure_dir(model_dir)

    # Backbone
    device = "cpu"
    backbone, feat_dim = _build_backbone(mcfg["backbone"], mcfg["layer"])

    imgs = list_images(train_dir)
    if not imgs:
        print(f"No images found in {train_dir}.")
        return

    feats = []
    print(f"Extracting features from {len(imgs)} normal images...")
    for p in tqdm(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        f = extract_feature(img, backbone, device=device)
        feats.append(f)
    feats = np.stack(feats, axis=0)

    # Fit KNN
    knn = AnomalyKNN(k=mcfg["knn_k"])
    knn.fit(feats)
    knn_path = str(model_dir / "knn.joblib")
    knn.save(knn_path)

    # Save feature stats for quantile
    dists = []
    for f in feats:
        d = knn.score(f)
        dists.append(d)
    dists = np.array(dists)
    q = float(np.quantile(dists, mcfg["threshold_q"]))
    th_json = {
        "mode": mcfg["threshold_mode"],
        "q": mcfg["threshold_q"],
        "q_value": q,
        "abs": mcfg["threshold_abs"],
    }
    (model_dir / "threshold.json").write_text(_json.dumps(th_json, indent=2), encoding="utf-8")
    print("Saved:", knn_path)


def _load_threshold(model_dir: Path, cfg):
    m = model_dir / "threshold.json"
    if m.exists():
        j = _json.loads(m.read_text("utf-8"))
        if j.get("mode", "absolute") == "quantile":
            return float(j.get("q_value", cfg["model"]["threshold_abs"]))
        else:
            return float(j.get("abs", cfg["model"]["threshold_abs"]))
    # fallback
    return float(cfg["model"]["threshold_abs"])


# ============================== INFER ==============================

def infer(args):
    cfg = load_cfg(args.cfg)
    in_dir   = Path(cfg["paths"]["infer_input"])
    out_dir  = Path(cfg["paths"]["out_dir"])
    label_root = Path(cfg["paths"].get("label_lookup_root","")) if cfg["paths"].get("label_lookup_root") else None

    infer_cfg = cfg.get("infer", {})
    mode = infer_cfg.get("mode", "hybrid").lower()

    # unlabeled options
    use_seg_unl   = bool(infer_cfg.get("use_seg_on_unlabeled", False))
    save_ov_unl   = bool(infer_cfg.get("save_overlay_on_unlabeled", False))
    req_seg_unl   = bool(infer_cfg.get("require_seg_for_unlabeled_fail", False))
    min_boxes_unl = int(infer_cfg.get("min_boxes_unlabeled", 1))
    min_area_unl  = int(infer_cfg.get("min_area_px_unlabeled", 0))

    label_exts = set(e.lower() for e in infer_cfg.get("label_exts", [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]))

    # ---------- สร้าง index ของ label: stem(ปกติ) -> path ของ mask ----------
    def norm_stem(s: str) -> str:
        s = s.lower()
        if s.endswith("_label"):
            return s[:-6]
        return s

    label_map = {}
    if label_root and label_root.exists():
        for lp in label_root.rglob("*"):
            if lp.is_file() and lp.suffix.lower() in label_exts:
                label_map[norm_stem(lp.stem)] = lp
        print(f"[label] indexed {len(label_map)} label files from: {label_root}")

    def find_label_for(img_path: Path):
        return label_map.get(img_path.stem.lower(), None)

    # ---------- เตรียมโฟลเดอร์ผลลัพธ์ ----------
    pass_dir = out_dir / "pass"; fail_dir = out_dir / "fail"; ov_dir = out_dir / "overlay"
    ensure_dir(pass_dir); ensure_dir(fail_dir); ensure_dir(ov_dir)

    imgs = list_images(in_dir)
    if not imgs:
        print(f"No images found in {in_dir}."); return

    rows = []

    # เตรียมของสำหรับ hybrid (ถ้าใช้)
    use_hybrid = (mode != "label_only")
    if use_hybrid:
        model_dir = out_dir / "models"
        knn_path = model_dir / "knn.joblib"
        if not knn_path.exists():
            print("[!] Model not found:", knn_path)
            print("    Run:  python -m qc.cli build --cfg qc.cfg.yaml")
            return
        knn = AnomalyKNN.load(str(knn_path))
        thr = _load_threshold(model_dir, cfg)
        device = "cpu"
        backbone, _ = _build_backbone(cfg["model"]["backbone"], cfg["model"]["layer"])

    gt_fail = mdl_fail = 0

    for p in tqdm(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        mask_path = find_label_for(p)

        # ---------- มี label: เชื่อ label 100% ----------
        if mask_path is not None:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            _, mbin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
            boxes = mask_to_boxes(mbin)
            typed = rough_type_from_boxes(boxes)
            is_def = True
            reason = "gt_label"
            score = 0.0
            gt_fail += 1

            if cfg["report"]["save_overlay"] and len(typed) > 0:
                overlay_and_save(img, typed, str(ov_dir / (p.stem + "_overlay.png")))

        # ---------- ไม่มี label ----------
        else:
            if not use_hybrid:
                # label_only โหมด: ไม่มี label = ผ่าน
                is_def = False
                reason = "normal"
                score = 0.0
                typed = []
            else:
                # 1) ฟีเจอร์ + สกอร์
                f = extract_feature(img, backbone, device=device)
                score = knn.score(f)

                typed = []
                boxes = []
                # 2) DIP เฉพาะเมื่อเปิดใช้งานบน unlabeled
                if use_seg_unl:
                    pre = preprocess(img, cfg["dip"])
                    mseg = segment_defects(pre, cfg["dip"])
                    raw_boxes = mask_to_boxes(mseg)
                    # กรองกล่องเล็ก ๆ
                    boxes = [b for b in raw_boxes if (b[2]*b[3]) >= min_area_unl]
                    typed = rough_type_from_boxes(boxes)

                # 3) ตัดสินใจ
                if req_seg_unl and use_seg_unl:
                    # ต้องได้ทั้ง score สูง + DIP ยืนยัน (≥ min_boxes_unl)
                    cond_anom = (score >= thr)
                    cond_seg  = (len(boxes) >= min_boxes_unl)
                    is_def = cond_anom and cond_seg
                    if cond_anom and cond_seg:
                        reason = "anomaly+seg"
                    elif cond_anom:
                        reason = "anomaly_but_no_seg"
                    elif cond_seg:
                        reason = "seg_only"
                    else:
                        reason = "normal"
                else:
                    if use_seg_unl:
                        is_def = (score >= thr) or (len(typed) > 0)
                        reason = "anomaly" if (score >= thr) else ("segmentation" if len(typed) > 0 else "normal")
                    else:
                        is_def = (score >= thr)
                        reason = "anomaly" if (score >= thr) else "normal"

                if is_def: mdl_fail += 1

                # overlay สำหรับ unlabeled: วาดเฉพาะเมื่ออนุญาต + มีกรอบจริง
                if cfg["report"]["save_overlay"] and save_ov_unl and is_def and len(typed) > 0:
                    overlay_and_save(img, typed, str(ov_dir / (p.stem + "_overlay.png")))

        pass_flag = int(not is_def)
        types_join = ";".join([(tb[4] if len(tb) >= 5 else "") for tb in typed if len(tb) >= 5])
        rows.append([p.name, pass_flag, reason, f"{score:.4f}", len(typed), types_join])

        if cfg["report"]["move_pass_fail"]:
            dst = pass_dir if pass_flag == 1 else fail_dir
            ensure_dir(dst)
            shutil.copy2(str(p), str(dst / p.name))

    write_csv(rows, str(out_dir / "report.csv"))
    print("Saved CSV:", out_dir / "report.csv")
    print(f"[summary] GT-fail (label present): {gt_fail}, model-based fail (no label): {mdl_fail}, total images: {len(imgs)}")


# ============================== EVAL ==============================

def eval_cmd(args):
    cfg = load_cfg(args.cfg)
    dagm_root = Path(cfg["paths"]["dagm_root"])
    out_dir = Path(cfg["paths"]["out_dir"])
    model_dir = out_dir / "models"
    knn = AnomalyKNN.load(str(model_dir / "knn.joblib"))
    thr = _load_threshold(model_dir, cfg)

    # label lookup (ถ้ามี)
    label_root = Path(cfg["paths"].get("label_lookup_root","")) if cfg["paths"].get("label_lookup_root") else None
    label_exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

    def norm_stem(s: str) -> str:
        s = s.lower()
        if s.endswith("_label"):
            return s[:-6]
        return s

    label_stems = set()
    if label_root and label_root.exists():
        for lp in label_root.rglob("*"):
            if lp.is_file() and lp.suffix.lower() in label_exts:
                label_stems.add(norm_stem(lp.stem))

    # Backbone
    device = "cpu"
    backbone, _ = _build_backbone(cfg["model"]["backbone"], cfg["model"]["layer"])

    # Collect image list + GT (มี label = defect)
    imgs, y_true = [], []
    for imp in dagm_root.rglob("*"):
        if not (imp.is_file() and imp.suffix.lower() in label_exts):
            continue
        if "_label" in imp.stem:
            continue

        if label_stems:
            has_mask = (norm_stem(imp.stem) in label_stems)
        else:
            # fallback: sibling *_label.*
            has_mask = len(list(imp.parent.glob(imp.stem + "_label.*"))) > 0

        imgs.append(imp)
        y_true.append(1 if has_mask else 0)

    if not imgs:
        print("No images found under dagm_root. Skipping eval.")
        return

    y_pred = []
    for p in tqdm(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        pre = preprocess(img, cfg["dip"])
        mask = segment_defects(pre, cfg["dip"])
        boxes = mask_to_boxes(mask)
        f = extract_feature(img, backbone, device=device)
        score = knn.score(f)
        # สำหรับการประเมินรวม: อนุญาตให้ DIP ช่วย (คงพฤติกรรมเดิม)
        is_def = (score >= thr) or (len(boxes) > 0)
        y_pred.append(1 if is_def else 0)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = int(((y_pred==1) & (y_true==1)).sum())
    tn = int(((y_pred==0) & (y_true==0)).sum())
    fp = int(((y_pred==1) & (y_true==0)).sum())
    fn = int(((y_pred==0) & (y_true==1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    print("=== Evaluation (image-level) ===")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  Accuracy: {acc:.4f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")


# ============================== CLI ==============================

def main():
    ap = argparse.ArgumentParser(description="QC / Anomaly pipeline CLI")
    ap.set_defaults(func=lambda _: ap.print_help())
    sp = ap.add_subparsers()

    ap_build = sp.add_parser("build", help="Build anomaly index from normal images")
    ap_build.add_argument("--cfg", type=str, default="qc.cfg.yaml")
    ap_build.set_defaults(func=build)

    ap_infer = sp.add_parser("infer", help="Run inference on mixed images")
    ap_infer.add_argument("--cfg", type=str, default="qc.cfg.yaml")
    ap_infer.set_defaults(func=infer)

    ap_eval = sp.add_parser("eval", help="Evaluate on DAGM test set (image-level)")
    ap_eval.add_argument("--cfg", type=str, default="qc.cfg.yaml")
    ap_eval.set_defaults(func=eval_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
