from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ทำให้มองเห็นแพ็กเกจ qc/ แม้รันจาก tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc.data_utils import list_images, ensure_dir
from qc.anomaly import _build_backbone, extract_feature, AnomalyKNN

# โฟลเดอร์ที่ถือว่าเป็น label/mask เสมอ
SKIP_DIRS = {"label", "labels", "mask", "masks", "gt", "groundtruth"}


def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick_paths(cfg: dict) -> tuple[Path, Path, Path]:
    """เลือกเส้นทาง; ถ้ามี *_Test ให้ใช้เพื่อกันปนกับงานชุดเก่า"""
    p = cfg.get("paths", {}) or {}
    in_dir = Path(p.get("infer_input_Test", p.get("infer_input", "data/infer_input")))
    out_dir = Path(p.get("out_dir_Test", p.get("out_dir", "out")))
    tn_dir = Path(p.get("train_normal_Test", p.get("train_normal", "data/train_normal")))
    return in_dir, out_dir, tn_dir


def main():
    ap = argparse.ArgumentParser(description="Bootstrap build from unlabeled images (no mixing with old data)")
    ap.add_argument("--cfg", default="qc.cfg.yaml", help="path to cfg yaml (เช่น qc.cfg.quick.yaml)")
    ap.add_argument("--keep", type=float, default=0.70,
                    help="ส่วนล่างของระยะ (ใกล้ค่าเฉลี่ยที่สุด) ที่จะเก็บเป็น normal [0..1], ใส่ 1.0 เพื่อเก็บทุกรูป")
    ap.add_argument("--mode", choices=["absolute", "quantile"], default=None,
                    help="บังคับโหมด threshold ใน threshold.json (ถ้าไม่ใส่จะอ่านจาก cfg หรือ fallback=absolute)")
    ap.add_argument("--abs", dest="abs_thr", type=float, default=None,
                    help="ค่า threshold_abs เมื่อ mode=absolute (fallback ถ้า cfg ไม่มี)")
    ap.add_argument("--q", dest="q_thr", type=float, default=None,
                    help="ค่า threshold_q เมื่อ mode=quantile (fallback ถ้า cfg ไม่มี)")
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    cfg = load_cfg(cfg_path)
    in_dir, out_dir, tn_dir = pick_paths(cfg)

    # เตรียมโฟลเดอร์ out_dir/models และ train_normal ให้สะอาด
    model_dir = out_dir / "models"
    ensure_dir(out_dir)
    ensure_dir(model_dir)

    if tn_dir.exists():
        shutil.rmtree(tn_dir)
    tn_dir.mkdir(parents=True, exist_ok=True)

    # -------- เตรียมโมเดล backbone --------
    mcfg = cfg.get("model", {}) or {}
    backbone_name = mcfg.get("backbone", "resnet18")
    layer = mcfg.get("layer", "avgpool")
    knn_k = int(mcfg.get("knn_k", 5))

    # threshold defaults (อ่านจาก cfg ก่อน ถ้า CLI ไม่ override)
    mode = args.mode or mcfg.get("threshold_mode", "absolute")
    q_cfg = float(args.q_thr if args.q_thr is not None else mcfg.get("threshold_q", 0.95))
    abs_cfg = float(args.abs_thr if args.abs_thr is not None else mcfg.get("threshold_abs", 0.60))

    device = "cpu"
    backbone, feat_dim = _build_backbone(backbone_name, layer)

    # -------- อ่านรูป unlabeled (กรอง *_label + โฟลเดอร์ label/mask/gt) --------
    imgs_all = list_images(in_dir)
    imgs = []
    for p in imgs_all:
        # ข้ามถ้าชื่อไฟล์บอกว่าเป็น label
        if "_label" in p.stem.lower():
            continue
        # ข้ามถ้าอยู่ใต้โฟลเดอร์ที่ถือว่าเป็น label/mask/gt
        parts = {s.lower() for s in p.parts}
        if parts & SKIP_DIRS:
            continue
        imgs.append(p)

    print(f"[bootstrap] scan under: {in_dir}")
    print(f"[bootstrap] found {len(imgs_all)} files, using {len(imgs)} after cleaning (*_label + label/mask dirs)")

    if not imgs:
        raise SystemExit(f"No images found after cleaning under: {in_dir}")

    # -------- Extract features --------
    feats = []
    print(f"[bootstrap] extracting features from {len(imgs)} unlabeled images ...")
    for p in tqdm(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        feats.append(extract_feature(img, backbone, device=device))

    if not feats:
        raise SystemExit("Failed to extract any features.")

    X = np.stack(feats, axis=0)  # (N, D)

    # -------- เลือก normal ชั่วคราว: ระยะจากค่าเฉลี่ย (ยิ่งใกล้ = ปกติ) --------
    mu = X.mean(axis=0)
    d = np.linalg.norm(X - mu, axis=1)

    keep = float(args.keep)
    # อนุญาต 1.00 ได้ และกันกรณีค่าแปลกนิดหน่อย
    if not (0.0 < keep <= 1.0):
        raise SystemExit("--keep ต้องอยู่ในช่วง (0, 1]")

    if keep >= 1.0:
        # เก็บทุกรูปเป็น normal ชั่วคราว
        kept_idx = np.arange(len(d))
        thr = float(d.max())
    else:
        thr = float(np.quantile(d, keep))
        kept_idx = np.where(d <= thr)[0]
    kept = [imgs[i] for i in kept_idx]
    print(f"[bootstrap] keep {len(kept)}/{len(imgs)} ({keep*100:.0f}%) as provisional normals")

    # คัดลอก normal ที่คัดเลือกไป train_normal
    for p_src in kept:
        shutil.copy2(str(p_src), str(tn_dir / p_src.name))

    # -------- สร้าง KNN และบันทึก --------
    knn = AnomalyKNN(k=knn_k)
    knn.fit(X[kept_idx])
    knn_path = str(model_dir / "knn.joblib")
    knn.save(knn_path)

    # -------- เขียน threshold.json --------
    dists = np.array([knn.score(f) for f in X[kept_idx]])
    q_value = float(np.quantile(dists, q_cfg))  # ไว้ใช้ถ้าเลือกโหมด quantile

    th_json = {
        "mode": mode,
        "q": q_cfg,
        "q_value": q_value,
        "abs": abs_cfg,
        "bootstrap_keep_percent": keep,
        "n_train": int(len(kept)),
        "paths_used": {
            "infer_input": str(in_dir),
            "out_dir": str(out_dir),
            "train_normal": str(tn_dir),
        },
        "backbone": backbone_name,
        "layer": layer,
        "knn_k": knn_k,
    }
    (model_dir / "threshold.json").write_text(json.dumps(th_json, indent=2), encoding="utf-8")

    # บันทึกชื่อไฟล์ที่ถูกเลือกเป็น normal
    sel_list = out_dir / "bootstrap_selected_normals.txt"
    sel_list.write_text("\n".join([p.name for p in kept]), encoding="utf-8")

    print("[bootstrap] saved model:", knn_path)
    print("[bootstrap] wrote:", sel_list)
    print("[bootstrap] threshold.json:", (model_dir / "threshold.json"))
    print(f"[bootstrap] paths_used: in={in_dir}  out={out_dir}  train_normal={tn_dir}")


if __name__ == "__main__":
    main()
