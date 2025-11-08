# ui/app.py
import io, json, shutil, subprocess, sys, time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

APP_TITLE = "QC Anomaly — Workspace"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {"label", "labels", "mask", "masks", "gt", "groundtruth", "overlay", "models"}

# ล็อก Label path ไว้ที่เดียว
LABEL_LOCK_PATH = Path("data/DAGM_2007/DAGM_KaggleUpload/Class10/Test/Label")

# =============== Helpers ===============

def list_images(root: Path):
    root = Path(root)
    if not root.exists(): return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def run(cmd, live_panel=None):
    buf = io.StringIO()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             bufsize=1, universal_newlines=True)
    except Exception as e:
        msg = f"รันคำสั่งไม่สำเร็จ: {' '.join(map(str, cmd))}\n{e}"
        if live_panel: live_panel.error(msg)
        return 1, msg
    while True:
        line = p.stdout.readline()
        if not line:
            if p.poll() is not None: break
            time.sleep(0.02); continue
        buf.write(line)
        if live_panel: live_panel.write(line)
    rc = p.wait()
    return rc, buf.getvalue()

def clean_copy_without_labels(src_dir: Path, dst_dir: Path):
    """คัดลอกเฉพาะรูปจาก src -> dst ตัด *_label และโฟลเดอร์ label/mask/gt/out_*"""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if dst_dir.exists(): shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists(): raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {src_dir}")

    total = copied = s1 = s2 = s3 = 0
    OUT_PREFIX = ("out_", "outui_", "outui", "out")
    for p in list_images(src_dir):
        total += 1
        parts = [s.lower() for s in p.parts]
        if any(part.startswith(OUT_PREFIX) for part in parts): s3 += 1; continue
        if "_label" in p.stem.lower(): s1 += 1; continue
        if set(parts) & SKIP_DIRS:     s2 += 1; continue
        try:
            rel = p.relative_to(src_dir)
        except Exception:
            rel = Path(p.name)
        outp = dst_dir / rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, outp); copied += 1

    st.info(f"[clean] total={total} | kept={copied} | drop *_label={s1} | drop label/mask dir={s2} | drop out_*={s3}")
    return copied

def ensure_threshold_json(out_dir: Path, default_abs: float = 0.13):
    out_dir = Path(out_dir); mdir = out_dir / "models"; mdir.mkdir(parents=True, exist_ok=True)
    th = mdir / "threshold.json"
    if not th.exists():
        th.write_text(json.dumps({"mode":"absolute","abs":float(default_abs),"q":0.95}, indent=2), encoding="utf-8")
    return th

def ensure_models_for(out_live: Path, model_root: Path):
    src = Path(model_root) / "models"; dst = Path(out_live) / "models"; dst.mkdir(parents=True, exist_ok=True)
    for name in ["knn.joblib", "threshold.json"]:
        p = src / name
        if p.exists(): shutil.copy2(p, dst / name)

def write_bootstrap_cfg(cfg_path: Path, infer_input: Path, out_dir: Path):
    import yaml as _y
    cfg = {
        "paths": {
            "infer_input_Test": str(Path(infer_input)),
            "out_dir_Test": str(Path(out_dir)),
            "train_normal_Test": str(Path(out_dir) / "train_normal"),
        },
        "model": {
            "backbone": "resnet18", "layer": "avgpool", "knn_k": 5, "feature_size": 512,
            "threshold_mode": "absolute", "threshold_abs": 0.13, "threshold_q": 0.95
        }
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    _y.safe_dump(cfg, open(cfg_path,"w",encoding="utf-8"), sort_keys=False, allow_unicode=True)
    return cfg_path

# ---------------------- NEW: quick cfg with DIP ----------------------
def write_quick_cfg(
    cfg_path: Path,
    infer_input: Path,
    out_dir: Path,
    threshold_abs: float,
    label_root: Path | None = None,
    infer_mode: str = "hybrid",

    # ----- DIP controls -----
    use_dip_on_unlabeled: bool = True,
    require_seg_for_unlabeled_fail: bool = True,
    min_boxes_unlabeled: int = 1,
    min_area_px_unlabeled: int = 8000,   # ↑ คัดก้อนใหญ่เท่านั้น
    dip_params: dict | None = None,

    # report/overlay
    save_overlay: bool = False,
):
    """
    เขียน qc.cfg.quick.yaml สำหรับ infer (มี DIP tuning)
    """
    import yaml as _y

    # ----- DIP defaults ที่ปลอดภัยกว่า -----
    dip_defaults = {
        "resize_max": 1024,
        "clahe_clip": 2.0,
        "bilateral_d": 3,          # ↓ เบาลงเพื่อลดขอบเทียม
        "sharpen_amount": 0.30,    # ↓ ลด sharpen กัน noise
        "adaptive_block": 51,      # ↑ ใหญ่ขึ้น ตัด noise เล็กได้ดีขึ้น (ต้องเป็นเลขคี่)
        "adaptive_C": -7,          # ↓ เข้มขึ้นเล็กน้อย
        "morph_open": 3,
        "morph_close": 7,
        "min_area_px": int(min_area_px_unlabeled),
    }

    # allow alias -> actual keys
    def normalize_keys(d: dict) -> dict:
        alias = {
            "sharpen_strength": "sharpen_amount",
            "resize": "resize_max",
            "min_area_px_unlabeled": "min_area_px",
        }
        out = {}
        for k, v in d.items():
            out[alias.get(k, k)] = v
        return out

    dip_cfg = dip_defaults.copy()
    if dip_params:
        dip_cfg.update(normalize_keys(dip_params))

    # บังคับให้ adaptive_block เป็นเลขคี่ >= 3
    blk = int(dip_cfg.get("adaptive_block", 51))
    if blk < 3:
        blk = 3
    if blk % 2 == 0:
        blk += 1
    dip_cfg["adaptive_block"] = blk

    cfg = {
        "paths": {
            "train_normal": "__dummy_not_used__",
            "infer_input": str(Path(infer_input)),
            "out_dir": str(Path(out_dir)),
        },
        "infer": {
            "mode": infer_mode,
            "use_seg_on_unlabeled": bool(use_dip_on_unlabeled),
            "save_overlay_on_unlabeled": True,
            "require_seg_for_unlabeled_fail": bool(require_seg_for_unlabeled_fail),
            "min_boxes_unlabeled": int(min_boxes_unlabeled),
            "min_area_px_unlabeled": int(min_area_px_unlabeled),
        },
        "dip": dip_cfg,  # << ส่งให้ qc/dip.py ใช้จริง
        "report": {
            "save_overlay": bool(save_overlay),
            "move_pass_fail": True
        },
        "model": {
            "backbone": "resnet18",
            "layer": "avgpool",
            "knn_k": 5,
            "feature_size": 512,
            "threshold_mode": "absolute",
            "threshold_abs": float(threshold_abs),
            "threshold_q": 0.95,
        },
    }
    if label_root:
        cfg["paths"]["label_lookup_root"] = str(Path(label_root))

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    _y.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"),
                 sort_keys=False, allow_unicode=True)
    return cfg_path
# ---------------------------------------------------------------------

def read_report_flexible(out_live: Path):
    out_live = Path(out_live)
    for name in ["report.csv", "anomaly_raw.csv"]:
        p = out_live / name
        if p.exists():
            try: return pd.read_csv(p), p
            except Exception: pass
    rows = []
    for p in (out_live/"pass").glob("*"):
        if p.is_file(): rows.append({"image":str(p),"pass":True})
    for p in (out_live/"fail").glob("*"):
        if p.is_file(): rows.append({"image":str(p),"pass":False})
    if rows: return pd.DataFrame(rows), None
    return None, None

def chunk(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i+n]

def scan_out_dirs(roots: list[str]) -> list[Path]:
    found = []
    for r in roots:
        rp = Path(r)
        if not rp.exists(): continue
        for p in rp.rglob("report.csv"):
            found.append(p.parent)
    return sorted(set(found))

def load_many_reports(out_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for od in out_dirs:
        df, _ = read_report_flexible(od)
        if df is None: continue
        df = df.copy()
        df["out_dir"] = str(od)
        frames.append(df)
    if not frames: return pd.DataFrame()
    cols = list({c for f in frames for c in f.columns})
    return pd.concat(frames, ignore_index=True)[cols]

# =============== Page ===============

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

tab1, tab2, tab3 = st.tabs([
    "QC Anomaly — Quick Test (3-step)",
    "QC Anomaly",
    "All Results (รวมผลหลาย out)"
])

# ---------------- Tab 1: Quick Test ----------------
with tab1:
    st.header("Step 1: สร้างโมเดล")
    ss = st.session_state
    ss.setdefault("train_src", "data_train_mixed")
    ss.setdefault("model_out", "out_class10")
    ss.setdefault("keep_ratio", 0.70)
    ss.setdefault("thr_abs", 0.13)

    c1, c2 = st.columns([3,3])
    with c1: train_src = st.text_input("พาธโฟลเดอร์รูป (train source)", value=ss["train_src"])
    with c2: model_out = st.text_input("พาธ <out_dir> สำหรับเก็บโมเดล", value=ss["model_out"])

    c3, c4 = st.columns([2,2])
    with c3: keep_ratio = st.slider("สัดส่วนเก็บเป็น normal (bootstrap keep)", 0.50, 1.00, float(ss["keep_ratio"]), 0.05)
    with c4: thr_abs = st.number_input("Absolute threshold (ไว้ใช้ตอน infer)", min_value=0.0, step=0.01, value=float(ss["thr_abs"]), format="%.4f")

    if st.button("สร้างโมเดล (clean + bootstrap)", type="primary"):
        tool = Path("tools") / "bootstrap_build_from_unlabeled.py"
        if not tool.exists(): st.error(f"ไม่พบสคริปต์: {tool}")
        else:
            train_src_p = Path(train_src.strip()); out_dir = Path(model_out.strip())
            out_dir.mkdir(parents=True, exist_ok=True)
            ensure_threshold_json(out_dir, default_abs=thr_abs)

            tmp_clean = out_dir / "bootstrap_clean"
            copied = clean_copy_without_labels(train_src_p, tmp_clean)
            st.info(f"[clean] → {tmp_clean} (copied={copied})")

            cfg_boot = out_dir / "qc.cfg.bootstrap.yaml"
            write_bootstrap_cfg(cfg_boot, infer_input=tmp_clean, out_dir=out_dir)

            cmd = [sys.executable, str(tool), "--cfg", str(cfg_boot), "--keep", str(keep_ratio),
                   "--mode", "absolute", "--abs", str(thr_abs)]
            st.caption("Run: " + " ".join(cmd))
            log = st.empty(); log.write("---- LOG (bootstrap) ----")
            rc, _ = run(cmd, live_panel=log)
            if rc == 0 and (out_dir/"models/knn.joblib").exists():
                st.success(f"โมเดลถูกบันทึกที่: {out_dir/'models/knn.joblib'}")
                ss["model_out"] = str(out_dir); ss["keep_ratio"] = keep_ratio; ss["thr_abs"] = float(thr_abs)
            else:
                st.error("สร้างโมเดลไม่สำเร็จ — ตรวจ LOG")

    st.divider()
    st.header("Step 2: เลือกโฟลเดอร์ทดสอบ → รัน Infer")
    ss.setdefault("test_src", "data_test_mixed")
    ss.setdefault("out_live", "out_ui/quick_run")
    c21, c22 = st.columns([3,3])
    with c21: test_src = st.text_input("พาธโฟลเดอร์ทดสอบ (raw)", value=ss["test_src"])
    with c22: out_live = st.text_input("พาธผลลัพธ์รอบนี้ (<out_live>)", value=ss["out_live"])
    quick_clean = Path("data_ui/quick_clean")
    if st.button("เตรียมชุดทดสอบ (คัดลอกไป data_ui/quick_clean และตัด *_label)", type="secondary"):
        try:
            copied = clean_copy_without_labels(Path(test_src), quick_clean)
            st.success(f"คัดลอก {copied} ไฟล์ → {quick_clean}")
        except Exception as e:
            st.error(f"เตรียมชุดทดสอบไม่สำเร็จ: {e}")

    thr_abs_ui = st.number_input("Absolute threshold (ใช้รอบ infer นี้)", min_value=0.0, step=0.01,
                                 value=float(ss.get("thr_abs", 0.13)), format="%.4f")
    if st.button("Run Infer (ใช้โมเดลจาก Step 1)", type="primary"):
        model_dir = Path(model_out)
        if not (model_dir/"models/knn.joblib").exists():
            st.error("ไม่พบโมเดล — กลับไป Step 1 ก่อน")
        elif not quick_clean.exists() or not list_images(quick_clean):
            st.error("ยังไม่มีรูปใน data_ui/quick_clean — กดเตรียมชุดทดสอบก่อน")
        else:
            out_live_dir = Path(out_live); out_live_dir.mkdir(parents=True, exist_ok=True)
            ensure_models_for(out_live_dir, model_dir)
            cfg_path = out_live_dir / "qc.cfg.quick.yaml"

            # ใช้ DIP แบบ OR + ค่าที่ไม่แรงเกินไป
            write_quick_cfg(
            cfg_path=cfg_path,
            infer_input=quick_clean,
            out_dir=out_live_dir,
            threshold_abs=thr_abs_ui,          # แนะนำเริ่ม 0.20
            label_root=None,
            infer_mode="hybrid",

            # DIP เปิดใช้งาน แต่ “คัดก้อนใหญ่”
            use_dip_on_unlabeled=True,
            require_seg_for_unlabeled_fail=True,
            min_boxes_unlabeled=1,
            min_area_px_unlabeled=8000,        # ลอง 8000–12000 ตามขนาดภาพ

            # จะส่งค่าทับ defaults ได้ที่นี่
            dip_params={
                "adaptive_block": 51,
                "adaptive_C": -7,
                "bilateral_d": 3,
                "sharpen_amount": 0.30,
                "morph_open": 3,
                "morph_close": 7,
            },

            save_overlay=True,
        )
            cmd = [sys.executable, "-m", "qc.cli", "infer", "--cfg", str(cfg_path)]
            st.caption("Run: " + " ".join(cmd))
            log = st.empty(); log.write("---- LOG (infer) ----")
            rc, _ = run(cmd, live_panel=log)
            if rc == 0:
                st.success(f"Infer เสร็จสิ้น → {out_live_dir}")
                ss["out_live"] = str(out_live_dir); ss["thr_abs"] = float(thr_abs_ui)
            else:
                st.error("Infer ล้มเหลว — ตรวจ LOG")

    # ---------- Step 3: ผลลัพธ์ & รายงาน (แสดงรูปด้วย) ----------
    st.divider()
    st.header("Step 3: ผลลัพธ์ & รายงาน")

    current_out = st.session_state.get("out_live", out_live)
    df, used = read_report_flexible(current_out)

    if df is None:
        st.warning("ยังไม่พบรายงาน")
    else:
        # --- สรุปตัวเลข
        def to_bool(x): return (str(x).strip().lower() in {"1","true","yes","y","pass","passed"})
        total = len(df)
        passed = sum(to_bool(v) for v in df["pass"]) if "pass" in df.columns else None
        failed = total - passed if passed is not None else None

        m1, m2, m3 = st.columns(3)
        m1.metric("Images", total)
        if passed is not None:
            m2.metric("Pass", passed)
            m3.metric("Fail", failed)

        # --- ตารางผล
        st.subheader("ตารางผลลัพธ์")
        st.dataframe(df, use_container_width=True)
        if used and used.exists():
            with open(used, "rb") as f:
                st.download_button("ดาวน์โหลด report.csv", f, file_name=used.name, mime="text/csv")

        # ---------- แสดงกริดรูป Fail ----------
        out_live_dir = Path(current_out)
        fail_dir = out_live_dir / "fail"
        fail_imgs = []
        if fail_dir.exists():
            for ext in IMG_EXTS:
                fail_imgs.extend(sorted(fail_dir.glob(f"*{ext}")))
        st.subheader(f"Fail Browser — รูปที่จัดเป็น Fail ({len(fail_imgs)} รูป)")
        if not fail_imgs:
            st.info("ยังไม่มีรูปในโฟลเดอร์ fail/")
        else:
            for row in chunk(fail_imgs, 4):
                cols = st.columns(4)
                for c, p in zip(cols, row):
                    with c:
                        try:
                            st.image(Image.open(p), caption=p.name, use_container_width=True)
                            with st.expander(f"ดูเดี่ยว: {p.name}", expanded=False):
                                st.image(Image.open(p), use_container_width=True)
                        except Exception as e:
                            st.warning(f"เปิดรูปไม่ได้: {p.name} ({e})")

        # ---------- Overlay ----------
        st.subheader("Overlays")
        ov_dir = out_live_dir / "overlay"
        gen_tool = Path("tools") / "generate_overlays_from_report.py"
        c_ov_btn, _ = st.columns([1,3])
        with c_ov_btn:
            if st.button("Generate overlays (global circle)"):
                if not gen_tool.exists():
                    st.error(f"ไม่พบสคริปต์: {gen_tool}")
                else:
                    cmd_ov = [sys.executable, str(gen_tool), "--out_dir", str(out_live_dir), "--global-circle"]
                    log2 = st.empty(); log2.write("---- LOG (overlay) ----")
                    rc2, _ = run(cmd_ov, live_panel=log2)
                    if rc2 == 0:
                        st.success("สร้าง Overlay สำเร็จ")
                    else:
                        st.warning("สร้าง Overlay ไม่สำเร็จ (ดู LOG)")

        # แสดงกริด overlay ถ้ามี
        ov_imgs = []
        if ov_dir.exists():
            for ext in IMG_EXTS:
                ov_imgs.extend(sorted(ov_dir.glob(f"*{ext}")))
        if ov_imgs:
            st.caption(f"Overlay files: {len(ov_imgs)} รูป")
            for row in chunk(ov_imgs, 4):
                cols = st.columns(4)
                for c, p in zip(cols, row):
                    with c:
                        try:
                            st.image(Image.open(p), caption=p.name, use_container_width=True)
                        except Exception as e:
                            st.warning(f"เปิดรูปไม่ได้: {p.name} ({e})")
        else:
            st.info("ยังไม่มีไฟล์ overlay — กดปุ่ม Generate overlays ได้ด้านบน")

# ---------------- Tab 2: Label-only ----------------
with tab2:
    st.header("ตัดสินผลตาม Label")

    ss = st.session_state
    ss.setdefault("label_test_src", "data/DAGM_2007/DAGM_KaggleUpload/Class10/Test")
    ss.setdefault("label_out_live", "out_ui/label_run")

    c1, c2 = st.columns([3,3])
    with c1:
        test_src2 = st.text_input("พาธโฟลเดอร์ทดสอบ (raw)", value=ss["label_test_src"], key="lab_src")
    with c2:
        out_live2  = st.text_input("พาธผลลัพธ์ (<out_live>)", value=ss["label_out_live"], key="lab_out")

    st.caption(f"Label path (locked): {LABEL_LOCK_PATH.resolve()}")
    if not LABEL_LOCK_PATH.exists():
        st.error(f"ไม่พบโฟลเดอร์ Label ที่ล็อกไว้: {LABEL_LOCK_PATH}")
        st.stop()

    quick_clean2 = Path("data_ui/quick_clean_label")

    def ensure_clean_test():
        """ถ้า data_ui/quick_clean_label ยังไม่มีรูป ให้ทำความสะอาด/คัดลอกให้อัตโนมัติ"""
        imgs = list_images(quick_clean2) if quick_clean2.exists() else []
        if imgs:
            return len(imgs)
        copied = clean_copy_without_labels(Path(test_src2), quick_clean2)
        return copied

    if st.button("เตรียมชุดทดสอบ (ตัด *_label ออก)", type="secondary", key="lab_prep"):
        try:
            n = ensure_clean_test()
            st.success(f"พร้อมทดสอบ: {n} รูป ที่ {quick_clean2}")
        except Exception as e:
            st.error(f"เตรียมชุดทดสอบไม่สำเร็จ: {e}")

    if st.button("Run (label_only)", type="primary", key="lab_run"):
        try:
            n = ensure_clean_test()
            if n == 0:
                st.error("ไม่พบนรูปสำหรับทดสอบหลังทำความสะอาด")
                st.stop()

            out_live_dir = Path(out_live2); out_live_dir.mkdir(parents=True, exist_ok=True)

            # เขียน cfg (label_only) + เปิด save_overlay
            cfg_path = out_live_dir / "qc.cfg.quick.yaml"
            write_quick_cfg(
                cfg_path,
                infer_input=quick_clean2,
                out_dir=out_live_dir,
                threshold_abs=0.13,             # ค่า dummy สำหรับ label_only
                label_root=LABEL_LOCK_PATH,     # ใช้ label ที่ล็อกไว้
                infer_mode="label_only",
                save_overlay=True,
                use_dip_on_unlabeled=False,     # ปิด DIP ในโหมด label_only
            )

            cmd = [sys.executable, "-m", "qc.cli", "infer", "--cfg", str(cfg_path)]
            st.caption("Run infer: " + " ".join(cmd))
            log = st.empty(); log.write("---- LOG (infer label_only) ----")
            rc, _ = run(cmd, live_panel=log)
            if rc != 0:
                st.error("รันไม่สำเร็จ — ตรวจ LOG")
                st.stop()

            df2, used2 = read_report_flexible(out_live_dir)
            if df2 is None:
                st.error("ไม่พบ report.csv หลังรัน")
                st.stop()

            def tb(x): return (str(x).strip().lower() in {"1","true","yes","y","pass"})
            total = len(df2)
            passed = sum(tb(v) for v in df2["pass"]) if "pass" in df2.columns else None
            failed = total - passed if passed is not None else None

            cols = st.columns(3)
            cols[0].metric("Images", total)
            if failed is not None:
                cols[1].metric("Pass", passed)
                cols[2].metric("Fail", failed)

            st.subheader("ตารางผลลัพธ์ (label_only)")
            st.dataframe(df2, use_container_width=True)
            if used2 and used2.exists():
                with open(used2, "rb") as f:
                    st.download_button("ดาวน์โหลด report.csv", f, file_name=used2.name, mime="text/csv")

            # Overlay ที่ infer เซฟไว้
            ov_dir = out_live_dir / "overlay"
            ov_imgs = []
            if ov_dir.exists():
                for ext in IMG_EXTS:
                    ov_imgs.extend(sorted(ov_dir.glob(f"*{ext}")))
            if ov_imgs:
                st.subheader(f"Overlay ({len(ov_imgs)} รูป)")
                for row in chunk(ov_imgs, 4):
                    cols = st.columns(4)
                    for c, p in zip(cols, row):
                        with c:
                            try:
                                st.image(Image.open(p), caption=p.name, use_container_width=True)
                            except Exception as e:
                                st.warning(f"เปิดรูปไม่ได้: {p.name} ({e})")
            else:
                st.info("ยังไม่มีไฟล์ overlay ให้แสดง")
        except Exception as e:
            st.error(str(e))

# ---------------- Tab 3: All Results ----------------
with tab3:
    st.header("รวมผลลัพธ์จากหลาย out_dir")
    roots = st.text_input("โฟลเดอร์หลักที่อยากสแกน (คั่นด้วย ; )", value="out_ui;out_class10;out")
    extra = st.text_input("หรือระบุ out_dir เพิ่ม (คั่นด้วย ; )", value="")
    want = [s.strip() for s in (roots.split(";")+extra.split(";")) if s.strip()]
    found_outs = scan_out_dirs(want) if want else []
    st.caption(f"พบรายงาน {len(found_outs)} ชุด")

    if not found_outs:
        st.info("ยังไม่พบ report.csv ใต้โฟลเดอร์ที่กำหนด")
    else:
        # เลือกหลายอันเพื่อรวมผลในตารางได้
        selected = st.multiselect("เลือก out_dir ที่จะรวมแสดง (สำหรับตาราง)",
                                  options=[str(p) for p in found_outs],
                                  default=[str(found_outs[0])])

        # เลือก out_dir ที่ต้องการ "ดูรูป" โดยเฉพาะ (ดูทีละอัน)
        viewer_out = st.selectbox(
            "เลือก out_dir เพื่อดู Overlays",
            options=[str(p) for p in found_outs],
            index=0
        )

        # --- ตารางรวมผลของชุดที่เลือก ---
        if selected:
            df_all = load_many_reports([Path(s) for s in selected])
            if df_all.empty:
                st.info("ยังไม่มีตารางให้แสดง")
            else:
                st.subheader("ตารางรวมทั้งหมด")
                st.dataframe(df_all, use_container_width=True)

                if "pass" in df_all.columns:
                    df_all["pass_bool"] = df_all["pass"].astype(str).str.lower().isin(
                        ["1","true","yes","y","pass"]
                    )
                    agg = df_all.groupby("out_dir")["pass_bool"].agg(images="count", passed="sum")
                    agg["failed"] = agg["images"] - agg["passed"]
                    st.subheader("สรุปต่อ out_dir")
                    st.dataframe(agg, use_container_width=True)

                csv_all = df_all.to_csv(index=False).encode("utf-8")
                st.download_button("ดาวน์โหลดผลรวม (CSV)", csv_all, file_name="all_results.csv", mime="text/csv")

        # -------- Viewer: แสดงเฉพาะ Overlays ของ out_dir ที่เลือก --------
        st.divider()
        st.subheader("Viewer: Overlays")

        max_show = st.number_input(
            "จำนวน overlay สูงสุดที่แสดง (เพื่อความลื่นไหล)",
            min_value=4, max_value=200, value=48, step=4
        )

        od = Path(viewer_out)

        # ปุ่มสร้าง overlay (กรณียังไม่มี)
        gen_tool = Path("tools") / "generate_overlays_from_report.py"
        if st.button(f"Generate overlays สำหรับ {od.name}"):
            if not gen_tool.exists():
                st.error(f"ไม่พบสคริปต์: {gen_tool}")
            else:
                cmd_ov = [sys.executable, str(gen_tool), "--out_dir", str(od), "--global-circle"]
                log = st.empty(); log.write("---- LOG (overlay) ----")
                rc, _ = run(cmd_ov, live_panel=log)
                if rc == 0:
                    st.success("สร้าง Overlay สำเร็จ")
                else:
                    st.warning("สร้าง Overlay ไม่สำเร็จ (ดู LOG)")

        # --- แสดง Overlays ---
        ov_dir = od / "overlay"
        ov_imgs = []
        if ov_dir.exists():
            for ext in IMG_EXTS:
                ov_imgs.extend(sorted(ov_dir.glob(f"*{ext}")))

        st.markdown(
            f"**Overlays:** {len(ov_imgs)} รูป"
            + (f"  (แสดง {int(min(len(ov_imgs), max_show))} แรก)" if len(ov_imgs) > max_show else "")
        )

        if not ov_imgs:
            st.info("ยังไม่มีไฟล์ overlay ในโฟลเดอร์ overlay/")
        else:
            show_list = ov_imgs[: int(max_show)]
            for row in chunk(show_list, 4):
                cols = st.columns(4)
                for c, p in zip(cols, row):
                    with c:
                        try:
                            st.image(Image.open(p), caption=p.name, use_container_width=True)
                        except Exception as e:
                            st.warning(f"เปิดรูปไม่ได้: {p.name} ({e})")
