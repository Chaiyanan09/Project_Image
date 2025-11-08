# tools/filter_normal_by_label.py
# คัด "รูปปกติ" จาก DAGM ClassX/Test โดยอ้างอิงรายชื่อไฟล์ในโฟลเดอร์ Test/Label
# เงื่อนไข: ถ้าใน Test/Label มีไฟล์ <stem>_label.<ext> แสดงว่า <stem>.<ext> คือรูป defect
# เราจะคัดเฉพาะรูปที่ไม่มี label คู่ ไปยังโฟลเดอร์ปลายทาง

import argparse
from pathlib import Path
import shutil

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", default="data/DAGM_2007/DAGM_KaggleUpload/Class9/Test",
                    help="โฟลเดอร์ Test ที่มีภาพและโฟลเดอร์ Label อยู่ภายใน")
    ap.add_argument("--dst", default="data_test/class9_clean_normals",
                    help="โฟลเดอร์ปลายทางสำหรับเก็บรูปปกติที่คัดแล้ว")
    ap.add_argument("--clear_dst", action="store_true",
                    help="ลบไฟล์เดิมในปลายทางก่อนคัดใหม่")
    ap.add_argument("--dry_run", action="store_true",
                    help="โหมดลองรัน: แค่รายงานจำนวน ไม่คัดลอกไฟล์")
    args = ap.parse_args()

    test_root = Path(args.test_root).resolve()
    label_dir = test_root / "Label"
    dst = Path(args.dst).resolve()

    assert test_root.exists(), f"ไม่พบโฟลเดอร์ Test: {test_root}"
    if not label_dir.exists():
        print(f"[คำเตือน] ไม่พบโฟลเดอร์ Label ภายใต้ {test_root} — จะถือว่าไม่มีรูป defect ทั้งหมด")
    all_imgs = [p for p in list_images(test_root) if "label" not in p.stem.lower()]  # กันรูป *_label.* ที่หลุดนอก Label
    # สร้างเซ็ตชื่อรูปที่มี label คู่ (stem แบบ lower)
    labeled_stems = set()
    if label_dir.exists():
        for m in list_images(label_dir):
            stem = m.stem.lower()
            if stem.endswith("_label"):
                stem = stem[:-6]
            labeled_stems.add(stem)

    normals = []
    defects = 0
    for p in all_imgs:
        # ข้ามไฟล์ในโฟลเดอร์ Label
        try:
            p.relative_to(label_dir)
            # อยู่ใน Label — ข้าม
            continue
        except ValueError:
            pass
        stem = p.stem.lower()
        if stem in labeled_stems:
            defects += 1
            continue
        normals.append(p)

    print(f"[สรุป] พบภาพทั้งหมด (ไม่รวม Label/*): {len(all_imgs)}")
    print(f"       จับคู่กับ Label ได้ (defect):   {defects}")
    print(f"       คัดเป็นรูปปกติ (normal):       {len(normals)}")
    print(f"ปลายทาง: {dst}")

    if args.dry_run:
        print("[dry-run] ไม่ได้คัดลอกไฟล์จริง")
        return

    dst.mkdir(parents=True, exist_ok=True)
    if args.clear_dst:
        for f in dst.glob("*"):
            if f.is_file():
                f.unlink()

    copied = 0
    for p in normals:
        shutil.copy2(str(p), str(dst / p.name))
        copied += 1
    print(f"[OK] คัดลอกไฟล์ปกติแล้ว: {copied} รูป → {dst}")

if __name__ == "__main__":
    main()
