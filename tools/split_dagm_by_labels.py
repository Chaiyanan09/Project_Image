# tools/split_dagm_by_labels.py
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]

def normalize_stem(stem: str):
    # รองรับทั้งกรณีชื่อ label เหมือนรูป และกรณีมี suffix "_label"
    if stem.endswith("_label"):
        return stem[:-6]
    return stem

def collect_label_stems(label_dir: Path):
    stems = set()
    if not label_dir.exists():
        return stems
    for lp in list_images(label_dir):
        stems.add(normalize_stem(lp.stem))
    return stems

def copy(p: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(p), str(dst_dir / p.name))

def split_class10(class10_dir: Path,
                  out_train_normal: Path,
                  out_train_defect: Path,
                  out_infer_input: Path):
    train = class10_dir / "Train"
    test  = class10_dir / "Test"

    # ----- Train -----
    train_label_dir = train / "Label"
    label_stems = collect_label_stems(train_label_dir)

    for ip in list_images(train):
        if "Label" in ip.parts:   # ข้ามไฟล์ในโฟลเดอร์ Label
            continue
        stem = ip.stem
        is_defect = (normalize_stem(stem) in label_stems)
        if is_defect:
            copy(ip, out_train_defect)
        else:
            copy(ip, out_train_normal)

    # ----- Test → infer_input (คละ) -----
    for ip in list_images(test):
        if "Label" in ip.parts:
            continue
        copy(ip, out_infer_input)

if __name__ == "__main__":
    # ปรับ path ตรงนี้ให้ตรงกับของคุณ
    class10_dir = Path("data/DAGM_2007/DAGM_KaggleUpload/Class10")
    out_train_normal = Path("data/train_normal")
    out_train_defect = Path("data/train_defect")
    out_infer_input  = Path("data/infer_input")

    split_class10(class10_dir, out_train_normal, out_train_defect, out_infer_input)
    print("Done.")
    print(" - data/train_normal :", len(list(out_train_normal.glob('*'))), "files")
    print(" - data/train_defect :", len(list(out_train_defect.glob('*'))), "files")
    print(" - data/infer_input  :", len(list(out_infer_input.glob('*'))), "files")
