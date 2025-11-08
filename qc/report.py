from pathlib import Path
import csv
import cv2

def overlay_and_save(img, boxes_typed, out_path):
    vis = img.copy()
    for (x,y,w,h,t) in boxes_typed:
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(vis, t, (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)

def write_csv(rows, csv_path):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "pass", "reason", "anomaly_score", "defect_count", "types"])
        for r in rows:
            w.writerow(r)
