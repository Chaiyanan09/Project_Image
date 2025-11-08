import pandas as pd
from pathlib import Path

report = Path("out/report.csv")
assert report.exists(), "ไม่พบ out/report.csv (กรุณารัน infer ก่อน)"

df = pd.read_csv(report)
df.columns = [c.strip() for c in df.columns]

# เฉพาะรปท "ไมม label" (ในโหมด hybrid reason จะเปน 3 แบบนเทานน)
unl = df[df["reason"].isin(["normal","anomaly","segmentation"])].copy()

total = len(unl)
n_pass = int((unl["pass"]==1).sum())
n_fail = total - n_pass
print("=== Unlabeled set (วดจากโมเดล) ===")
if total:
    print(f"images: {total}, pass: {n_pass}, fail: {n_fail}, fail_rate: {n_fail/total:.3f}")
else:
    print("empty")

print("\nby reason:")
print(unl["reason"].value_counts())

# จดอนดบตวอยางทระบบจบผด (ชวยรววผลโมเดลจรง)
for col in ["defect_count","anomaly_score"]:
    if col not in unl.columns:
        unl[col] = 0
cols = [c for c in ["image","reason","defect_count","anomaly_score","types","pass"] if c in unl.columns]
top = unl[unl["pass"]==0].sort_values(["defect_count","anomaly_score"], ascending=[False,False]).head(50)
out_csv = Path("out/unlabeled_top_fails.csv")
top[cols].to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
