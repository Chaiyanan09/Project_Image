import pandas as pd, matplotlib.pyplot as plt, zipfile, shutil
from pathlib import Path
from collections import Counter

OUT = Path("out"); REPORT = OUT / "report.csv"
assert REPORT.exists(), "ไมพบ out/report.csv (กรณารน infer กอน)"

df = pd.read_csv(REPORT)
df.columns = [c.strip() for c in df.columns]
df["pass"] = pd.to_numeric(df["pass"], errors="coerce").fillna(0).astype(int)
df["defect_count"] = pd.to_numeric(df["defect_count"], errors="coerce").fillna(0).astype(int)
df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0)

total = len(df); n_pass = int((df["pass"]==1).sum()); n_fail = total - n_pass
by_reason = df["reason"].value_counts().to_dict()
summary = pd.DataFrame({"metric":["total_images","pass","fail","pass_rate"] + [f"reason:{k}" for k in by_reason],
                        "value":[total, n_pass, n_fail, round(n_pass/total,4)] + [by_reason[k] for k in by_reason]})
summary.to_csv(OUT/"summary_metrics.csv", index=False)

# นบ type
ctr = Counter()
for s in df["types"].fillna("").astype(str):
    for t in [x.strip() for x in s.split(";") if x.strip()]:
        ctr[t]+=1
type_df = pd.DataFrame({"type":list(ctr.keys()),"count":list(ctr.values())}).sort_values("count", ascending=False)
type_df.to_csv(OUT/"summary_type_counts.csv", index=False)

# Top 24 fails
top = df[df["pass"]==0].sort_values(["defect_count","anomaly_score"], ascending=[False,False]).head(24)
top.to_csv(OUT/"summary_top24_fails.csv", index=False)

# กราฟ
plt.figure(); df[df["pass"]==0]["defect_count"].plot(kind="hist", bins=20, title="Defect Count Distribution (Fail only)")
plt.xlabel("defect_count"); plt.ylabel("images"); plt.tight_layout(); plt.savefig(OUT/"summary_defect_hist.png"); plt.close()

if not type_df.empty:
    plt.figure(figsize=(6,4)); type_df.set_index("type")["count"].plot(kind="bar", title="Defect Type Frequency")
    plt.xlabel("type"); plt.ylabel("count"); plt.tight_layout(); plt.savefig(OUT/"summary_type_bar.png"); plt.close()

# ZIP overlays เดน
ov_dir = OUT/"overlay"; pick_dir = OUT/"summary"; pick_dir.mkdir(parents=True, exist_ok=True)
picked = 0
if ov_dir.exists():
    for _,r in top.iterrows():
        p = ov_dir / f"{Path(r['image']).stem}_overlay.png"
        if p.exists(): shutil.copy2(p, pick_dir / p.name); picked += 1
with zipfile.ZipFile(OUT/"summary_overlays_top24.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in pick_dir.glob("*_overlay.png"): z.write(p, p.name)

print("OK ->",
      OUT/"summary_metrics.csv",
      OUT/"summary_type_counts.csv",
      OUT/"summary_top24_fails.csv",
      OUT/"summary_defect_hist.png",
      OUT/"summary_type_bar.png",
      OUT/"summary_overlays_top24.zip",
      f"(overlays:{picked})")
