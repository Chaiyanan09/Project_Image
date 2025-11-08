"""
Download DAGM 2007 dataset via kagglehub.

This script uses KaggleHub (no CLI key needed). Internet is required.
If your environment blocks external access, download manually and place under data/DAGM_2007.
"""
import os
import shutil
from pathlib import Path
import kagglehub

def main():
    print("Downloading DAGM 2007 via kagglehub...")
    path = kagglehub.dataset_download("mhskjelvareid/dagm-2007-competition-dataset-optical-inspection")
    print("Path to dataset files:", path)

    dst = Path("data/DAGM_2007")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy (preserve structure)
    if dst.exists():
        print("Existing data/DAGM_2007 found. Keeping it (no overwrite).")
    else:
        print("Copying to data/DAGM_2007 ...")
        shutil.copytree(path, dst)
        print("Done:", dst)

if __name__ == "__main__":
    main()
