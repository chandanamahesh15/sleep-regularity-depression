"""Download StudentLife from the Kaggle mirror and link it to data/raw/dataset.

The original Dartmouth server (studentlife.cs.dartmouth.edu) often times out,
so this uses the Kaggle copy `dartweichen/student-life` (~390 MB).

Usage: python scripts/download_data.py
"""

import glob
import os
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "dataset"

path = kagglehub.dataset_download("dartweichen/student-life")
hits = glob.glob(f"{path}/**/survey/PHQ-9.csv", recursive=True)
if not hits:
    raise SystemExit(f"PHQ-9.csv not found under {path}")
source = Path(hits[0]).parents[1]

TARGET.parent.mkdir(parents=True, exist_ok=True)
if TARGET.is_symlink():
    TARGET.unlink()  # refresh an old link
if TARGET.exists():
    print(f"{TARGET} already exists (not a link); leaving it as is")
else:
    os.symlink(source, TARGET, target_is_directory=True)

for sub in ["sensing/phonelock", "sensing/conversation", "app_usage", "survey"]:
    print(f"{sub:22s} {len(os.listdir(TARGET / sub))} files")
print(f"\nData ready at {TARGET} -> {source}")
