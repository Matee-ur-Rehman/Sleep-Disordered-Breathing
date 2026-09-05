"""
Quick check: load a processed .npz and print the sleep-stage label distribution.
Usage: python check_label_distribution.py data/processed/test_run.npz
"""
import sys
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/test_run.npz"
d = np.load(path, allow_pickle=True)
y = d["y"]
subjects = d["subjects"]

print(f"Total epochs: {len(y)}")
print(f"Unique subjects: {len(set(subjects))}")
print()
print("Overall label distribution:")
counts = np.bincount(y, minlength=5)
for i, s in enumerate(STAGES):
    pct = 100 * counts[i] / len(y)
    print(f"  {s:4s}: {counts[i]:6d}  ({pct:5.1f}%)")

print()
print("Paper's reported Sleep-EDF distribution (Table 1) for comparison:")
print("  W: 30.1%, N1: 8.9%, N2: 38.4%, N3: 12.5%, REM: 10.1%")
