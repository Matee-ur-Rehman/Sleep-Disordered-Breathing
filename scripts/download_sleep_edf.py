"""
Download the Sleep-EDF Expanded, Sleep Cassette (SC) subset.

This uses MNE's built-in fetcher (mne.datasets.sleep_physionet), which pulls
directly from PhysioNet and correctly pairs each *PSG.edf with its matching
*Hypnogram.edf annotation file. No PhysioNet account is required - this
dataset is fully open.

Run this on YOUR machine (with internet access), not in a sandboxed
environment. Expect ~8 GB total for all 78 subjects / 153 recordings.

Usage:
    python download_sleep_edf.py --out ./data/sleep-edf-sc --subjects all
    python download_sleep_edf.py --out ./data/sleep-edf-sc --subjects 0-9   # quick test

We match the paper's stated cohort (78 subjects, ages 25-101) using the Sleep
Cassette subset. Per MNE's own documentation (verified against
mne.datasets.sleep_physionet.age.fetch_data), valid subject indices run 0-82
inclusive, EXCLUDING indices 39, 68, 69, 78, 79 (not available at PhysioNet).
That is 83 - 5 = 78 subjects total, 153 recordings (two nights each, except
subjects 13, 36, 52 which have one missing night) - this exactly matches the
paper's stated 78 subjects / 25-101 age range, confirming SC is the right
subset (the paper's "200 recordings" figure does not match any known release
and appears to be an error in the manuscript; the real total is 153).
"""

import argparse
import sys
from pathlib import Path

MISSING_SUBJECT_IDS = {39, 68, 69, 78, 79}  # verified via MNE source / docs


def parse_subject_range(spec: str, max_subjects: int = 83):
    if spec == "all":
        ids = range(max_subjects)
    elif "-" in spec:
        lo, hi = spec.split("-")
        ids = range(int(lo), int(hi) + 1)
    else:
        ids = [int(x) for x in spec.split(",")]
    return [i for i in ids if i not in MISSING_SUBJECT_IDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="./data/sleep-edf-sc",
                         help="Output directory for downloaded EDF files.")
    parser.add_argument("--subjects", type=str, default="all",
                         help="'all', a range like '0-9', or a comma list like '0,1,5'.")
    parser.add_argument("--recording", type=str, default="1,2",
                         help="Which night(s) to fetch: '1', '2', or '1,2' (default both).")
    args = parser.parse_args()

    try:
        from mne.datasets.sleep_physionet.age import fetch_data
    except ImportError:
        print("ERROR: mne is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subject_range(args.subjects)
    recordings = [int(x) for x in args.recording.split(",")]

    print(f"Fetching Sleep-EDF Sleep Cassette: {len(subjects)} subject(s), "
          f"recording(s) {recordings} -> {out_dir}")

    # fetch_data returns a list of [psg_path, hypnogram_path] pairs and also
    # caches files under MNE's data directory; we point it at our own out_dir
    # via the `path` argument so everything lands in one predictable place.
    paths = fetch_data(subjects=subjects, recording=recordings, path=str(out_dir),
                        on_missing="raise")

    print(f"Done. {len(paths)} PSG/hypnogram pairs downloaded.")
    for p in paths[:5]:
        print("  ", p)
    if len(paths) > 5:
        print(f"  ... and {len(paths) - 5} more.")

    print("\nNext step: run scripts/verify_download.py to sanity-check the files, "
          "then src/preprocessing.py to build the preprocessed dataset.")


if __name__ == "__main__":
    main()
