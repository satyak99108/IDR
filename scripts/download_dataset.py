"""
IDR MVP -- Dataset Downloader
Downloads/clones the IO-VNBD dataset repository.

Usage:
    python scripts/download_dataset.py
"""

import subprocess
import sys
from pathlib import Path

DATASET_URL = "https://github.com/onyekpeu/IO-VNBD.git"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "IO-VNBD"


def download():
    print("+==========================================================+")
    print("|          IDR MVP -- IO-VNBD Dataset Downloader           |")
    print("+==========================================================+\n")

    if DATASET_DIR.exists() and any(DATASET_DIR.rglob("*.csv")):
        csv_count = len(list(DATASET_DIR.rglob("*.csv")))
        print(f"  Dataset already exists at: {DATASET_DIR}")
        print(f"  Found {csv_count} CSV files.")
        print("  To re-download, delete the directory first.")
        return

    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Cloning IO-VNBD from: {DATASET_URL}")
    print(f"  Destination: {DATASET_DIR}")
    print(f"  This may take several minutes (large dataset)...\n")

    try:
        result = subprocess.run(
            ["git", "clone", DATASET_URL, str(DATASET_DIR)],
            check=True,
            capture_output=False,
        )
        print(f"\n  [OK] Clone complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n  [FAIL] Clone failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n  [FAIL] Git not found. Please install Git and try again.")
        sys.exit(1)

    # Verify
    csv_files = list(DATASET_DIR.rglob("*.csv"))
    print(f"\n  Verification:")
    print(f"    CSV files found: {len(csv_files)}")
    for f in csv_files[:10]:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"      {f.relative_to(DATASET_DIR)} ({size_mb:.1f} MB)")
    if len(csv_files) > 10:
        print(f"      ... and {len(csv_files) - 10} more")

    if csv_files:
        print(f"\n  [OK] Dataset ready!")
    else:
        print(f"\n  [WARN] No CSV files found -- check the repository structure.")


if __name__ == "__main__":
    download()
