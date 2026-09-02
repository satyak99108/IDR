# Dataset Directory

The raw and processed datasets are intentionally excluded from version control via `.gitignore` to keep the repository lightweight.

## Downloading the Dataset

To download and extract the calibrated IO-VNBD dataset (`SYNC_s1_calibrated.csv`), run:

```bash
python scripts/download_dataset.py
```

This will automatically create and populate:
- `data/raw/` — Raw sensor files
- `data/processed/SYNC_s1_calibrated.csv` — Synchronized and calibrated 10 Hz dataset
