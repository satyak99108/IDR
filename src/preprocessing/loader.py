"""
IDR MVP -- Dataset Loader
Auto-discovers and loads IO-VNBD CSV files.
Handles both smartphone (S-) and vehicle (V-) datasets,
including synchronized versions.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetLoader:
    """Discovers and loads IO-VNBD dataset CSV files."""

    # Known subfolder names in the IO-VNBD repo
    SYNCED_FOLDER_PATTERNS = [
        "Synchronised V and S datasets",
        "Synchronised V abd S datasets",  # typo in actual IO-VNBD repo
        "Synchronised",
        "synchronised",
        "synced",
    ]

    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: Path to the cloned IO-VNBD repository root.
        """
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.dataset_root}"
            )

        self._csv_files: list[Path] = []
        self._synced_folder: Optional[Path] = None
        self._discover()

    def _discover(self):
        """Walk the dataset directory and discover all CSV files."""
        self._csv_files = sorted(self.dataset_root.rglob("*.csv"))

        # Find the synchronized dataset folder
        for folder in self.dataset_root.rglob("*"):
            if folder.is_dir():
                for pattern in self.SYNCED_FOLDER_PATTERNS:
                    if pattern.lower() in folder.name.lower():
                        self._synced_folder = folder
                        break
            if self._synced_folder:
                break

    @property
    def all_csv_files(self) -> list[Path]:
        """All discovered CSV files."""
        return self._csv_files

    @property
    def synced_folder(self) -> Optional[Path]:
        """Path to the synchronized datasets folder, if found."""
        return self._synced_folder

    def get_smartphone_files(self) -> list[Path]:
        """Return CSV files that appear to be smartphone data (S- prefix)."""
        return [f for f in self._csv_files if f.stem.startswith("S")]

    def get_vehicle_files(self) -> list[Path]:
        """Return CSV files that appear to be vehicle data (V- prefix)."""
        return [f for f in self._csv_files if f.stem.startswith("V")]

    def get_synced_files(self) -> list[Path]:
        """Return CSV files from the synchronized folder."""
        if self._synced_folder is None:
            return []
        return sorted(self._synced_folder.rglob("*.csv"))

    def get_trip_names(self) -> list[str]:
        """Extract unique trip names from file stems."""
        names = set()
        for f in self._csv_files:
            # Strip S- or V- prefix to get trip identifier
            stem = f.stem
            for prefix in ("S-", "V-", "S_", "V_"):
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
                    break
            names.add(stem)
        return sorted(names)

    def load_csv(
        self,
        filepath: Path,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load a single CSV file into a DataFrame.
        Attempts multiple delimiter strategies for robustness.

        Args:
            filepath: Path to the CSV file.
            nrows: Optional limit on rows to load.

        Returns:
            Loaded DataFrame.
        """
        # Try multiple encoding + delimiter combinations
        for encoding in ["utf-8", "latin1", "cp1252"]:
            for sep in [",", ";", "\t"]:
                try:
                    df = pd.read_csv(
                        filepath,
                        sep=sep,
                        nrows=nrows,
                        low_memory=False,
                        encoding=encoding,
                    )
                    if len(df.columns) > 1:
                        return df
                except Exception:
                    continue

        # Fallback: let pandas infer with latin1
        return pd.read_csv(
            filepath, nrows=nrows, low_memory=False, encoding="latin1"
        )

    def load_trip(
        self,
        trip_index: int = 0,
        data_type: str = "smartphone",
        synced: bool = True,
        nrows: Optional[int] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Load a trip by index.

        Args:
            trip_index: Index into the available files list.
            data_type: 'smartphone', 'vehicle', or 'synced'.
            synced: If True and data_type != 'synced', prefer synced folder.
            nrows: Optional row limit.

        Returns:
            Tuple of (DataFrame, metadata_dict).
        """
        if data_type == "synced" or synced:
            files = self.get_synced_files()
            if not files:
                # Fall back to individual files
                if data_type == "vehicle":
                    files = self.get_vehicle_files()
                else:
                    files = self.get_smartphone_files()
        elif data_type == "vehicle":
            files = self.get_vehicle_files()
        else:
            files = self.get_smartphone_files()

        if not files:
            raise FileNotFoundError(
                f"No {data_type} CSV files found in {self.dataset_root}"
            )

        if trip_index >= len(files):
            raise IndexError(
                f"Trip index {trip_index} out of range. "
                f"Available: 0-{len(files) - 1}"
            )

        filepath = files[trip_index]
        df = self.load_csv(filepath, nrows=nrows)

        metadata = {
            "filepath": str(filepath),
            "filename": filepath.name,
            "trip_index": trip_index,
            "data_type": data_type,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
        }

        return df, metadata

    def get_synced_pairs(self) -> dict[str, dict[str, Path]]:
        """
        Find all matched pairs of (S-*, V-*) files in the synchronized dataset.

        Returns:
            Dictionary mapping trip_id -> {'smartphone': Path, 'vehicle': Path}
        """
        if self._synced_folder is None:
            return {}

        s_files = {}
        v_files = {}

        for f in self._synced_folder.rglob("*.csv"):
            stem = f.stem
            stem_lower = stem.lower()
            if stem_lower.startswith("s-") or stem_lower.startswith("s_"):
                trip_id = stem[2:].lower()
                s_files[trip_id] = f
            elif stem_lower.startswith("v-") or stem_lower.startswith("v_"):
                trip_id = stem[2:].lower()
                v_files[trip_id] = f

        pairs = {}
        for trip_id, s_path in s_files.items():
            if trip_id in v_files:
                pairs[trip_id] = {
                    "smartphone": s_path,
                    "vehicle": v_files[trip_id],
                }

        return pairs

    def load_synced_pair(
        self,
        trip_index: int = 0,
        trip_name: Optional[str] = None,
        nrows: Optional[int] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Load both smartphone and vehicle CSV files for a synchronized trip,
        strip whitespace from column headers, prefix vehicle columns with 'REF_',
        and merge them side-by-side into a single unified DataFrame.

        Args:
            trip_index: Index in available paired trips list.
            trip_name: Specific trip ID to load (e.g. 's1', 'm').
            nrows: Optional row limit.

        Returns:
            Tuple of (merged DataFrame, metadata dict).
        """
        pairs = self.get_synced_pairs()
        if not pairs:
            # Fallback to load_trip if no pairs found
            return self.load_trip(trip_index=trip_index, nrows=nrows)

        trip_ids = sorted(pairs.keys())
        if trip_name:
            clean_name = trip_name.lower().replace("s-", "").replace("v-", "")
            if clean_name not in pairs:
                raise KeyError(f"Trip '{trip_name}' not found in synced pairs. Available: {trip_ids[:10]}")
            selected_trip = clean_name
        else:
            if trip_index >= len(trip_ids):
                raise IndexError(f"Trip index {trip_index} out of range (0-{len(trip_ids)-1})")
            selected_trip = trip_ids[trip_index]

        s_path = pairs[selected_trip]["smartphone"]
        v_path = pairs[selected_trip]["vehicle"]

        df_s = self.load_csv(s_path, nrows=nrows)
        df_v = self.load_csv(v_path, nrows=nrows)

        # Strip whitespace from column names
        df_s.columns = [c.strip() for c in df_s.columns]
        df_v.columns = [c.strip() for c in df_v.columns]

        # Prefix vehicle columns with 'REF_' to avoid collision and clearly denote ground truth
        df_v_renamed = df_v.copy()
        df_v_renamed.columns = [f"REF_{c}" for c in df_v.columns]

        # Combine side-by-side (both are synchronized at 10 Hz)
        min_len = min(len(df_s), len(df_v_renamed))
        df_merged = pd.concat(
            [df_s.iloc[:min_len].reset_index(drop=True),
             df_v_renamed.iloc[:min_len].reset_index(drop=True)],
            axis=1,
        )

        metadata = {
            "trip_id": selected_trip,
            "smartphone_file": str(s_path),
            "vehicle_file": str(v_path),
            "filename": f"SYNC_{selected_trip}.csv",
            "shape": df_merged.shape,
            "smartphone_rows": len(df_s),
            "vehicle_rows": len(df_v),
            "merged_rows": min_len,
            "has_ground_truth": True,
        }

        return df_merged, metadata

    def summary(self) -> dict:
        """Return a summary of the discovered dataset."""
        synced_pairs = self.get_synced_pairs()
        return {
            "dataset_root": str(self.dataset_root),
            "total_csv_files": len(self._csv_files),
            "smartphone_files": len(self.get_smartphone_files()),
            "vehicle_files": len(self.get_vehicle_files()),
            "synced_folder": str(self._synced_folder) if self._synced_folder else None,
            "synced_files": len(self.get_synced_files()),
            "synced_pairs_count": len(synced_pairs),
            "synced_pair_trips": sorted(list(synced_pairs.keys())),
            "trip_names": self.get_trip_names(),
            "all_files": [str(f.relative_to(self.dataset_root)) for f in self._csv_files],
        }
