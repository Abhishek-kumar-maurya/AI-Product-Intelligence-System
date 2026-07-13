"""
preprocessing.py
================
Dataset loading, cleaning, and exploratory analysis for the
Fashion Product Images Small dataset.

Responsibilities
----------------
* Load ``styles.csv`` and ``images.csv`` into DataFrames.
* Validate that each metadata row has a corresponding image on disk.
* Remove rows with missing critical fields (productDisplayName, articleType…).
* Expose a clean ``DataFrame`` and convenience accessors used by downstream
  modules (embedding, recommendation, dedup, search).
* Compute and return dataset statistics for the Home dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import config
import utils

logger = utils.get_logger(__name__)

# Columns that *must* be present in the cleaned DataFrame
REQUIRED_COLUMNS: list[str] = [
    "id",
    "articleType",
    "masterCategory",
    "subCategory",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName",
]


# ─────────────────────────────────────────────────────────────────────────────
# DatasetLoader
# ─────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads and cleans the Fashion Product Images Small dataset.

    Attributes:
        df (pd.DataFrame): Cleaned metadata (one row per valid product).
    """

    def __init__(self, max_products: Optional[int] = config.MAX_PRODUCTS) -> None:
        """
        Args:
            max_products: Cap on the number of products to load.
                          ``None`` loads the entire dataset.
        """
        self.max_products = max_products
        self.df: pd.DataFrame = pd.DataFrame()
        self._raw_df: pd.DataFrame = pd.DataFrame()

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> "DatasetLoader":
        """
        Full pipeline: read CSV → clean → validate images → cap size.

        Returns:
            *self* for fluent chaining: ``loader.load().df``.

        Raises:
            FileNotFoundError: If ``styles.csv`` is absent.
        """
        logger.info("Loading dataset …")
        self._raw_df = self._read_styles_csv()
        self.df = self._clean(self._raw_df.copy())
        self.df = self._validate_images(self.df)
        if self.max_products and len(self.df) > self.max_products:
            self.df = self.df.sample(n=self.max_products, random_state=42).reset_index(drop=True)
            logger.info(f"Capped dataset to {self.max_products} products.")
        logger.info(f"Dataset ready: {len(self.df)} valid products.")
        return self

    def get_statistics(self) -> dict:
        """
        Compute descriptive statistics for the Home dashboard.

        Returns:
            Dict with keys: total_products, categories, article_types,
            brands (if available), missing_before, missing_after,
            category_counts, article_type_counts.
        """
        if self.df.empty:
            return {}

        raw_len = len(self._raw_df) if not self._raw_df.empty else 0
        stats: dict = {
            "total_products": len(self.df),
            "raw_products": raw_len,
            "missing_before": raw_len - len(self.df),
            "master_categories": int(self.df["masterCategory"].nunique()),
            "article_types": int(self.df["articleType"].nunique()),
            "seasons": int(self.df["season"].nunique()),
            "category_counts": (
                self.df["masterCategory"]
                .value_counts()
                .to_dict()
            ),
            "article_type_counts": (
                self.df["articleType"]
                .value_counts()
                .head(20)
                .to_dict()
            ),
            "usage_counts": (
                self.df["usage"]
                .value_counts()
                .to_dict()
            ),
            "colour_counts": (
                self.df["baseColour"]
                .value_counts()
                .head(15)
                .to_dict()
            ),
        }
        return stats

    def get_sample_images(self, n: int = config.SAMPLE_DISPLAY) -> list[tuple[Image.Image, str]]:
        """
        Return *n* random (image, label) pairs for display.

        Args:
            n: Number of samples.

        Returns:
            List of ``(PIL.Image, label_string)`` tuples, skipping
            any products whose image cannot be loaded.
        """
        samples = self.df.sample(min(n * 2, len(self.df)), random_state=0)
        results: list[tuple[Image.Image, str]] = []
        for _, row in samples.iterrows():
            img_path = utils.find_image_path(row["id"])
            if img_path is None:
                continue
            img = utils.load_image_safe(img_path, size=(224, 224))
            if img is None:
                continue
            label = utils.truncate_str(str(row["productDisplayName"]), 30)
            results.append((img, label))
            if len(results) >= n:
                break
        return results

    def get_product_by_id(self, product_id: int) -> Optional[pd.Series]:
        """Return the metadata row for a given product ID, or None."""
        rows = self.df[self.df["id"] == product_id]
        return rows.iloc[0] if not rows.empty else None

    def get_products_by_article_type(self, article_type: str) -> pd.DataFrame:
        """Return all rows matching *article_type* (case-insensitive)."""
        mask = self.df["articleType"].str.lower() == article_type.lower()
        return self.df[mask]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_styles_csv(self) -> pd.DataFrame:
        """Read ``styles.csv`` with robust encoding / error handling."""
        csv_path = config.STYLES_CSV
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}.\n"
                "Download from: https://www.kaggle.com/datasets/paramaggarwal/"
                "fashion-product-images-small\n"
                f"Place styles.csv inside: {config.DATA_DIR}"
            )
        df = pd.read_csv(
            csv_path,
            on_bad_lines="skip",   # skip malformed rows
            encoding="utf-8",
            low_memory=False,
        )
        logger.info(f"Loaded styles.csv: {len(df)} rows, {len(df.columns)} columns.")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with critical missing values and normalise column types.

        Steps:
        1. Strip whitespace from column names.
        2. Drop rows missing any ``REQUIRED_COLUMNS``.
        3. Cast ``id`` to int.
        4. Fill non-critical columns with sensible defaults.
        5. Strip leading / trailing whitespace from string columns.
        """
        # Normalise column names
        df.columns = [c.strip() for c in df.columns]

        # Keep only columns that exist (graceful if CSV differs slightly)
        present = [c for c in REQUIRED_COLUMNS if c in df.columns]
        missing_cols = set(REQUIRED_COLUMNS) - set(present)
        if missing_cols:
            logger.warning(f"Missing columns in CSV: {missing_cols}. Proceeding.")

        # Drop rows with NaN in critical columns
        before = len(df)
        df.dropna(subset=present, inplace=True)
        logger.info(f"Dropped {before - len(df)} rows with missing critical values.")

        # Type casting
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df.dropna(subset=["id"], inplace=True)
        df["id"] = df["id"].astype(int)

        # Remove duplicate IDs
        dup_before = len(df)
        df.drop_duplicates(subset=["id"], inplace=True)
        logger.info(f"Removed {dup_before - len(df)} duplicate product IDs.")

        # Strip whitespace in string columns
        str_cols = df.select_dtypes(include="object").columns.tolist()
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()

        # Fill non-critical fields
        if "brand" not in df.columns:
            df["brand"] = "Unknown"
        df.fillna({"brand": "Unknown", "baseColour": "Unknown", "usage": "Unknown"}, inplace=True)

        df.reset_index(drop=True, inplace=True)
        return df

    def _validate_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only rows whose image exists on disk.

        Adds an ``image_path`` column (str) with the resolved path.
        """
        logger.info("Validating image files on disk …")
        valid_mask = []
        paths: list[str] = []

        for _, row in df.iterrows():
            p = utils.find_image_path(row["id"])
            if p is not None:
                valid_mask.append(True)
                paths.append(str(p))
            else:
                valid_mask.append(False)
                paths.append("")

        df["image_path"] = paths
        before = len(df)
        df = df[valid_mask].reset_index(drop=True)
        logger.info(
            f"Image validation: {len(df)} valid / {before - len(df)} missing files removed."
        )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────

_loader_instance: Optional[DatasetLoader] = None


def get_dataset_loader() -> DatasetLoader:
    """
    Return the module-level singleton :class:`DatasetLoader`.

    Loads the dataset on first call; subsequent calls return the cached instance.

    Returns:
        Loaded :class:`DatasetLoader`.
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DatasetLoader().load()
    return _loader_instance


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = DatasetLoader(max_products=500).load()
    stats = loader.get_statistics()
    print("\n=== Dataset Statistics ===")
    for key, val in stats.items():
        if not isinstance(val, dict):
            print(f"  {key}: {val}")
    print("\nTop 5 article types:")
    for at, cnt in list(stats.get("article_type_counts", {}).items())[:5]:
        print(f"  {at}: {cnt}")
    print("\nSample row:")
    print(loader.df.head(1).T)
