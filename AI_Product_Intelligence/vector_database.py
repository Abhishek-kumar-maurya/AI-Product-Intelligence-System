"""
vector_database.py
==================
FAISS-based vector index for fast approximate nearest-neighbour search.

Responsibilities
----------------
* Build a searchable FAISS index from precomputed CLIP embeddings.
* Attach rich product metadata to each index entry for retrieval.
* Persist the index and metadata to disk.
* Provide a clean ``search(vector, k)`` interface returning structured results.

Index type: ``IndexFlatIP`` (inner product on L2-normalised vectors = cosine similarity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

import config
import utils

logger = utils.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SearchResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Holds one nearest-neighbour result returned by the FAISS index."""
    rank: int
    product_id: int
    image_path: str
    product_name: str
    article_type: str
    master_category: str
    sub_category: str
    base_colour: str
    brand: str
    usage: str
    similarity: float            # cosine similarity ∈ [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# FAISSVectorDB
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorDB:
    """
    Wraps a FAISS flat inner-product index with aligned product metadata.

    Attributes:
        index (faiss.Index): The FAISS index object.
        metadata (pd.DataFrame): DataFrame aligned row-for-row with the index.
        product_ids (np.ndarray): int64 array of product IDs.
        embed_dim (int): Embedding dimensionality.
        n_products (int): Number of indexed products.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.metadata: pd.DataFrame = pd.DataFrame()
        self.product_ids: np.ndarray = np.array([], dtype=np.int64)
        self.embed_dim: int = 0
        self.n_products: int = 0

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        product_ids: np.ndarray,
        metadata: pd.DataFrame,
    ) -> None:
        """
        Build a FAISS ``IndexFlatIP`` from precomputed embeddings.

        The embeddings must already be L2-normalised (inner product on
        unit vectors == cosine similarity).

        Args:
            embeddings:  Shape ``(N, D)`` float32 array, L2-normalised.
            product_ids: Shape ``(N,)`` int64 array.
            metadata:    DataFrame with one row per product, aligned with
                         *embeddings* by position.

        Raises:
            ValueError: If array shapes are inconsistent.
        """
        if embeddings.shape[0] != len(product_ids):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(product_ids)} product IDs."
            )

        self.embed_dim = embeddings.shape[1]
        self.n_products = embeddings.shape[0]
        self.product_ids = product_ids.copy()
        self.metadata = metadata.reset_index(drop=True)

        logger.info(
            f"Building FAISS IndexFlatIP: {self.n_products} vectors × {self.embed_dim}D …"
        )

        with utils.timer("FAISS index build", logger):
            self.index = faiss.IndexFlatIP(self.embed_dim)
            # Ensure float32 contiguous array
            vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
            self.index.add(vecs)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors indexed.")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        k: int = config.TOP_K_SEARCH_RESULTS,
        exclude_ids: Optional[list[int]] = None,
    ) -> list[SearchResult]:
        """
        Search the index for the *k* nearest neighbours of *query_vector*.

        Args:
            query_vector: Shape ``(D,)`` or ``(1, D)`` float32, L2-normalised.
            k:            Number of results to return.
            exclude_ids:  Optional list of product IDs to exclude from results
                          (e.g., the query product itself).

        Returns:
            Ordered list of :class:`SearchResult` objects (best match first).

        Raises:
            RuntimeError: If the index has not been built / loaded yet.
        """
        if self.index is None:
            raise RuntimeError("Index not initialised. Call build_index() or load().")

        # Reshape to (1, D)
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)

        # Request extra candidates so we can filter exclusions
        fetch_k = min(k + len(exclude_ids or []) + 5, self.index.ntotal)
        distances, indices = self.index.search(q, fetch_k)

        results: list[SearchResult] = []
        for sim, idx in zip(distances[0], indices[0]):
            if idx == -1:                     # FAISS padding for short results
                continue
            pid = int(self.product_ids[idx])
            if exclude_ids and pid in exclude_ids:
                continue
            row = self._get_row(idx)
            if row is None:
                continue
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    product_id=pid,
                    image_path=str(row.get("image_path", "")),
                    product_name=str(row.get("productDisplayName", "Unknown")),
                    article_type=str(row.get("articleType", "Unknown")),
                    master_category=str(row.get("masterCategory", "Unknown")),
                    sub_category=str(row.get("subCategory", "Unknown")),
                    base_colour=str(row.get("baseColour", "Unknown")),
                    brand=str(row.get("brand", "Unknown")),
                    usage=str(row.get("usage", "Unknown")),
                    similarity=float(np.clip(sim, 0.0, 1.0)),
                )
            )
            if len(results) >= k:
                break

        return results

    def search_by_product_id(
        self,
        product_id: int,
        k: int = config.TOP_K_SEARCH_RESULTS,
    ) -> list[SearchResult]:
        """
        Search for nearest neighbours of a product by its ID.

        Args:
            product_id: The source product's ID.
            k:          Number of *other* products to return.

        Returns:
            List of :class:`SearchResult` (excludes the query product).

        Raises:
            KeyError: If *product_id* is not in the index.
        """
        idx = self._id_to_index(product_id)
        if idx is None:
            raise KeyError(f"Product ID {product_id} not found in index.")

        q_vec = self.index.reconstruct(int(idx))
        return self.search(q_vec, k=k, exclude_ids=[product_id])

    def get_embedding_by_id(self, product_id: int) -> Optional[np.ndarray]:
        """
        Retrieve the stored embedding vector for a product.

        Args:
            product_id: Product ID.

        Returns:
            Shape ``(D,)`` float32 array, or ``None`` if not found.
        """
        idx = self._id_to_index(product_id)
        if idx is None:
            return None
        return self.index.reconstruct(int(idx))

    def get_all_embeddings(self) -> np.ndarray:
        """Return all stored embeddings as a ``(N, D)`` float32 array."""
        if self.index is None or self.n_products == 0:
            return np.array([], dtype=np.float32)
        vecs = np.zeros((self.n_products, self.embed_dim), dtype=np.float32)
        for i in range(self.n_products):
            vecs[i] = self.index.reconstruct(i)
        return vecs

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Save the FAISS index and aligned metadata to disk.

        Files written:

        * ``faiss_index/product_index.faiss``
        * ``faiss_index/product_metadata.pkl``
        """
        if self.index is None:
            raise RuntimeError("No index to save.")
        config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(config.FAISS_INDEX_FILE))
        meta_bundle = {
            "metadata": self.metadata,
            "product_ids": self.product_ids,
            "embed_dim": self.embed_dim,
        }
        utils.save_pickle(meta_bundle, config.FAISS_METADATA_FILE)
        logger.info(
            f"FAISS index saved → {config.FAISS_INDEX_FILE} "
            f"({self.index.ntotal} vectors)."
        )

    def load(self) -> None:
        """
        Load the FAISS index and metadata from disk.

        Raises:
            FileNotFoundError: If index or metadata files are missing.
        """
        if not config.FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {config.FAISS_INDEX_FILE}.\n"
                "Run the system setup first (python app.py will do this automatically)."
            )
        logger.info(f"Loading FAISS index from {config.FAISS_INDEX_FILE} …")
        self.index = faiss.read_index(str(config.FAISS_INDEX_FILE))

        meta_bundle = utils.load_pickle(config.FAISS_METADATA_FILE)
        self.metadata = meta_bundle["metadata"]
        self.product_ids = meta_bundle["product_ids"]
        self.embed_dim = meta_bundle["embed_dim"]
        self.n_products = self.index.ntotal

        logger.info(
            f"FAISS index loaded: {self.n_products} vectors, dim={self.embed_dim}."
        )

    @classmethod
    def is_saved(cls) -> bool:
        """Return True if saved index files exist on disk."""
        return config.FAISS_INDEX_FILE.exists() and config.FAISS_METADATA_FILE.exists()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _id_to_index(self, product_id: int) -> Optional[int]:
        """Return FAISS row index for a given product ID, or None."""
        matches = np.where(self.product_ids == product_id)[0]
        return int(matches[0]) if len(matches) > 0 else None

    def _get_row(self, faiss_index: int) -> Optional[dict]:
        """Return metadata dict for a FAISS row index."""
        if faiss_index < 0 or faiss_index >= len(self.metadata):
            return None
        return self.metadata.iloc[faiss_index].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_db_instance: Optional[FAISSVectorDB] = None


def get_vector_db() -> FAISSVectorDB:
    """
    Return the module-level :class:`FAISSVectorDB` singleton.

    Loads from disk on first call if saved index exists;
    otherwise returns an empty, uninitialised instance.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = FAISSVectorDB()
        if FAISSVectorDB.is_saved():
            _db_instance.load()
    return _db_instance


def build_and_save_index(
    embeddings: np.ndarray,
    product_ids: np.ndarray,
    metadata: pd.DataFrame,
    force_rebuild: bool = False,
) -> FAISSVectorDB:
    """
    Build (or load) the FAISS index and update the module singleton.

    Args:
        embeddings:   Shape ``(N, D)`` float32, L2-normalised.
        product_ids:  Shape ``(N,)`` int64.
        metadata:     Aligned DataFrame.
        force_rebuild: If True, rebuild even if saved files exist.

    Returns:
        The ready-to-use :class:`FAISSVectorDB` instance.
    """
    global _db_instance

    if not force_rebuild and FAISSVectorDB.is_saved():
        logger.info("Loading existing FAISS index from disk.")
        db = FAISSVectorDB()
        db.load()
        _db_instance = db
        return db

    db = FAISSVectorDB()
    db.build_index(embeddings, product_ids, metadata)
    db.save()
    _db_instance = db
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import DatasetLoader
    from embedding import generate_and_cache_embeddings

    loader = DatasetLoader(max_products=300).load()
    embeddings, ids, meta = generate_and_cache_embeddings(loader.df)

    db = build_and_save_index(embeddings, ids, meta, force_rebuild=True)

    print(f"\nIndex size: {db.n_products} vectors × {db.embed_dim}D")

    # Test search with first product's embedding
    first_pid = int(ids[0])
    results = db.search_by_product_id(first_pid, k=3)
    print(f"\nTop-3 similar to product {first_pid}:")
    for r in results:
        print(f"  [{r.rank}] {r.product_id} | {r.product_name[:40]} | sim={r.similarity:.4f}")
