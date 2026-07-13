"""
recommendation.py
=================
Smart complementary product recommendation engine.

Architecture
------------
The engine uses a **two-stage hybrid** approach:

Stage 1 – Rule-Based Filtering
    For the query product's article type, look up the configured
    ``COMPLEMENTARY_MAPPING`` to get a list of compatible article types.
    Filter the product catalog to those types.

Stage 2 – Embedding Similarity Ranking
    Within the filtered candidates, rank by cosine similarity of CLIP
    embeddings so the most visually / semantically coherent matches
    appear at the top.

This approach is transparent, explainable, and does not recommend
visually similar (competing) products.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config
import utils
from vector_database import FAISSVectorDB, SearchResult

logger = utils.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RecommendationResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    """Holds one recommended product with its explanation."""
    rank: int
    product_id: int
    image_path: str
    product_name: str
    article_type: str
    master_category: str
    base_colour: str
    brand: str
    reason: str            # human-readable explanation
    score: float           # confidence score ∈ [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# ProductRecommender
# ─────────────────────────────────────────────────────────────────────────────

class ProductRecommender:
    """
    Recommends complementary products for a given product.

    The recommendations are intentionally *different* from the input product —
    they are items that pair well together (e.g., shoes → socks, shirt → belt).

    Args:
        vector_db: Initialised :class:`FAISSVectorDB` containing all products.
        metadata:  Cleaned product metadata DataFrame.
    """

    def __init__(
        self,
        vector_db: FAISSVectorDB,
        metadata: pd.DataFrame,
    ) -> None:
        self.db = vector_db
        self.metadata = metadata.copy()
        # Build a lower-case lookup for article type → rows
        self.metadata["_article_type_lower"] = (
            self.metadata["articleType"].str.lower().str.strip()
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend_products(
        self,
        product_id: int,
        top_k: int = config.TOP_K_RECOMMENDATIONS,
    ) -> list[RecommendationResult]:
        """
        Recommend complementary products for a given product ID.

        Args:
            product_id: The source product's ID.
            top_k:      Number of recommendations to return.

        Returns:
            Ordered list of :class:`RecommendationResult` (best first).

        Raises:
            KeyError: If *product_id* is not found in the metadata.
        """
        # ── Resolve source product ────────────────────────────────────────────
        query_row = self._get_product(product_id)
        if query_row is None:
            raise KeyError(f"Product ID {product_id} not found in metadata.")

        query_article = query_row["_article_type_lower"]
        query_embedding = self.db.get_embedding_by_id(product_id)
        if query_embedding is None:
            logger.warning(f"No embedding found for product {product_id}.")
            return []

        # ── Stage 1: Rule-based complementary category lookup ─────────────────
        complementary_types = self._get_complementary_types(query_article)
        logger.debug(
            f"Product '{query_article}' → complementary types: {complementary_types}"
        )

        # ── Stage 2: Filter metadata to complementary types ───────────────────
        candidates = self.metadata[
            self.metadata["_article_type_lower"].isin(complementary_types)
        ].copy()

        if candidates.empty:
            logger.info(
                f"No direct complementary mapping for '{query_article}'. "
                "Using embedding similarity fallback."
            )
            return self._fallback_recommend(
                query_embedding, query_article, product_id, top_k
            )

        # ── Stage 3: Score candidates by cosine similarity ────────────────────
        candidate_ids = candidates["id"].tolist()
        scored = self._score_candidates(
            query_embedding, candidate_ids, query_row
        )

        # ── Stage 4: Build results list ───────────────────────────────────────
        results: list[RecommendationResult] = []
        for rank, (pid, score) in enumerate(scored[:top_k], start=1):
            row = self._get_product(pid)
            if row is None:
                continue
            reason = self._build_reason(query_article, row["_article_type_lower"])
            results.append(
                RecommendationResult(
                    rank=rank,
                    product_id=pid,
                    image_path=str(row.get("image_path", "")),
                    product_name=str(row.get("productDisplayName", "Unknown")),
                    article_type=str(row.get("articleType", "Unknown")),
                    master_category=str(row.get("masterCategory", "Unknown")),
                    base_colour=str(row.get("baseColour", "Unknown")),
                    brand=str(row.get("brand", "Unknown")),
                    reason=reason,
                    score=score,
                )
            )

        logger.info(
            f"Recommended {len(results)} complementary products for "
            f"product {product_id} ({query_article})."
        )
        return results

    def recommend_by_name(
        self,
        product_name: str,
        top_k: int = config.TOP_K_RECOMMENDATIONS,
    ) -> list[RecommendationResult]:
        """
        Recommend complementary products by partial product display name.

        Finds the closest matching product in the catalog and delegates
        to :meth:`recommend_products`.

        Args:
            product_name: Full or partial product display name.
            top_k:        Number of recommendations.

        Returns:
            List of :class:`RecommendationResult`.

        Raises:
            ValueError: If no matching product is found.
        """
        name_lower = product_name.lower().strip()
        mask = self.metadata["productDisplayName"].str.lower().str.contains(
            name_lower, na=False, regex=False
        )
        matches = self.metadata[mask]
        if matches.empty:
            raise ValueError(
                f"No product found matching: '{product_name}'.\n"
                "Try a shorter or different product name."
            )
        product_id = int(matches.iloc[0]["id"])
        logger.info(
            f"Matched query '{product_name}' → "
            f"ID {product_id}: {matches.iloc[0]['productDisplayName']}"
        )
        return self.recommend_products(product_id, top_k=top_k)

    def get_product_info(self, product_id: int) -> Optional[dict]:
        """Return metadata dict for a product ID, or None."""
        row = self._get_product(product_id)
        return row.to_dict() if row is not None else None

    def list_product_names(self) -> list[str]:
        """Return sorted list of all product display names."""
        return sorted(self.metadata["productDisplayName"].dropna().unique().tolist())

    def list_product_ids(self) -> list[int]:
        """Return list of all product IDs."""
        return self.metadata["id"].tolist()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_product(self, product_id: int) -> Optional[pd.Series]:
        """Look up a product row by ID."""
        rows = self.metadata[self.metadata["id"] == product_id]
        return rows.iloc[0] if not rows.empty else None

    def _get_complementary_types(self, article_type: str) -> list[str]:
        """
        Return the list of complementary article types for *article_type*.

        Tries exact match first, then partial-string match as fallback.
        """
        mapping = config.COMPLEMENTARY_MAPPING

        # Exact match
        if article_type in mapping:
            return [t.lower() for t in mapping[article_type]]

        # Partial match (e.g., "sports shoes" matches "shoes")
        for key, values in mapping.items():
            if key in article_type or article_type in key:
                return [t.lower() for t in values]

        # Default fallback
        return [t.lower() for t in config.DEFAULT_COMPLEMENTARY_CATEGORIES]

    def _score_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_ids: list[int],
        query_row: pd.Series,
    ) -> list[tuple[int, float]]:
        """
        Score candidate products by cosine similarity + category bonus.

        Args:
            query_embedding: Shape ``(D,)`` embedding of the source product.
            candidate_ids:   List of candidate product IDs.
            query_row:       Metadata row of the source product.

        Returns:
            Sorted list of ``(product_id, score)`` tuples, highest first.
        """
        scored: list[tuple[int, float]] = []
        q = query_embedding.reshape(1, -1).astype(np.float32)

        for pid in candidate_ids:
            emb = self.db.get_embedding_by_id(pid)
            if emb is None:
                continue
            cos_sim = float(np.dot(q[0], emb))
            cos_sim = float(np.clip(cos_sim, 0.0, 1.0))

            # Small bonus for same master category (fashion compatibility)
            row = self._get_product(pid)
            bonus = 0.0
            if row is not None:
                if row.get("masterCategory") == query_row.get("masterCategory"):
                    bonus += 0.02
                if row.get("usage") == query_row.get("usage"):
                    bonus += 0.01

            final_score = min(cos_sim + bonus, 1.0)
            scored.append((pid, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _fallback_recommend(
        self,
        query_embedding: np.ndarray,
        query_article: str,
        exclude_id: int,
        top_k: int,
    ) -> list[RecommendationResult]:
        """
        Fallback: recommend top-K products from *different* master categories.

        Used when no complementary mapping exists.
        """
        query_row = self.metadata[self.metadata["id"] == exclude_id]
        query_category = (
            query_row.iloc[0]["masterCategory"] if not query_row.empty else ""
        )

        # Exclude same category to avoid recommending similar items
        candidates = self.metadata[
            self.metadata["masterCategory"] != query_category
        ]
        candidate_ids = candidates["id"].tolist()
        scored = self._score_candidates(query_embedding, candidate_ids, query_row.iloc[0] if not query_row.empty else pd.Series())

        results: list[RecommendationResult] = []
        for rank, (pid, score) in enumerate(scored[:top_k], start=1):
            row = self._get_product(pid)
            if row is None or pid == exclude_id:
                continue
            results.append(
                RecommendationResult(
                    rank=rank,
                    product_id=pid,
                    image_path=str(row.get("image_path", "")),
                    product_name=str(row.get("productDisplayName", "Unknown")),
                    article_type=str(row.get("articleType", "Unknown")),
                    master_category=str(row.get("masterCategory", "Unknown")),
                    base_colour=str(row.get("baseColour", "Unknown")),
                    brand=str(row.get("brand", "Unknown")),
                    reason=f"Complementary accessory for {query_article} (embedding similarity)",
                    score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _build_reason(source_type: str, target_type: str) -> str:
        """Construct a human-readable recommendation reason string."""
        return (
            f"'{target_type.title()}' is a popular complementary item "
            f"for '{source_type.title()}' — based on fashion compatibility rules "
            f"and CLIP embedding similarity."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import DatasetLoader
    from embedding import generate_and_cache_embeddings
    from vector_database import build_and_save_index

    loader = DatasetLoader(max_products=500).load()
    embeddings, ids, meta = generate_and_cache_embeddings(loader.df)
    db = build_and_save_index(embeddings, ids, meta)

    recommender = ProductRecommender(db, meta)

    # Pick first product with a known category
    sample_id = int(ids[0])
    sample_name = meta[meta["id"] == sample_id]["productDisplayName"].iloc[0]
    print(f"\nRecommendations for: {sample_name} (ID: {sample_id})")
    print("-" * 60)

    recs = recommender.recommend_products(sample_id, top_k=5)
    for r in recs:
        print(
            f"  [{r.rank}] {r.product_name[:40]}"
            f"  | {r.article_type}"
            f"  | score={r.score:.3f}"
            f"\n       Reason: {r.reason}"
        )
