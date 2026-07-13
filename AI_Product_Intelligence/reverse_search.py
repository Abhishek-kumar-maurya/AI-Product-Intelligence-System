"""
reverse_search.py
=================
Text-to-image reverse product search powered by CLIP + FAISS.

How it works
------------
1. The user provides a free-form text query (e.g. "blue casual shirt").
2. CLIP's text encoder converts the query to a 512-D embedding in the
   same semantic space as the image embeddings.
3. FAISS performs a cosine-similarity search over the indexed product
   embeddings to find the top-K nearest images.
4. The results are returned with metadata (product name, category, score).

Performance measurement
-----------------------
The module tracks per-query latency and computes Precision@K when
ground-truth labels are available.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config
import utils
from embedding import CLIPEmbedder, get_embedder
from vector_database import FAISSVectorDB, SearchResult

logger = utils.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TextSearchResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextSearchResult:
    """Holds one result from a text-based reverse product search."""
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
    similarity: float          # cosine similarity ∈ [0, 1]
    similarity_pct: str        # formatted as "87.3%"


# ─────────────────────────────────────────────────────────────────────────────
# SearchMetrics dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchMetrics:
    """Performance metrics for a single search query."""
    query: str
    top_k: int
    inference_time_ms: float   # CLIP text encoding time
    retrieval_time_ms: float   # FAISS search time
    total_time_ms: float
    n_results: int
    max_similarity: float
    min_similarity: float
    avg_similarity: float


# ─────────────────────────────────────────────────────────────────────────────
# ReverseSearchEngine
# ─────────────────────────────────────────────────────────────────────────────

class ReverseSearchEngine:
    """
    Text-to-image product search engine using CLIP + FAISS.

    Args:
        vector_db: Initialised :class:`FAISSVectorDB`.
        embedder:  :class:`CLIPEmbedder` instance (shared with other modules).
    """

    def __init__(
        self,
        vector_db: FAISSVectorDB,
        embedder: Optional[CLIPEmbedder] = None,
    ) -> None:
        self.db = vector_db
        self.embedder: CLIPEmbedder = embedder or get_embedder()
        self._query_history: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_SEARCH_RESULTS,
        filters: Optional[dict] = None,
    ) -> tuple[list[TextSearchResult], SearchMetrics]:
        """
        Search for products matching a natural-language text query.

        Args:
            query:  Free-form text description, e.g. ``"blue casual shirt"``.
            top_k:  Number of results to return.
            filters: Optional dict of column→value filters applied after
                     retrieval (e.g. ``{"masterCategory": "Apparel"}``).

        Returns:
            Tuple ``(results, metrics)`` where *results* is a list of
            :class:`TextSearchResult` ordered by similarity (best first)
            and *metrics* contains timing and quality information.

        Raises:
            ValueError: If *query* is empty.
        """
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")

        query = query.strip()
        logger.info(f"Searching for: '{query}' (top_k={top_k})")

        # ── Step 1: Encode text query ─────────────────────────────────────────
        t0 = time.perf_counter()
        text_embedding = self.embedder.embed_text(query)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # ── Step 2: FAISS search ──────────────────────────────────────────────
        t2 = time.perf_counter()
        raw_results = self.db.search(text_embedding, k=top_k + 10)   # fetch extra for filtering
        t3 = time.perf_counter()
        retrieval_ms = (t3 - t2) * 1000

        # ── Step 3: Apply optional filters ───────────────────────────────────
        if filters:
            raw_results = self._apply_filters(raw_results, filters)

        raw_results = raw_results[:top_k]

        # ── Step 4: Convert to TextSearchResult ──────────────────────────────
        results: list[TextSearchResult] = [
            TextSearchResult(
                rank=i + 1,
                product_id=r.product_id,
                image_path=r.image_path,
                product_name=r.product_name,
                article_type=r.article_type,
                master_category=r.master_category,
                sub_category=r.sub_category,
                base_colour=r.base_colour,
                brand=r.brand,
                usage=r.usage,
                similarity=r.similarity,
                similarity_pct=utils.format_score(r.similarity),
            )
            for i, r in enumerate(raw_results)
        ]

        # ── Step 5: Compute metrics ───────────────────────────────────────────
        total_ms = (t3 - t0) * 1000
        sims = [r.similarity for r in results]
        metrics = SearchMetrics(
            query=query,
            top_k=top_k,
            inference_time_ms=round(inference_ms, 2),
            retrieval_time_ms=round(retrieval_ms, 2),
            total_time_ms=round(total_ms, 2),
            n_results=len(results),
            max_similarity=round(max(sims), 4) if sims else 0.0,
            min_similarity=round(min(sims), 4) if sims else 0.0,
            avg_similarity=round(float(np.mean(sims)), 4) if sims else 0.0,
        )

        self._query_history.append({
            "query": query,
            "n_results": len(results),
            "total_ms": total_ms,
            "avg_sim": metrics.avg_similarity,
        })

        logger.info(
            f"Search complete: {len(results)} results in {total_ms:.1f}ms "
            f"(encode={inference_ms:.1f}ms, faiss={retrieval_ms:.1f}ms)."
        )
        return results, metrics

    def search_multiple_queries(
        self,
        queries: list[str],
        top_k: int = config.TOP_K_SEARCH_RESULTS,
    ) -> list[tuple[list[TextSearchResult], SearchMetrics]]:
        """
        Run multiple queries sequentially and return all results.

        Useful for evaluation / batch testing.

        Args:
            queries: List of text queries.
            top_k:   Number of results per query.

        Returns:
            List of ``(results, metrics)`` tuples, one per query.
        """
        return [self.search(q, top_k=top_k) for q in queries]

    def compute_precision_at_k(
        self,
        query: str,
        relevant_article_types: list[str],
        k: int = 5,
    ) -> float:
        """
        Compute Precision@K for a query against ground-truth article types.

        Precision@K = (# relevant results in top-K) / K

        Args:
            query:                  Text query.
            relevant_article_types: Article types considered relevant
                                    (lower-cased).
            k:                      K value.

        Returns:
            Precision@K score ∈ [0.0, 1.0].
        """
        results, _ = self.search(query, top_k=k)
        relevant_lower = [t.lower() for t in relevant_article_types]
        hits = sum(
            1
            for r in results
            if r.article_type.lower() in relevant_lower
        )
        prec = hits / k if k > 0 else 0.0
        logger.info(
            f"Precision@{k} for '{query}': {prec:.2f} ({hits}/{k} relevant)"
        )
        return prec

    def get_performance_summary(self) -> dict:
        """
        Return aggregate performance stats across all queries made so far.

        Returns:
            Dict with avg / max latency and retrieval quality.
        """
        if not self._query_history:
            return {"message": "No queries made yet."}

        times = [q["total_ms"] for q in self._query_history]
        sims = [q["avg_sim"] for q in self._query_history]

        return {
            "total_queries": len(self._query_history),
            "avg_latency_ms": round(float(np.mean(times)), 2),
            "max_latency_ms": round(float(np.max(times)), 2),
            "avg_similarity": round(float(np.mean(sims)), 4),
            "queries": [q["query"] for q in self._query_history],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_filters(
        results: list[SearchResult],
        filters: dict,
    ) -> list[SearchResult]:
        """Apply column equality filters to a list of SearchResult objects."""
        filtered = []
        for r in results:
            match = all(
                str(getattr(r, col, "")).lower() == str(val).lower()
                for col, val in filters.items()
            )
            if match:
                filtered.append(r)
        return filtered


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

    engine = ReverseSearchEngine(db)

    test_queries = [
        "blue casual shirt",
        "running sports shoes",
        "black leather formal shoes",
        "red dress",
        "sports watch",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Query: '{query}'")
        results, metrics = engine.search(query, top_k=5)
        for r in results:
            print(
                f"  [{r.rank}] {r.product_name[:40]}"
                f"  | {r.article_type}"
                f"  | {r.similarity_pct}"
            )
        print(
            f"  Timing: {metrics.total_time_ms:.1f}ms "
            f"(encode={metrics.inference_time_ms:.1f}ms, "
            f"faiss={metrics.retrieval_time_ms:.1f}ms)"
        )

    # Precision@5 example
    prec = engine.compute_precision_at_k(
        "casual shirt",
        relevant_article_types=["shirts", "t-shirts", "tops"],
        k=5,
    )
    print(f"\nPrecision@5 for 'casual shirt': {prec:.2f}")

    print("\n=== Performance Summary ===")
    for k, v in engine.get_performance_summary().items():
        print(f"  {k}: {v}")
