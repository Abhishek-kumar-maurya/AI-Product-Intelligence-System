"""
duplicate_detection.py
======================
Identifies near-duplicate products and produces a clean unique catalog.

Algorithm
---------
1. Compute an all-pairs cosine similarity matrix from CLIP embeddings.
   (For large datasets, a FAISS radius search is used instead.)
2. Threshold the matrix at ``config.DUPLICATE_SIMILARITY_THRESHOLD`` to
   obtain a binary adjacency graph where edges = "likely duplicates".
3. Apply Union-Find (disjoint set union) to cluster connected components.
4. For each cluster, elect one *representative* product (highest image
   quality heuristic: largest file size → most detail).
5. Export ``unique_catalog.csv`` and ``duplicate_report.csv``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
import utils
from vector_database import FAISSVectorDB

logger = utils.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find (Disjoint Set Union)
# ─────────────────────────────────────────────────────────────────────────────

class UnionFind:
    """Path-compressed, union-by-rank disjoint set structure."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def get_components(self) -> dict[int, list[int]]:
        """Return mapping root → list of members."""
        components: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            components.setdefault(root, []).append(i)
        return components


# ─────────────────────────────────────────────────────────────────────────────
# DuplicateCluster dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DuplicateCluster:
    """One cluster of near-duplicate products."""
    cluster_id: int
    representative_id: int          # elected "best" product
    representative_name: str
    representative_image: str
    member_ids: list[int] = field(default_factory=list)
    member_names: list[str] = field(default_factory=list)
    avg_similarity: float = 0.0     # mean pairwise similarity within cluster
    size: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# DuplicateDetector
# ─────────────────────────────────────────────────────────────────────────────

class DuplicateDetector:
    """
    Finds near-duplicate products using CLIP embeddings and cosine similarity.

    Args:
        vector_db: Initialised :class:`FAISSVectorDB`.
        metadata:  Cleaned product metadata DataFrame.
        threshold: Cosine similarity threshold for declaring duplicates.
                   Default: ``config.DUPLICATE_SIMILARITY_THRESHOLD``.
    """

    def __init__(
        self,
        vector_db: FAISSVectorDB,
        metadata: pd.DataFrame,
        threshold: float = config.DUPLICATE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.db = vector_db
        self.metadata = metadata.reset_index(drop=True).copy()
        self.threshold = threshold
        self.clusters: list[DuplicateCluster] = []
        self.unique_catalog: pd.DataFrame = pd.DataFrame()
        self._product_ids: np.ndarray = self.metadata["id"].to_numpy(dtype=np.int64)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self) -> list[DuplicateCluster]:
        """
        Run the full duplicate detection pipeline.

        Returns:
            List of :class:`DuplicateCluster` objects (only clusters with
            size ≥ 2 are returned; singletons are implicitly unique).
        """
        logger.info(
            f"Running duplicate detection on {len(self.metadata)} products "
            f"(threshold={self.threshold}) …"
        )

        with utils.timer("Embedding retrieval", logger):
            embeddings = self._get_all_embeddings()

        with utils.timer("Similarity + Union-Find clustering", logger):
            uf = self._build_clusters(embeddings)

        components = uf.get_components()
        self.clusters = self._build_cluster_objects(components, embeddings)

        dup_count = sum(c.size for c in self.clusters)
        logger.info(
            f"Detection complete: {len(self.clusters)} duplicate groups "
            f"({dup_count} products involved)."
        )
        return self.clusters

    def build_unique_catalog(self) -> pd.DataFrame:
        """
        Build and return the de-duplicated product catalog.

        One representative product is kept per cluster; all other
        cluster members are removed.  Singletons (unique products)
        are kept as-is.

        Returns:
            De-duplicated :class:`pd.DataFrame`.
        """
        if not self.clusters:
            logger.warning("No clusters found. Run detect() first.")
            self.unique_catalog = self.metadata.copy()
            return self.unique_catalog

        # IDs to remove (all cluster members except the representative)
        remove_ids: set[int] = set()
        for cluster in self.clusters:
            for mid in cluster.member_ids:
                if mid != cluster.representative_id:
                    remove_ids.add(mid)

        self.unique_catalog = self.metadata[
            ~self.metadata["id"].isin(remove_ids)
        ].reset_index(drop=True)

        original = len(self.metadata)
        logger.info(
            f"Unique catalog: {len(self.unique_catalog)} products "
            f"(removed {original - len(self.unique_catalog)} duplicates "
            f"from {len(self.clusters)} clusters)."
        )
        return self.unique_catalog

    def export_results(self) -> tuple[Path, Path]:
        """
        Export the unique catalog and duplicate report CSVs.

        Returns:
            Tuple ``(unique_catalog_path, duplicate_report_path)``.
        """
        if self.unique_catalog.empty:
            self.build_unique_catalog()

        # ── Unique catalog ────────────────────────────────────────────────────
        export_cols = [
            c for c in [
                "id", "productDisplayName", "masterCategory", "subCategory",
                "articleType", "baseColour", "brand", "usage", "season",
                "year", "image_path",
            ]
            if c in self.unique_catalog.columns
        ]
        self.unique_catalog[export_cols].to_csv(config.UNIQUE_CATALOG_CSV, index=False)
        logger.info(f"Unique catalog saved → {config.UNIQUE_CATALOG_CSV}")

        # ── Duplicate report ──────────────────────────────────────────────────
        report_rows = []
        for c in self.clusters:
            for mid, mname in zip(c.member_ids, c.member_names):
                report_rows.append({
                    "cluster_id": c.cluster_id,
                    "product_id": mid,
                    "product_name": mname,
                    "is_representative": mid == c.representative_id,
                    "representative_id": c.representative_id,
                    "representative_name": c.representative_name,
                    "avg_similarity": round(c.avg_similarity, 4),
                    "cluster_size": c.size,
                })

        pd.DataFrame(report_rows).to_csv(config.DUPLICATE_REPORT_CSV, index=False)
        logger.info(f"Duplicate report saved → {config.DUPLICATE_REPORT_CSV}")

        return config.UNIQUE_CATALOG_CSV, config.DUPLICATE_REPORT_CSV

    def get_summary_stats(self) -> dict:
        """Return summary statistics about the duplicate detection run."""
        if not self.clusters:
            return {
                "total_products": len(self.metadata),
                "duplicate_clusters": 0,
                "products_in_clusters": 0,
                "unique_products": len(self.metadata),
                "reduction_percent": 0.0,
            }

        products_in_clusters = sum(c.size for c in self.clusters)
        unique = len(self.unique_catalog) if not self.unique_catalog.empty else (
            len(self.metadata) - sum(c.size - 1 for c in self.clusters)
        )
        return {
            "total_products": len(self.metadata),
            "duplicate_clusters": len(self.clusters),
            "products_in_clusters": products_in_clusters,
            "unique_products": unique,
            "reduction_percent": round(
                100 * (1 - unique / len(self.metadata)), 2
            ) if len(self.metadata) > 0 else 0.0,
            "threshold": self.threshold,
            "avg_cluster_size": round(
                products_in_clusters / len(self.clusters), 2
            ) if self.clusters else 0.0,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_all_embeddings(self) -> np.ndarray:
        """Retrieve all embeddings aligned with self.metadata rows."""
        n = len(self.metadata)
        dim = self.db.embed_dim
        embeddings = np.zeros((n, dim), dtype=np.float32)

        for i, pid in enumerate(self._product_ids):
            emb = self.db.get_embedding_by_id(int(pid))
            if emb is not None:
                embeddings[i] = emb

        return embeddings

    def _build_clusters(self, embeddings: np.ndarray) -> UnionFind:
        """
        Build a Union-Find structure from pairwise cosine similarities.

        Uses a block-wise computation to avoid OOM on large datasets.
        """
        n = len(embeddings)
        uf = UnionFind(n)
        block = 256   # process rows in blocks

        for i in range(0, n, block):
            end_i = min(i + block, n)
            block_a = embeddings[i:end_i]                    # (B, D)
            sim_matrix = block_a @ embeddings.T              # (B, N)

            for local_idx in range(end_i - i):
                global_i = i + local_idx
                sims = sim_matrix[local_idx]
                # Find indices above threshold (excluding self)
                above = np.where(sims >= self.threshold)[0]
                for j in above:
                    if int(j) != global_i:
                        uf.union(global_i, int(j))

        return uf

    def _build_cluster_objects(
        self,
        components: dict[int, list[int]],
        embeddings: np.ndarray,
    ) -> list[DuplicateCluster]:
        """Convert Union-Find components into :class:`DuplicateCluster` objects."""
        clusters: list[DuplicateCluster] = []
        cluster_id = 0

        for root, members in components.items():
            if len(members) < config.MIN_CLUSTER_SIZE:
                continue   # singleton → skip

            member_ids = [int(self._product_ids[m]) for m in members]
            member_rows = [self._get_metadata_row(pid) for pid in member_ids]

            # Elect representative: largest image file (proxy for quality)
            rep_idx = self._elect_representative(members)
            rep_pid = int(self._product_ids[rep_idx])
            rep_row = self._get_metadata_row(rep_pid)

            # Average pairwise similarity
            member_embs = embeddings[members]   # (K, D)
            avg_sim = float(np.mean(member_embs @ member_embs.T))

            clusters.append(
                DuplicateCluster(
                    cluster_id=cluster_id,
                    representative_id=rep_pid,
                    representative_name=self._row_name(rep_row),
                    representative_image=self._row_image(rep_row),
                    member_ids=member_ids,
                    member_names=[self._row_name(r) for r in member_rows],
                    avg_similarity=avg_sim,
                    size=len(members),
                )
            )
            cluster_id += 1

        logger.debug(f"Formed {len(clusters)} duplicate clusters.")
        return clusters

    def _elect_representative(self, member_indices: list[int]) -> int:
        """Return the index of the member with the largest image file."""
        best_idx = member_indices[0]
        best_size = -1
        for idx in member_indices:
            pid = int(self._product_ids[idx])
            row = self._get_metadata_row(pid)
            path_str = self._row_image(row)
            if path_str:
                try:
                    sz = Path(path_str).stat().st_size
                    if sz > best_size:
                        best_size = sz
                        best_idx = idx
                except OSError:
                    pass
        return best_idx

    def _get_metadata_row(self, product_id: int) -> Optional[pd.Series]:
        rows = self.metadata[self.metadata["id"] == product_id]
        return rows.iloc[0] if not rows.empty else None

    @staticmethod
    def _row_name(row: Optional[pd.Series]) -> str:
        if row is None:
            return "Unknown"
        return str(row.get("productDisplayName", "Unknown"))

    @staticmethod
    def _row_image(row: Optional[pd.Series]) -> str:
        if row is None:
            return ""
        return str(row.get("image_path", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import DatasetLoader
    from embedding import generate_and_cache_embeddings
    from vector_database import build_and_save_index

    loader = DatasetLoader(max_products=300).load()
    embeddings, ids, meta = generate_and_cache_embeddings(loader.df)
    db = build_and_save_index(embeddings, ids, meta)

    detector = DuplicateDetector(db, meta, threshold=0.95)
    clusters = detector.detect()
    catalog = detector.build_unique_catalog()
    stats = detector.get_summary_stats()

    print("\n=== Duplicate Detection Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if clusters:
        print(f"\nSample cluster (ID {clusters[0].cluster_id}):")
        c = clusters[0]
        print(f"  Representative: {c.representative_name} (ID {c.representative_id})")
        print(f"  Members ({c.size}): {c.member_names}")
        print(f"  Avg similarity: {c.avg_similarity:.4f}")

    catalog_path, report_path = detector.export_results()
    print(f"\nUnique catalog: {catalog_path}")
    print(f"Duplicate report: {report_path}")
