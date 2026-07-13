"""
visualization.py
================
All chart generation for the AI Product Intelligence System.

Each function returns a ``matplotlib.figure.Figure`` that can be:
  * Displayed inline in Gradio (``gr.Plot``).
  * Saved to the ``outputs/`` directory.

Functions
---------
plot_category_distribution  – Horizontal bar chart of master categories.
plot_article_type_distribution – Top-N article types.
plot_embedding_tsne         – 2-D t-SNE scatter of product embeddings.
plot_recommendation_cards   – Grid of recommended product images.
plot_duplicate_clusters     – Before/after deduplication visualisation.
plot_search_results         – Image grid for reverse-search results.
plot_performance_metrics    – Bar chart of latency / similarity metrics.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe in Gradio)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

import config
import utils

logger = utils.get_logger(__name__)

# ── Shared style ─────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"
BG_COLOR = "#0f0f1a"
CARD_BG = "#1a1a2e"
TEXT_COLOR = "#e2e8f0"
ACCENT = "#6366f1"
ACCENT2 = "#8b5cf6"
PALETTE = config.COLOR_PALETTE


def _apply_dark_theme(fig: Figure, ax_list: list) -> None:
    """Apply a consistent dark theme to a figure."""
    fig.patch.set_facecolor(BG_COLOR)
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d2d4e")


# ─────────────────────────────────────────────────────────────────────────────
# Category Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_category_distribution(stats: dict) -> Figure:
    """
    Horizontal bar chart of master category product counts.

    Args:
        stats: Output of :meth:`DatasetLoader.get_statistics`.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    cat_counts: dict = stats.get("category_counts", {})
    if not cat_counts:
        return _empty_figure("No category data available.")

    labels = list(cat_counts.keys())
    values = list(cat_counts.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.6)), dpi=config.FIG_DPI)
    bars = ax.barh(labels, values, color=colors, edgecolor="#2d2d4e", linewidth=0.5)

    # Value annotations
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center", color=TEXT_COLOR, fontsize=9,
        )

    ax.set_xlabel("Number of Products", color=TEXT_COLOR)
    ax.set_title("Product Distribution by Master Category", color=TEXT_COLOR, fontsize=13, pad=12)
    ax.invert_yaxis()
    _apply_dark_theme(fig, [ax])
    fig.tight_layout()
    return fig


def plot_article_type_distribution(stats: dict, top_n: int = 15) -> Figure:
    """
    Vertical bar chart of the most common article types.

    Args:
        stats:  Output of :meth:`DatasetLoader.get_statistics`.
        top_n:  Maximum categories to show.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    at_counts: dict = stats.get("article_type_counts", {})
    if not at_counts:
        return _empty_figure("No article type data available.")

    items = list(at_counts.items())[:top_n]
    labels = [t[0] for t in items]
    values = [t[1] for t in items]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=config.FIG_DPI)
    ax.bar(range(len(labels)), values, color=colors, edgecolor="#2d2d4e", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Count", color=TEXT_COLOR)
    ax.set_title(f"Top {top_n} Article Types", color=TEXT_COLOR, fontsize=13, pad=12)
    _apply_dark_theme(fig, [ax])
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE Embedding Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Product Embedding Space (t-SNE)",
) -> Figure:
    """
    2-D t-SNE scatter plot of product embeddings coloured by label.

    Large datasets are subsampled to ``config.TSNE_SAMPLE_SIZE`` for speed.

    Args:
        embeddings: Shape ``(N, D)`` float32 array.
        labels:     Length-N list of category / article-type strings.
        title:      Chart title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        return _empty_figure("scikit-learn not installed (needed for t-SNE).")

    n = len(embeddings)
    sample_size = min(n, config.TSNE_SAMPLE_SIZE)

    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = [labels[i] for i in idx]

    logger.info(f"Running t-SNE on {len(embeddings)} points …")

    # PCA pre-reduction for speed
    if embeddings.shape[1] > config.PCA_N_COMPONENTS:
        pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=42)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(
        n_components=2,
        perplexity=min(config.TSNE_PERPLEXITY, len(embeddings) - 1),
        n_iter=config.TSNE_N_ITER,
        random_state=42,
        verbose=0,
    )
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    label_to_color = {
        lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(unique_labels)
    }
    point_colors = [label_to_color[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=config.FIG_DPI)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=point_colors, s=18, alpha=0.7, linewidths=0,
    )

    # Legend (at most 12 categories)
    handles = [
        mpatches.Patch(color=label_to_color[l], label=l)
        for l in unique_labels[:12]
    ]
    ax.legend(
        handles=handles, loc="upper right", fontsize=7,
        facecolor=CARD_BG, edgecolor="#2d2d4e", labelcolor=TEXT_COLOR,
    )
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12)
    ax.set_xlabel("t-SNE dim 1", color=TEXT_COLOR)
    ax.set_ylabel("t-SNE dim 2", color=TEXT_COLOR)
    _apply_dark_theme(fig, [ax])
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sample Image Grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_images(
    samples: list[tuple[Image.Image, str]],
    cols: int = 4,
    title: str = "Sample Products",
) -> Figure:
    """
    Display a grid of sample product images with labels.

    Args:
        samples: List of ``(PIL.Image, label)`` tuples.
        cols:    Number of columns in the grid.
        title:   Figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    if not samples:
        return _empty_figure("No sample images to display.")

    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.2), dpi=config.FIG_DPI)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(samples):
            img, label = samples[i]
            ax.imshow(img)
            ax.set_title(
                textwrap.fill(label, 18), color=TEXT_COLOR,
                fontsize=7.5, pad=3,
            )

    fig.suptitle(title, color=TEXT_COLOR, fontsize=13, y=1.01)
    _apply_dark_theme(fig, axes.tolist())
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation Cards
# ─────────────────────────────────────────────────────────────────────────────

def plot_recommendation_cards(
    query_image: Optional[Image.Image],
    query_name: str,
    recommendations: list,    # list[RecommendationResult]
) -> Figure:
    """
    Display recommendation results as a visual card grid.

    Args:
        query_image:     PIL Image of the source product (or None).
        query_name:      Display name of the source product.
        recommendations: List of :class:`RecommendationResult`.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    n = len(recommendations)
    if n == 0:
        return _empty_figure("No recommendations to display.")

    total_cols = n + 1   # 1 for query + n for recs
    fig = plt.figure(figsize=(total_cols * 2.6, 5.5), dpi=config.FIG_DPI)
    gs = gridspec.GridSpec(2, total_cols, height_ratios=[4, 1])

    # ── Query image ───────────────────────────────────────────────────────────
    ax_q = fig.add_subplot(gs[0, 0])
    ax_q.axis("off")
    if query_image is not None:
        ax_q.imshow(query_image)
    ax_q.set_title("Query\nProduct", color="#fbbf24", fontsize=8, pad=4)

    # Arrow label
    ax_arrow = fig.add_subplot(gs[1, 0])
    ax_arrow.axis("off")
    ax_arrow.text(
        0.5, 0.5, textwrap.fill(query_name, 16),
        ha="center", va="center", color=TEXT_COLOR,
        fontsize=7, transform=ax_arrow.transAxes,
    )

    # ── Recommended products ──────────────────────────────────────────────────
    for i, rec in enumerate(recommendations):
        col = i + 1
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.axis("off")

        rec_img = utils.load_image_safe(rec.image_path, size=(224, 224))
        if rec_img is not None:
            ax_img.imshow(rec_img)

        score_color = _score_color(rec.score)
        ax_img.set_title(
            f"#{rec.rank}  {utils.format_score(rec.score)}",
            color=score_color, fontsize=8, pad=4,
        )
        # Draw colored border
        for spine in ax_img.spines.values():
            spine.set_edgecolor(score_color)
            spine.set_linewidth(2)

        ax_txt = fig.add_subplot(gs[1, col])
        ax_txt.axis("off")
        short_name = textwrap.fill(rec.product_name, 16)
        ax_txt.text(
            0.5, 0.7, short_name,
            ha="center", va="top", color=TEXT_COLOR,
            fontsize=7, transform=ax_txt.transAxes,
        )
        ax_txt.text(
            0.5, 0.2, rec.article_type,
            ha="center", va="top", color="#94a3b8",
            fontsize=6.5, transform=ax_txt.transAxes,
        )

    fig.suptitle(
        "Complementary Product Recommendations",
        color=TEXT_COLOR, fontsize=12, y=1.02,
    )
    _apply_dark_theme(fig, fig.get_axes())
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Clusters
# ─────────────────────────────────────────────────────────────────────────────

def plot_duplicate_clusters(
    clusters: list,          # list[DuplicateCluster]
    max_clusters: int = 4,
    max_members: int = 4,
) -> Figure:
    """
    Visualise duplicate clusters (before) and representatives (after).

    Args:
        clusters:     Output of :meth:`DuplicateDetector.detect`.
        max_clusters: Maximum clusters to show.
        max_members:  Maximum cluster members to show per row.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    if not clusters:
        return _empty_figure("No duplicate clusters detected.")

    show_clusters = clusters[:max_clusters]
    rows = len(show_clusters)
    cols = max_members + 2   # members + separator + representative

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 2.5, rows * 3.2),
        dpi=config.FIG_DPI,
    )
    if rows == 1:
        axes = axes.reshape(1, -1)

    for row_i, cluster in enumerate(show_clusters):
        axs = axes[row_i]

        # ── Member images ─────────────────────────────────────────────────────
        members = list(zip(cluster.member_ids, cluster.member_names))
        for col_j in range(max_members):
            ax = axs[col_j]
            ax.axis("off")
            if col_j < len(members):
                pid, pname = members[col_j]
                row_data = _find_image_path_from_cluster(cluster, pid)
                img = utils.load_image_safe(row_data, size=(112, 112)) if row_data else None
                if img:
                    ax.imshow(img)
                ax.set_title(
                    textwrap.fill(pname, 12),
                    color=TEXT_COLOR, fontsize=6.5, pad=2,
                )

        # ── Arrow separator ───────────────────────────────────────────────────
        ax_sep = axs[max_members]
        ax_sep.axis("off")
        ax_sep.text(
            0.5, 0.5, "→\nKeep\nbest",
            ha="center", va="center", color=ACCENT,
            fontsize=9, fontweight="bold",
        )

        # ── Representative ────────────────────────────────────────────────────
        ax_rep = axs[max_members + 1]
        ax_rep.axis("off")
        rep_img = utils.load_image_safe(cluster.representative_image, size=(112, 112))
        if rep_img:
            ax_rep.imshow(rep_img)
        ax_rep.set_title(
            f"✓ Representative\n{textwrap.fill(cluster.representative_name, 12)}",
            color="#22c55e", fontsize=6.5, pad=2,
        )

    fig.suptitle(
        f"Duplicate Clusters (threshold = {config.DUPLICATE_SIMILARITY_THRESHOLD})",
        color=TEXT_COLOR, fontsize=12, y=1.01,
    )
    _apply_dark_theme(fig, [ax for row in axes for ax in row])
    fig.tight_layout()
    return fig


def plot_deduplication_summary(stats: dict) -> Figure:
    """
    Bar chart summarising before/after deduplication counts.

    Args:
        stats: Output of :meth:`DuplicateDetector.get_summary_stats`.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    labels = ["Total Products", "Duplicate Products", "Unique Products"]
    values = [
        stats.get("total_products", 0),
        stats.get("products_in_clusters", 0),
        stats.get("unique_products", 0),
    ]
    colors = [PALETTE[0], "#f43f5e", "#22c55e"]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=config.FIG_DPI)
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d2d4e", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            str(val),
            ha="center", color=TEXT_COLOR, fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Product Count", color=TEXT_COLOR)
    ax.set_title("Deduplication Summary", color=TEXT_COLOR, fontsize=13, pad=12)
    _apply_dark_theme(fig, [ax])
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Search Results Grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_search_results(
    query: str,
    results: list,   # list[TextSearchResult]
    cols: int = 5,
) -> Figure:
    """
    Display reverse-search results as an image grid with metadata.

    Args:
        query:   The user's search query.
        results: List of :class:`TextSearchResult`.
        cols:    Number of columns.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    if not results:
        return _empty_figure(f"No results found for: '{query}'")

    n = len(results)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.6), dpi=config.FIG_DPI)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        r = results[i]
        img = utils.load_image_safe(r.image_path, size=(224, 224))
        if img:
            ax.imshow(img)

        score_color = _score_color(r.similarity)
        ax.set_title(
            f"#{r.rank}  {r.similarity_pct}",
            color=score_color, fontsize=8, pad=3,
        )
        # Sub-label below image
        ax.text(
            0.5, -0.02,
            textwrap.fill(r.product_name, 20),
            ha="center", va="top",
            color=TEXT_COLOR, fontsize=6.5,
            transform=ax.transAxes,
        )
        ax.text(
            0.5, -0.14,
            f"{r.article_type} · {r.base_colour}",
            ha="center", va="top",
            color="#94a3b8", fontsize=6,
            transform=ax.transAxes,
        )

    fig.suptitle(
        f'Search Results for: "{query}"',
        color=TEXT_COLOR, fontsize=12, y=1.02,
    )
    _apply_dark_theme(fig, axes.tolist())
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance_metrics(metrics_list: list) -> Figure:
    """
    Bar chart showing search latency and similarity across multiple queries.

    Args:
        metrics_list: List of :class:`SearchMetrics` objects.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    if not metrics_list:
        return _empty_figure("No metrics data available.")

    queries = [m.query[:20] for m in metrics_list]
    latencies = [m.total_time_ms for m in metrics_list]
    sims = [m.avg_similarity * 100 for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=config.FIG_DPI)

    ax1.bar(queries, latencies, color=ACCENT, edgecolor="#2d2d4e")
    ax1.set_ylabel("Latency (ms)", color=TEXT_COLOR)
    ax1.set_title("Search Latency per Query", color=TEXT_COLOR, fontsize=11)

    ax2.bar(queries, sims, color=ACCENT2, edgecolor="#2d2d4e")
    ax2.set_ylabel("Avg Similarity (%)", color=TEXT_COLOR)
    ax2.set_title("Average Similarity per Query", color=TEXT_COLOR, fontsize=11)
    ax2.set_ylim(0, 100)

    for ax in [ax1, ax2]:
        ax.set_xticklabels(queries, rotation=30, ha="right", fontsize=8)

    _apply_dark_theme(fig, [ax1, ax2])
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> Figure:
    """Return a blank figure with a centred message."""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=config.FIG_DPI)
    ax.axis("off")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", color="#94a3b8",
        fontsize=11, transform=ax.transAxes,
    )
    _apply_dark_theme(fig, [ax])
    return fig


def _score_color(score: float) -> str:
    """Return a colour string based on a similarity score."""
    if score >= 0.85:
        return "#22c55e"   # green
    if score >= 0.70:
        return "#eab308"   # yellow
    return "#f43f5e"       # red


def _find_image_path_from_cluster(cluster, product_id: int) -> Optional[str]:
    """Extract image path for a product ID stored in a DuplicateCluster."""
    idx = cluster.member_ids.index(product_id) if product_id in cluster.member_ids else -1
    if idx == -1:
        return None
    return utils.find_image_path(product_id) and str(utils.find_image_path(product_id))


def save_figure(fig: Figure, filename: str) -> Path:
    """Save a figure to the outputs directory and return its path."""
    path = config.OUTPUTS_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight", facecolor=BG_COLOR)
    logger.info(f"Figure saved → {path}")
    return path
