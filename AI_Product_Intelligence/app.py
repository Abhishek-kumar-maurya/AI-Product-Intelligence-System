"""
app.py
======
Gradio-based web application for the AI Product Intelligence System.

Tabs
----
1. Home                – Dataset overview, statistics, sample images.
2. Smart Recommendation – Select a product, view complementary suggestions.
3. Duplicate Detection  – Run deduplication, view clusters, download catalog.
4. Reverse Search       – Enter text query, retrieve matching product images.

Usage
-----
    python app.py

The app initialises the full pipeline (embeddings + FAISS index) on startup
if they are not already cached.  This may take a few minutes on the first run.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ── Project imports ───────────────────────────────────────────────────────────
import config
import utils
import visualization as viz
from preprocessing import DatasetLoader, get_dataset_loader
from embedding import generate_and_cache_embeddings, get_embedder
from vector_database import FAISSVectorDB, build_and_save_index
from recommendation import ProductRecommender
from duplicate_detection import DuplicateDetector
from reverse_search import ReverseSearchEngine, SearchMetrics

logger = utils.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global State (initialised once on startup)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """Container for all shared, lazily-initialised system objects."""

    def __init__(self) -> None:
        self.loader: Optional[DatasetLoader] = None
        self.db: Optional[FAISSVectorDB] = None
        self.recommender: Optional[ProductRecommender] = None
        self.search_engine: Optional[ReverseSearchEngine] = None
        self.stats: dict = {}
        self.embeddings: Optional[np.ndarray] = None
        self.product_ids: Optional[np.ndarray] = None
        self.ready: bool = False
        self.error_message: str = ""


STATE = AppState()


def initialise_system() -> None:
    """
    Boot the full pipeline:
      1. Load & clean dataset.
      2. Generate (or load cached) CLIP embeddings.
      3. Build (or load) FAISS index.
      4. Instantiate recommendation engine and search engine.
    """
    global STATE
    try:
        logger.info("=" * 60)
        logger.info("Initialising AI Product Intelligence System …")
        logger.info("=" * 60)

        # Step 1 – Dataset
        with utils.timer("Dataset loading", logger):
            STATE.loader = DatasetLoader(max_products=config.MAX_PRODUCTS).load()
        STATE.stats = STATE.loader.get_statistics()

        # Step 2 – Embeddings
        with utils.timer("Embeddings", logger):
            STATE.embeddings, STATE.product_ids, meta = generate_and_cache_embeddings(
                STATE.loader.df
            )

        # Step 3 – FAISS index
        with utils.timer("FAISS index", logger):
            STATE.db = build_and_save_index(STATE.embeddings, STATE.product_ids, meta)

        # Step 4 – Recommendation engine
        STATE.recommender = ProductRecommender(STATE.db, meta)

        # Step 5 – Search engine
        STATE.search_engine = ReverseSearchEngine(STATE.db, get_embedder())

        STATE.ready = True
        logger.info("System ready ✓")

    except Exception as exc:
        STATE.error_message = str(exc)
        logger.error(f"Initialisation failed: {exc}")
        logger.debug(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safety wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _check_ready() -> Optional[str]:
    """Return an error string if the system is not ready, else None."""
    if not STATE.ready:
        msg = STATE.error_message or "System not initialised. Please check the dataset path."
        return msg
    return None


def _pil_or_placeholder(path: str, size: tuple = (224, 224)) -> Image.Image:
    """Load a PIL image or return a grey placeholder."""
    img = utils.load_image_safe(path, size=size)
    if img is None:
        placeholder = Image.new("RGB", size, color=(40, 40, 60))
        return placeholder
    return img


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Home
# ─────────────────────────────────────────────────────────────────────────────

def home_get_stats() -> tuple[str, plt.Figure, plt.Figure, plt.Figure]:
    """
    Return dataset statistics and overview charts for the Home tab.

    Returns:
        Tuple of (stats_markdown, category_chart, article_type_chart, sample_grid).
    """
    err = _check_ready()
    if err:
        empty = viz._empty_figure(f"System error: {err}")
        return f"❌ {err}", empty, empty, empty

    stats = STATE.stats
    md = f"""
### 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Products** | {stats.get('total_products', 0):,} |
| **Raw Records** | {stats.get('raw_products', 0):,} |
| **Removed (missing/corrupted)** | {stats.get('missing_before', 0):,} |
| **Master Categories** | {stats.get('master_categories', 0)} |
| **Article Types** | {stats.get('article_types', 0)} |
| **Seasons** | {stats.get('seasons', 0)} |
| **CLIP Model** | `{config.CLIP_MODEL_NAME}` |
| **Embedding Dim** | 512 |
| **FAISS Index** | IndexFlatIP (cosine) |
| **Duplicate Threshold** | {config.DUPLICATE_SIMILARITY_THRESHOLD} |
"""

    cat_chart = viz.plot_category_distribution(stats)
    at_chart = viz.plot_article_type_distribution(stats, top_n=15)
    samples = STATE.loader.get_sample_images(n=config.SAMPLE_DISPLAY)
    sample_fig = viz.plot_sample_images(samples, cols=4, title="Random Product Samples")

    return md, cat_chart, at_chart, sample_fig


def home_get_tsne() -> plt.Figure:
    """Generate t-SNE visualisation of product embeddings."""
    err = _check_ready()
    if err:
        return viz._empty_figure(f"System error: {err}")

    labels = STATE.loader.df["masterCategory"].tolist()
    return viz.plot_embedding_tsne(
        STATE.embeddings, labels,
        title="Product Embedding Space (t-SNE by Master Category)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Smart Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def get_product_choices() -> list[str]:
    """Return all product display names for the dropdown."""
    if not STATE.ready:
        return ["System not ready"]
    return STATE.recommender.list_product_names()


def recommend_by_name(
    product_name: str,
    top_k: int,
) -> tuple[Image.Image, str, plt.Figure, str]:
    """
    Gradio handler for the recommendation tab.

    Args:
        product_name: Selected product display name.
        top_k:        Number of recommendations (slider value).

    Returns:
        Tuple of (query_image, info_markdown, recommendation_figure, metrics_text).
    """
    err = _check_ready()
    placeholder = Image.new("RGB", (224, 224), (40, 40, 60))

    if err:
        return placeholder, f"❌ {err}", viz._empty_figure(""), ""

    if not product_name or product_name == "System not ready":
        return placeholder, "Please select a product.", viz._empty_figure(""), ""

    try:
        # Find the product
        recs = STATE.recommender.recommend_by_name(product_name, top_k=int(top_k))

        # Source product image
        mask = STATE.loader.df["productDisplayName"] == product_name
        if not mask.any():
            mask = STATE.loader.df["productDisplayName"].str.lower().str.contains(
                product_name.lower(), na=False
            )
        source_row = STATE.loader.df[mask].iloc[0] if mask.any() else None
        source_img = None
        if source_row is not None:
            source_img = _pil_or_placeholder(str(source_row.get("image_path", "")))

        # Markdown info
        if source_row is not None:
            info_md = f"""
**Selected Product**
- **Name:** {source_row.get('productDisplayName', 'N/A')}
- **Category:** {source_row.get('masterCategory', 'N/A')} → {source_row.get('subCategory', 'N/A')}
- **Type:** {source_row.get('articleType', 'N/A')}
- **Colour:** {source_row.get('baseColour', 'N/A')}
- **Usage:** {source_row.get('usage', 'N/A')}

**Found {len(recs)} complementary recommendations**
"""
        else:
            info_md = f"Found {len(recs)} recommendations."

        # Metrics
        if recs:
            avg_score = np.mean([r.score for r in recs])
            metrics_text = (
                f"Avg confidence: {avg_score:.3f} | "
                f"Engine: Rule-Based + CLIP Embedding Similarity"
            )
        else:
            metrics_text = "No recommendations found."

        # Build recommendation table
        rec_table_md = "\n### Recommendations\n"
        rec_table_md += "| Rank | Product | Type | Score | Reason |\n"
        rec_table_md += "|------|---------|------|-------|--------|\n"
        for r in recs:
            rec_table_md += (
                f"| {r.rank} | {r.product_name[:35]} | {r.article_type} "
                f"| {r.score:.3f} | {r.reason[:60]} |\n"
            )
        info_md += rec_table_md

        # Figure
        fig = viz.plot_recommendation_cards(source_img, product_name, recs)

        return source_img or placeholder, info_md, fig, metrics_text

    except Exception as exc:
        logger.error(f"Recommendation error: {exc}")
        return placeholder, f"❌ Error: {exc}", viz._empty_figure("Error"), ""


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Duplicate Detection
# ─────────────────────────────────────────────────────────────────────────────

def run_duplicate_detection(
    threshold: float,
) -> tuple[str, plt.Figure, plt.Figure, Optional[str]]:
    """
    Gradio handler for the duplicate detection tab.

    Args:
        threshold: Cosine similarity threshold (slider value).

    Returns:
        Tuple of (stats_markdown, cluster_figure, summary_chart, catalog_csv_path).
    """
    err = _check_ready()
    if err:
        empty = viz._empty_figure("")
        return f"❌ {err}", empty, empty, None

    try:
        detector = DuplicateDetector(
            STATE.db, STATE.loader.df, threshold=float(threshold)
        )
        clusters = detector.detect()
        catalog = detector.build_unique_catalog()
        stats = detector.get_summary_stats()
        catalog_path, _ = detector.export_results()

        stats_md = f"""
### Duplicate Detection Results (threshold = {threshold})

| Metric | Value |
|--------|-------|
| **Total Products Analysed** | {stats.get('total_products', 0):,} |
| **Duplicate Clusters Found** | {stats.get('duplicate_clusters', 0)} |
| **Products in Clusters** | {stats.get('products_in_clusters', 0)} |
| **Unique Products** | {stats.get('unique_products', 0):,} |
| **Catalog Reduction** | {stats.get('reduction_percent', 0)}% |
| **Avg Cluster Size** | {stats.get('avg_cluster_size', 0)} |

{'✅ **No duplicates found** — catalog is already unique!' if not clusters else f"⚠️ Found **{len(clusters)} duplicate group(s)**. Clean catalog saved."}
"""

        cluster_fig = viz.plot_duplicate_clusters(clusters, max_clusters=4, max_members=4)
        summary_fig = viz.plot_deduplication_summary(stats)

        return stats_md, cluster_fig, summary_fig, str(catalog_path)

    except Exception as exc:
        logger.error(f"Duplicate detection error: {exc}")
        empty = viz._empty_figure("")
        return f"❌ Error: {exc}", empty, empty, None


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Reverse Search
# ─────────────────────────────────────────────────────────────────────────────

def run_reverse_search(
    query: str,
    top_k: int,
    filter_category: str,
) -> tuple[str, plt.Figure, str]:
    """
    Gradio handler for the reverse search tab.

    Args:
        query:           Text query from the user.
        top_k:           Number of results.
        filter_category: Optional master category filter ("All" = no filter).

    Returns:
        Tuple of (results_markdown, results_figure, metrics_text).
    """
    err = _check_ready()
    if err:
        return f"❌ {err}", viz._empty_figure(""), ""

    if not query or not query.strip():
        return "⚠️ Please enter a search query.", viz._empty_figure(""), ""

    try:
        filters = None
        if filter_category and filter_category != "All":
            filters = {"master_category": filter_category}

        results, metrics = STATE.search_engine.search(
            query.strip(), top_k=int(top_k), filters=filters
        )

        if not results:
            return (
                f"No results found for: **{query}**",
                viz._empty_figure(f"No results for '{query}'"),
                "",
            )

        # Markdown table
        md = f"### 🔍 Top {len(results)} results for: **\"{query}\"**\n\n"
        md += "| Rank | Product | Type | Category | Colour | Similarity |\n"
        md += "|------|---------|------|----------|--------|------------|\n"
        for r in results:
            md += (
                f"| {r.rank} | {r.product_name[:35]} | {r.article_type} "
                f"| {r.master_category} | {r.base_colour} | **{r.similarity_pct}** |\n"
            )

        fig = viz.plot_search_results(query, results, cols=min(top_k, 5))

        metrics_text = (
            f"🕐 Total: {metrics.total_time_ms:.1f}ms  |  "
            f"CLIP encode: {metrics.inference_time_ms:.1f}ms  |  "
            f"FAISS: {metrics.retrieval_time_ms:.1f}ms  |  "
            f"Avg similarity: {utils.format_score(metrics.avg_similarity)}  |  "
            f"Max: {utils.format_score(metrics.max_similarity)}"
        )

        return md, fig, metrics_text

    except Exception as exc:
        logger.error(f"Search error: {exc}")
        return f"❌ Error: {exc}", viz._empty_figure("Error"), ""


def get_category_choices() -> list[str]:
    """Return master categories for the search filter dropdown."""
    if not STATE.ready:
        return ["All"]
    cats = ["All"] + sorted(STATE.loader.df["masterCategory"].dropna().unique().tolist())
    return cats


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI Definition
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ── */
body { font-family: 'Inter', 'Segoe UI', sans-serif; }
.gradio-container { background: #0f0f1a !important; }

/* ── Cards ── */
.gr-block, .gr-box { background: #1a1a2e !important; border: 1px solid #2d2d4e !important; border-radius: 12px !important; }

/* ── Buttons ── */
button.primary { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
button.primary:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Tab labels ── */
.tab-nav button { color: #94a3b8 !important; font-weight: 500 !important; }
.tab-nav button.selected { color: #6366f1 !important; border-bottom: 2px solid #6366f1 !important; }

/* ── Markdown ── */
.prose { color: #e2e8f0 !important; }
.prose table { border-collapse: collapse; }
.prose td, .prose th { border: 1px solid #2d2d4e !important; padding: 6px 12px; }

/* ── Sliders / inputs ── */
input[type=range] { accent-color: #6366f1; }
textarea, input[type=text] { background: #1a1a2e !important; color: #e2e8f0 !important; border: 1px solid #2d2d4e !important; border-radius: 8px !important; }
"""

HEADER_HTML = """
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 32px 24px; border-radius: 16px; text-align: center; margin-bottom: 8px;
    border: 1px solid #2d2d4e;
">
    <h1 style="
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 800; margin: 0 0 8px;
        font-family: 'Inter', sans-serif;
    ">🛍️ AI Product Intelligence System</h1>
    <p style="color: #94a3b8; font-size: 1rem; margin: 0; font-family: 'Inter', sans-serif;">
        Smart Recommendations · Duplicate Detection · Reverse Search
        <br><span style="font-size: 0.85rem; color: #64748b;">
        Powered by OpenAI CLIP + FAISS · Gen AI Bootcamp Day 2
        </span>
    </p>
</div>
"""


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="AI Product Intelligence System",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — HOME
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🏠 Home", id="tab_home"):
                gr.Markdown(
                    "## Dataset Overview\n"
                    "Load dataset statistics, category distributions, and sample products."
                )
                btn_load = gr.Button("📊 Load Dataset Stats", variant="primary")

                with gr.Row():
                    stats_md = gr.Markdown("*Click the button above to load stats.*")

                with gr.Row():
                    cat_chart = gr.Plot(label="Category Distribution")
                    at_chart = gr.Plot(label="Article Type Distribution")

                sample_grid = gr.Plot(label="Sample Products")

                gr.Markdown("---")
                gr.Markdown("### 🔮 Embedding Space (t-SNE)")
                btn_tsne = gr.Button("Generate t-SNE Visualisation", variant="secondary")
                tsne_plot = gr.Plot(label="t-SNE Embedding Space")

                btn_load.click(
                    fn=home_get_stats,
                    inputs=[],
                    outputs=[stats_md, cat_chart, at_chart, sample_grid],
                )
                btn_tsne.click(
                    fn=home_get_tsne,
                    inputs=[],
                    outputs=[tsne_plot],
                )

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — SMART RECOMMENDATION
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🎯 Smart Recommendation", id="tab_rec"):
                gr.Markdown(
                    "## Complementary Product Recommendation\n"
                    "Select a product to see what items pair well with it "
                    "(socks with shoes, belt with trousers, etc.)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        product_dropdown = gr.Dropdown(
                            choices=[], label="Select a Product",
                            info="Choose any product from the catalog",
                            interactive=True,
                        )
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Number of Recommendations",
                        )
                        btn_recommend = gr.Button("✨ Get Recommendations", variant="primary")
                        metrics_text = gr.Textbox(
                            label="Engine Metrics", interactive=False, lines=1
                        )
                        query_img = gr.Image(label="Selected Product", height=250)

                    with gr.Column(scale=2):
                        rec_info = gr.Markdown("*Select a product and click Get Recommendations.*")
                        rec_fig = gr.Plot(label="Recommendation Cards")

                # Populate dropdown on tab load
                product_dropdown.choices = get_product_choices()

                btn_recommend.click(
                    fn=recommend_by_name,
                    inputs=[product_dropdown, top_k_slider],
                    outputs=[query_img, rec_info, rec_fig, metrics_text],
                )

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 — DUPLICATE DETECTION
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🔍 Duplicate Detection", id="tab_dup"):
                gr.Markdown(
                    "## Unique Product Catalog Creation\n"
                    "Identifies near-duplicate products from different sellers "
                    "and produces a clean, unique catalog."
                )
                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=0.80, maximum=0.99, value=config.DUPLICATE_SIMILARITY_THRESHOLD,
                        step=0.01,
                        label="Similarity Threshold",
                        info="Higher = stricter (fewer duplicates detected)",
                    )
                    btn_detect = gr.Button("🔍 Run Duplicate Detection", variant="primary")

                dup_stats_md = gr.Markdown("*Click Run to start detection.*")

                with gr.Row():
                    cluster_fig = gr.Plot(label="Duplicate Clusters")
                    summary_fig = gr.Plot(label="Deduplication Summary")

                catalog_file = gr.File(label="⬇️ Download Unique Catalog (CSV)")

                btn_detect.click(
                    fn=run_duplicate_detection,
                    inputs=[threshold_slider],
                    outputs=[dup_stats_md, cluster_fig, summary_fig, catalog_file],
                )

            # ══════════════════════════════════════════════════════════════════
            # TAB 4 — REVERSE SEARCH
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🔎 Reverse Search", id="tab_search"):
                gr.Markdown(
                    "## Natural Language Product Search\n"
                    "Describe any product in plain English and find matching items "
                    "using CLIP semantic embeddings."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g. blue casual shirt, red running shoes, leather handbag …",
                            lines=2,
                        )
                        with gr.Row():
                            search_top_k = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="Top-K Results",
                            )
                            search_filter = gr.Dropdown(
                                choices=get_category_choices(),
                                value="All",
                                label="Filter by Category",
                            )
                        btn_search = gr.Button("🔎 Search Products", variant="primary")

                    with gr.Column(scale=1):
                        search_metrics = gr.Textbox(
                            label="Performance Metrics", interactive=False, lines=2
                        )

                search_results_md = gr.Markdown("*Enter a query and click Search.*")
                search_fig = gr.Plot(label="Search Results")

                # Example queries
                gr.Examples(
                    examples=[
                        ["blue casual shirt"],
                        ["running sports shoes"],
                        ["black leather formal shoes"],
                        ["red dress"],
                        ["sports watch fitness"],
                        ["casual summer shorts"],
                        ["women kurta ethnic"],
                        ["denim jeans slim fit"],
                    ],
                    inputs=[search_input],
                    label="Example Queries",
                )

                btn_search.click(
                    fn=run_reverse_search,
                    inputs=[search_input, search_top_k, search_filter],
                    outputs=[search_results_md, search_fig, search_metrics],
                )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("""
<div style="text-align:center; padding: 16px; color: #475569; font-size: 0.8rem; margin-top: 8px;">
    AI Product Intelligence System · Gen AI Bootcamp Day 2 ·
    Built with OpenAI CLIP + FAISS + Gradio
</div>
""")

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialise backend
    initialise_system()

    if not STATE.ready:
        print(
            "\n❌  System initialisation failed.\n"
            f"   Error: {STATE.error_message}\n\n"
            "   Please ensure:\n"
            f"   1. Dataset (styles.csv + images/) is in: {config.DATA_DIR}\n"
            "   2. All dependencies are installed: pip install -r requirements.txt\n"
            "   3. Download dataset from Kaggle:\n"
            "      https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small\n"
        )
        sys.exit(1)

    # Build and launch Gradio UI
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
