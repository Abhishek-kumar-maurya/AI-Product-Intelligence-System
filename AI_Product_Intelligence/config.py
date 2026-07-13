"""
config.py
=========
Central configuration file for the AI Product Intelligence System.
All paths, hyperparameters, model names, and category mappings live here.
Modify this file to adjust system behaviour without touching module code.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# BASE DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data"
IMAGES_DIR: Path = DATA_DIR / "images"          # raw product images
EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings"  # cached .npy files
FAISS_INDEX_DIR: Path = BASE_DIR / "faiss_index"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"
MODELS_DIR: Path = BASE_DIR / "models"
REPORT_DIR: Path = BASE_DIR / "report"

# Ensure all directories exist on import
for _d in [DATA_DIR, IMAGES_DIR, EMBEDDINGS_DIR, FAISS_INDEX_DIR,
           OUTPUTS_DIR, SCREENSHOTS_DIR, MODELS_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET FILES
# ─────────────────────────────────────────────────────────────────────────────

STYLES_CSV: Path = DATA_DIR / "styles.csv"          # main metadata file
IMAGES_CSV: Path = DATA_DIR / "images.csv"          # image filename mapping

# ─────────────────────────────────────────────────────────────────────────────
# CLIP MODEL
# ─────────────────────────────────────────────────────────────────────────────

CLIP_MODEL_NAME: str = "ViT-B/32"   # OpenAI CLIP variant
CLIP_DEVICE: str = "cpu"            # "cuda" if GPU available; auto-detected in embedding.py
CLIP_BATCH_SIZE: int = 32           # images per forward pass
IMAGE_SIZE: int = 224               # CLIP expected input resolution

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING CACHE FILES
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EMBEDDINGS_FILE: Path = EMBEDDINGS_DIR / "image_embeddings.npy"
PRODUCT_IDS_FILE: Path = EMBEDDINGS_DIR / "product_ids.npy"
METADATA_CACHE_FILE: Path = EMBEDDINGS_DIR / "metadata.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# FAISS INDEX FILES
# ─────────────────────────────────────────────────────────────────────────────

FAISS_INDEX_FILE: Path = FAISS_INDEX_DIR / "product_index.faiss"
FAISS_METADATA_FILE: Path = FAISS_INDEX_DIR / "product_metadata.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILES
# ─────────────────────────────────────────────────────────────────────────────

UNIQUE_CATALOG_CSV: Path = OUTPUTS_DIR / "unique_catalog.csv"
DUPLICATE_REPORT_CSV: Path = OUTPUTS_DIR / "duplicate_report.csv"

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MAX_PRODUCTS: int = 5000   # set to None to use full dataset (slower)
SAMPLE_DISPLAY: int = 12   # images shown in Home tab grid

# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

DUPLICATE_SIMILARITY_THRESHOLD: float = 0.92   # cosine ≥ this → duplicate
MIN_CLUSTER_SIZE: int = 2                       # ignore singletons

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_RECOMMENDATIONS: int = 5       # number of complementary products
RECOMMENDATION_FALLBACK_K: int = 20  # FAISS candidates before filtering

# ─────────────────────────────────────────────────────────────────────────────
# REVERSE SEARCH
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_SEARCH_RESULTS: int = 5

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEMENTARY CATEGORY MAPPING
# ─────────────────────────────────────────────────────────────────────────────
# Maps an article type → list of complementary article types.
# Keys and values are lower-cased for case-insensitive matching.

COMPLEMENTARY_MAPPING: dict = {
    # ── Footwear ──────────────────────────────────────────────────────────
    "shoes": ["socks", "watch", "sports shoes", "casual shoes", "shorts", "trousers"],
    "sports shoes": ["socks", "shorts", "track pants", "sports watch", "water bottle", "cap"],
    "casual shoes": ["jeans", "shorts", "belt", "sunglasses", "watch"],
    "formal shoes": ["trousers", "belt", "tie", "cufflinks", "socks"],
    "heels": ["skirts", "dresses", "clutches", "kurtas"],
    "sandals": ["shorts", "skirts", "dresses", "sunglasses"],
    "flip flops": ["shorts", "t-shirts", "sunglasses"],
    "boots": ["jeans", "trousers", "jacket", "scarf"],

    # ── Tops ──────────────────────────────────────────────────────────────
    "shirts": ["trousers", "jeans", "belt", "watch", "formal shoes", "tie"],
    "t-shirts": ["jeans", "shorts", "casual shoes", "watch", "cap"],
    "tops": ["jeans", "skirts", "flats", "clutches", "earrings"],
    "kurtas": ["churidars", "leggings", "flats", "earrings", "dupatta"],
    "blouses": ["saree", "skirts", "trousers", "earrings"],
    "sweatshirts": ["track pants", "casual shoes", "cap", "backpacks"],
    "jackets": ["jeans", "shirts", "boots", "scarf", "watch"],
    "sweaters": ["jeans", "trousers", "boots", "scarf", "watch"],
    "hoodies": ["track pants", "casual shoes", "cap", "backpacks"],
    "dresses": ["heels", "sandals", "clutches", "earrings", "belt"],
    "sarees": ["blouses", "flats", "earrings", "clutches"],

    # ── Bottoms ───────────────────────────────────────────────────────────
    "jeans": ["shirts", "t-shirts", "casual shoes", "belt", "watch"],
    "trousers": ["shirts", "formal shoes", "belt", "tie", "watch"],
    "shorts": ["t-shirts", "sports shoes", "socks", "cap", "sunglasses"],
    "skirts": ["tops", "blouses", "heels", "sandals", "earrings"],
    "leggings": ["kurtas", "tops", "flats", "earrings"],
    "churidars": ["kurtas", "flats", "earrings", "dupatta"],
    "track pants": ["sweatshirts", "sports shoes", "socks", "cap"],

    # ── Accessories ───────────────────────────────────────────────────────
    "watch": ["shirts", "formal shoes", "belt", "sunglasses"],
    "sunglasses": ["t-shirts", "casual shoes", "cap", "shorts"],
    "belt": ["shirts", "trousers", "jeans", "formal shoes"],
    "cap": ["t-shirts", "shorts", "sports shoes", "sunglasses"],
    "backpacks": ["hoodies", "track pants", "casual shoes"],
    "handbags": ["dresses", "tops", "heels", "sunglasses"],
    "clutches": ["dresses", "heels", "earrings"],
    "earrings": ["tops", "dresses", "kurtas"],
    "bracelet": ["watch", "shirts", "casual shoes"],
    "tie": ["shirts", "formal shoes", "trousers"],
    "scarf": ["jackets", "boots", "sweaters"],
    "socks": ["sports shoes", "casual shoes", "formal shoes"],
    "dupatta": ["kurtas", "churidars"],
}

# Fallback categories when no mapping exists
DEFAULT_COMPLEMENTARY_CATEGORIES: list = [
    "accessories", "watches", "sunglasses", "belts", "bags"
]

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

TSNE_PERPLEXITY: int = 30
TSNE_N_ITER: int = 1000
TSNE_SAMPLE_SIZE: int = 500   # subsample for speed
PCA_N_COMPONENTS: int = 50    # PCA before t-SNE

FIG_DPI: int = 120
COLOR_PALETTE: list = [
    "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316",
    "#eab308", "#22c55e", "#14b8a6", "#0ea5e9", "#3b82f6",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
