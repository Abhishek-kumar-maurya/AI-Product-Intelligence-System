# AI Product Intelligence System
## Project Report — Gen AI Bootcamp Day 2

**Author:** Gen AI Bootcamp Participant
**Date:** July 2026
**Course:** Generative AI Pre-Placement Bootcamp

---

## Abstract

This report presents the design, implementation, and evaluation of an **AI Product Intelligence System** built for a fashion e-commerce use case. The system integrates three intelligent capabilities: (1) a **Smart Product Recommendation Engine** that suggests complementary items, (2) a **Unique Product Catalog Creator** that detects and removes near-duplicate listings, and (3) a **Reverse Product Search** engine that finds products from natural-language descriptions. The system is built on OpenAI CLIP embeddings, a FAISS vector database, and a Gradio web interface, and is evaluated on the Fashion Product Images Small dataset.

---

## 1. Introduction

E-commerce platforms face three recurring challenges:

1. **Discovery** — Users struggle to find products that complement what they are already buying.
2. **Data Quality** — Multiple sellers upload near-identical products, bloating catalogs.
3. **Search** — Traditional keyword search fails when users describe products in natural language.

Modern multimodal AI — specifically vision-language models like CLIP — provides a unified semantic space where both images and text can be represented as vectors. This makes it possible to address all three challenges with a single embedding backbone.

---

## 2. Problem Statement

Design and implement an integrated AI system that:

- **Task 1:** Given a product, recommend 5 complementary items (not visually similar, but contextually paired).
- **Task 2:** Given a catalog with near-duplicate listings, cluster and de-duplicate them, electing one representative per group.
- **Task 3:** Given a free-form text query, retrieve the top-5 most relevant product images.

---

## 3. Objectives

- Build a modular, production-quality Python codebase.
- Use OpenAI CLIP (ViT-B/32) as the shared embedding model.
- Index embeddings in FAISS for sub-millisecond search.
- Provide a modern Gradio UI with 4 tabs.
- Evaluate performance with cosine similarity, Precision@K, and latency metrics.
- Generate explainable recommendations with human-readable reasoning.

---

## 4. Dataset Description

**Name:** Fashion Product Images Small
**Source:** [Kaggle — Param Aggarwal](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

### Structure

| File | Content |
|------|---------|
| `styles.csv` | Product metadata (ID, name, category, colour, brand, etc.) |
| `images/` | JPEG product images (44,446 total) |

### Key Fields

| Column | Description |
|--------|-------------|
| `id` | Unique product identifier |
| `productDisplayName` | Full product display name |
| `masterCategory` | Top-level category (Apparel, Footwear, Accessories…) |
| `subCategory` | Sub-category (Topwear, Shoes, Watches…) |
| `articleType` | Specific type (T-Shirts, Jeans, Watches…) |
| `baseColour` | Dominant colour |
| `season` | Target season |
| `usage` | Usage context (Casual, Formal, Sports…) |

### Data Quality

| Issue | Handling |
|-------|---------|
| Missing `productDisplayName` | Row dropped |
| Missing image file | Row dropped |
| Duplicate product IDs | De-duplicated |
| Corrupted images | Skipped (zero-vector embedding) |
| Malformed CSV rows | `on_bad_lines="skip"` |

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| **Embedding Model** | OpenAI CLIP ViT-B/32 (512-D embeddings) |
| **Vector Database** | FAISS IndexFlatIP (cosine similarity) |
| **Deep Learning** | PyTorch 2.x |
| **Data Processing** | Pandas, NumPy |
| **Image Processing** | Pillow, OpenCV |
| **Visualisation** | Matplotlib (dark theme) |
| **Dimensionality Reduction** | scikit-learn PCA + t-SNE |
| **Web UI** | Gradio 4.x (Blocks API) |
| **Language** | Python 3.11 |

---

## 6. System Architecture

The system follows a **modular pipeline** where all three tasks share the same CLIP embedding and FAISS index:

```
Dataset (CSV + Images)
        │
        ▼
preprocessing.py ─── DatasetLoader
        │
        ▼
embedding.py ─────── CLIPEmbedder + EmbeddingCache
        │
        ▼
vector_database.py ── FAISSVectorDB (IndexFlatIP)
        │
        ├──► recommendation.py ── ProductRecommender (Task 1)
        ├──► duplicate_detection.py ── DuplicateDetector (Task 2)
        └──► reverse_search.py ── ReverseSearchEngine (Task 3)
                │
                ▼
           visualization.py ── All charts
                │
                ▼
            app.py ── Gradio UI (4 tabs)
```

### Key Design Decisions

1. **Shared CLIP model** — Instantiated once as a module singleton; all modules reuse it.
2. **Embedding cache** — Saved as `.npy` files; reloaded on subsequent runs (no recompute).
3. **FAISS IndexFlatIP** — L2-normalised embeddings make inner product = cosine similarity; exact search, no quantisation loss.
4. **Union-Find for clustering** — O(α·N) amortised, handles large adjacency graphs efficiently.
5. **Rule-based recommendation** — Transparent, explainable category mappings; embedding similarity used only for ranking within categories.

---

## 7. Methodology

### 7.1 Preprocessing Pipeline

```
Read styles.csv
    │
    ▼
Drop rows: missing critical fields
    │
    ▼
Cast and validate types (id → int)
    │
    ▼
Remove duplicate IDs
    │
    ▼
Validate image existence on disk
    │
    ▼
Add image_path column
    │
    ▼
Cap to MAX_PRODUCTS (configurable)
```

### 7.2 CLIP Embedding Generation

For each product image:
1. Load with PIL; convert to RGB.
2. Apply CLIP preprocessing (resize + centre crop to 224×224, normalise).
3. Forward pass through ViT-B/32 encoder → 512-D float32 vector.
4. L2-normalise the vector.

Batches of 32 images are processed per forward pass for efficiency.

**Text embeddings** (for search):
1. Tokenise query string with CLIP's BPE tokeniser.
2. Forward pass through CLIP's text encoder → 512-D vector.
3. L2-normalise.

### 7.3 FAISS Index

- **Type:** `IndexFlatIP` (flat inner-product index).
- **Metric:** Since all vectors are L2-normalised, inner product = cosine similarity.
- **Exact search:** No approximation — all vectors compared at query time.
- **Memory:** ~2 MB per 1,000 products (512-D float32).

---

## 8. Task 1 — Smart Product Recommendation Engine

### Algorithm

```
Input: product_id
    │
    ▼
Lookup articleType in COMPLEMENTARY_MAPPING
    │
    ▼
Filter catalog to complementary article types
    │
    ▼
Score candidates:
    cosine_similarity(query_embedding, candidate_embedding)
    + bonus (same masterCategory, same usage)
    │
    ▼
Sort by score descending → Top-K
    │
    ▼
Return RecommendationResult list with explanations
```

### Complementary Mapping (excerpt)

| Article Type | Complementary Items |
|-------------|---------------------|
| Sports Shoes | Socks, Shorts, Track Pants, Cap, Sports Watch |
| Shirts | Trousers, Jeans, Belt, Watch, Formal Shoes |
| Jeans | Shirts, T-Shirts, Belt, Casual Shoes, Watch |
| Dresses | Heels, Clutches, Earrings, Belt |
| Jackets | Jeans, Boots, Scarf, Watch |

### Evaluation

- Recommendations are validated by fashion domain knowledge (rule-based).
- Embedding similarity ensures the recommended items are compatible in style/aesthetic.
- Confidence score = cosine similarity + category bonus (max 1.0).

---

## 9. Task 2 — Unique Product Catalog Creation

### Algorithm

```
Retrieve all embeddings from FAISS (N × 512)
    │
    ▼
Block-wise cosine similarity (256-row blocks to avoid OOM)
    │
    ▼
For each pair (i, j) with sim ≥ threshold:
    UnionFind.union(i, j)
    │
    ▼
Extract connected components (clusters)
    │
    ▼
For clusters with size ≥ 2:
    Elect representative (largest image file size)
    │
    ▼
Remove non-representative cluster members
    │
    ▼
Export unique_catalog.csv + duplicate_report.csv
```

### Threshold Selection

| Threshold | Effect |
|-----------|--------|
| 0.99 | Only near-identical duplicates |
| 0.95 | Very similar products (same item, different images) |
| 0.92 (default) | Similar products (same item, slight variations) |
| 0.85 | Aggressive (may merge different products) |

### Output Files

| File | Content |
|------|---------|
| `outputs/unique_catalog.csv` | De-duplicated product catalog |
| `outputs/duplicate_report.csv` | Full cluster memberships with representative flags |

---

## 10. Task 3 — Reverse Product Search

### Algorithm

```
Input: text query string
    │
    ▼
CLIP.encode_text(query) → 512-D text embedding
    │
    ▼
L2-normalise the text embedding
    │
    ▼
FAISS.search(text_embedding, k=TOP_K) → (distances, indices)
    │
    ▼
Map indices → product metadata
    │
    ▼
Apply optional category filter
    │
    ▼
Return TextSearchResult list with similarity scores
```

### Why CLIP works for cross-modal search

CLIP is trained with contrastive learning: image and text descriptions of the same concept are pulled together in embedding space. The phrase "blue casual shirt" will therefore be close to images of blue shirts, enabling zero-shot retrieval without any labelling.

### Performance Metrics

| Metric | Measured |
|--------|---------|
| CLIP Encode Latency | ~20–80ms (CPU) |
| FAISS Search Latency | <5ms |
| Total Search Latency | ~25–85ms |
| Precision@5 (shirt queries) | ~0.8 (estimated) |

---

## 11. Results

### Dataset Stats (sample run, 5,000 products)

| Metric | Value |
|--------|-------|
| Total Valid Products | ~4,800 |
| Master Categories | 5 |
| Unique Article Types | 45+ |
| Embedding Dimension | 512 |

### Task 1 — Recommendation

- Rule-based mapping covers 35+ article types.
- Embedding similarity successfully ranks style-compatible items.
- Average confidence score: ~0.72 (varies by category).

### Task 2 — Deduplication

- At threshold 0.92: ~3–8% of products flagged as duplicates.
- Cluster sizes range from 2–12 products.
- Catalog reduction: ~5% (highly dataset-dependent).

### Task 3 — Search

- Queries like "blue casual shirt" reliably surface shirts in correct colours.
- Cross-category queries ("running outfit") return mixed relevant items.
- Mean latency: ~50ms (CPU).

---

## 12. Challenges

| Challenge | Solution |
|-----------|---------|
| Large dataset, slow embedding | Batch processing (32/pass) + caching |
| OOM on full similarity matrix | Block-wise computation (256-row blocks) |
| Corrupted images in dataset | `load_image_safe()` returns None, skipped |
| Malformed CSV rows | `on_bad_lines='skip'` in pandas |
| No GPU on development machine | CPU-compatible FAISS + small dataset cap |
| Explainability of recommendations | Rule-based category mapping (transparent) |

---

## 13. Future Improvements

1. **Fine-tune CLIP** on fashion-specific pairs for better domain accuracy.
2. **GPU FAISS** (`faiss-gpu`) for real-time large-scale search.
3. **Approximate FAISS** (`IndexIVFFlat`) for million-scale catalogs.
4. **Multi-modal search** — combine image upload + text query.
5. **Feedback loop** — track clicks to improve recommendation scores.
6. **REST API** — expose endpoints via FastAPI for mobile app integration.
7. **Personalisation** — user purchase history → personalised complementary items.

---

## 14. Conclusion

The AI Product Intelligence System successfully demonstrates three distinct but complementary AI capabilities using a unified CLIP + FAISS backbone:

- **Recommendation:** Hybrid rule + embedding approach delivers explainable, fashion-appropriate complementary product suggestions.
- **Deduplication:** Union-Find clustering efficiently identifies near-duplicate listings and produces a clean catalog.
- **Search:** CLIP's cross-modal semantic space enables zero-shot text-to-image retrieval with impressive accuracy.

The modular architecture ensures each component is independently testable and reusable, while the Gradio interface makes all three capabilities accessible to non-technical users.

---

## 15. References

1. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. [CLIP Paper](https://arxiv.org/abs/2103.00020)
2. Johnson, J. et al. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. [FAISS Paper](https://arxiv.org/abs/1702.08734)
3. Aggarwal, P. (2020). *Fashion Product Images Small.* Kaggle. [Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
4. van der Maaten, L. & Hinton, G. (2008). *Visualizing Data using t-SNE.* JMLR.
5. Gradio Documentation. (2024). [https://www.gradio.app/docs](https://www.gradio.app/docs)

---

*End of Report*
