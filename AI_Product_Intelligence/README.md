# 🛍️ AI Product Intelligence System

> **Gen AI Bootcamp — Day 2 Assignment**
> A complete, production-ready AI application for fashion product intelligence,
> powered by **OpenAI CLIP**, **FAISS**, and **Gradio**.

---

## 📌 Project Overview

This system integrates three AI-powered capabilities into a single, modular application:

| Task | Description |
|------|-------------|
| **Smart Recommendation** | Recommends complementary products (e.g. shoes → socks, shirt → belt) |
| **Duplicate Detection** | Finds near-duplicate listings from different sellers and builds a clean catalog |
| **Reverse Search** | Lets users find products by describing them in natural language |

---

## ✨ Features

- 🎯 **Hybrid recommendation engine** — rule-based category mapping + CLIP embedding similarity
- 🔍 **Cosine-similarity duplicate detection** — Union-Find clustering + representative election
- 🔎 **Text-to-image semantic search** — CLIP text embeddings searched via FAISS
- 📊 **Rich visualisations** — t-SNE, category charts, recommendation cards, cluster views
- 💾 **Caching pipeline** — embeddings and FAISS index saved to disk (no recomputation)
- 🌐 **Modern Gradio UI** — 4 tabs, dark theme, responsive layout
- 📈 **Performance metrics** — latency tracking, Precision@K evaluation

---

## 🏗️ Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   Gradio UI (app.py)                │
│  Tab 1: Home │ Tab 2: Recommend │ Tab 3: Dedup │ Tab 4: Search │
└──────────┬──────────┬──────────────┬────────────────┘
           │          │              │
    ┌──────▼──┐  ┌────▼──────┐  ┌───▼────────────┐
    │Recommend│  │ Duplicate │  │ Reverse Search │
    │ Engine  │  │ Detector  │  │    Engine      │
    └──────┬──┘  └────┬──────┘  └───┬────────────┘
           │          │              │
           └──────────┼──────────────┘
                      │
              ┌───────▼────────┐
              │  FAISS Vector  │
              │   Database     │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  CLIP Embedder │
              │  (ViT-B/32)    │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Dataset       │
              │  (Fashion CSV) │
              └────────────────┘
```

---

## 📂 Project Structure

```
AI_Product_Intelligence/
│
├── app.py                  # Gradio web application (entry point)
├── config.py               # All configuration (paths, thresholds, mappings)
├── preprocessing.py        # Dataset loading and cleaning
├── embedding.py            # CLIP image/text embedding generation
├── vector_database.py      # FAISS index build, search, persist
├── recommendation.py       # Complementary product recommendation engine
├── duplicate_detection.py  # Near-duplicate detection + catalog creation
├── reverse_search.py       # Text-to-image FAISS search engine
├── visualization.py        # All chart generation (matplotlib)
├── utils.py                # Shared utilities (logger, timer, image loader)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   ├── styles.csv          # ← Place Kaggle CSV here
│   └── images/             # ← Place Kaggle images folder here
│
├── embeddings/             # Auto-generated CLIP embedding cache
├── faiss_index/            # Auto-generated FAISS index
├── outputs/                # unique_catalog.csv, duplicate_report.csv
├── screenshots/            # App screenshots
├── models/                 # (reserved for future fine-tuned models)
└── report/
    └── report.md           # Project report
```

---

## 🚀 Installation

### 1. Clone / Download

```bash
cd AI_Product_Intelligence
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The CLIP package installs directly from GitHub.
> Ensure you have `git` installed.

---

## 📦 Dataset Download

1. Go to [Kaggle — Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
2. Download and extract the dataset.
3. Place files as follows:

```
AI_Product_Intelligence/
└── data/
    ├── styles.csv          # metadata
    └── images/             # folder containing XXXXX.jpg files
        ├── 15970.jpg
        ├── 39386.jpg
        └── ...
```

---

## ▶️ Running the Application

```bash
python app.py
```

The app will:
1. Load and clean the dataset
2. Generate CLIP embeddings *(skipped if cache exists)*
3. Build FAISS index *(skipped if cache exists)*
4. Launch Gradio at **http://localhost:7860**

> **First run** takes ~5–10 minutes to generate embeddings.
> Subsequent runs load from cache instantly.

---

## 🗝️ Configuration

Edit **`config.py`** to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_PRODUCTS` | `5000` | Cap on dataset size (set `None` for all) |
| `CLIP_MODEL_NAME` | `"ViT-B/32"` | CLIP variant |
| `DUPLICATE_SIMILARITY_THRESHOLD` | `0.92` | Cosine threshold for duplicates |
| `TOP_K_RECOMMENDATIONS` | `5` | Recommendations per product |
| `TOP_K_SEARCH_RESULTS` | `5` | Search results per query |
| `COMPLEMENTARY_MAPPING` | *dict* | Category compatibility rules |

---

## 🧠 How Each Task Works

### Task 1 — Smart Recommendation

1. User selects a product (e.g. "Running Shoes").
2. The engine looks up `COMPLEMENTARY_MAPPING` for "sports shoes" → `["socks", "shorts", "cap", ...]`.
3. Candidate products in those categories are scored by **cosine similarity** of CLIP embeddings.
4. Top-K are returned with an explanation of *why* each product was recommended.

### Task 2 — Duplicate Detection

1. All CLIP embeddings are loaded.
2. A **block-wise cosine similarity matrix** is computed.
3. Pairs above the threshold are joined via **Union-Find clustering**.
4. Within each cluster, the product with the **largest image file** (quality proxy) is elected as representative.
5. A clean `unique_catalog.csv` is exported.

### Task 3 — Reverse Search

1. User types a text query (e.g. "blue casual shirt").
2. **CLIP text encoder** converts it to a 512-D embedding.
3. **FAISS** performs a nearest-neighbour search over product image embeddings.
4. Top-K results are returned with images, metadata, and similarity scores.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Cosine Similarity | Quality of embedding matches |
| Precision@K | % of top-K results that are relevant |
| CLIP Encode Latency | Time to encode a text query |
| FAISS Search Latency | Time to perform nearest-neighbour search |
| Catalog Reduction % | Duplicates removed vs total |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding Model | OpenAI CLIP ViT-B/32 |
| Vector Database | FAISS (IndexFlatIP) |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Image Processing | Pillow, OpenCV |
| Visualisation | Matplotlib |
| Web UI | Gradio 4.x |
| Dimensionality Reduction | scikit-learn (PCA + t-SNE) |

---

## 🔮 Future Work

- [ ] Fine-tune CLIP on fashion-specific data
- [ ] GPU acceleration for real-time embedding
- [ ] User feedback loop (implicit rating)
- [ ] Multi-modal search (image + text)
- [ ] Seller dashboard with duplicate alerts
- [ ] REST API wrapper (FastAPI)

---

## 👤 Author

**Gen AI Bootcamp — Day 2 Assignment**
*AI Product Intelligence System*

---

## 📄 License

For educational use only. Dataset from Kaggle (Param Aggarwal).
