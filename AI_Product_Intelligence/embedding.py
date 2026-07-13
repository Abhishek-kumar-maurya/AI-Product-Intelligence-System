"""
embedding.py
============
CLIP-based image and text embedding generation.

Responsibilities
----------------
* Load the OpenAI CLIP model (ViT-B/32) onto the best available device.
* Generate L2-normalised image embeddings for every product in the dataset.
* Generate L2-normalised text embeddings for reverse-search queries.
* Cache embeddings to disk (.npy) to avoid recomputation on subsequent runs.
* Provide a batch-processing pipeline with progress logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import clip          # pip install openai-clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

import config
import utils

logger = utils.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLIPEmbedder
# ─────────────────────────────────────────────────────────────────────────────

class CLIPEmbedder:
    """
    Wrapper around OpenAI CLIP for image and text embedding.

    The model is loaded once on instantiation and reused for all
    subsequent calls, making it suitable as a module-level singleton.

    Attributes:
        model:     The CLIP model.
        preprocess: CLIP's canonical image preprocessing transform.
        device:    Torch device (cuda / cpu).
        embed_dim: Embedding dimensionality (512 for ViT-B/32).
    """

    def __init__(
        self,
        model_name: str = config.CLIP_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: CLIP variant string, e.g. ``"ViT-B/32"``.
            device:     ``"cuda"`` / ``"cpu"``.  Auto-detected if ``None``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: str = device
        logger.info(f"Loading CLIP model '{model_name}' on {device} …")

        with utils.timer("CLIP model load", logger):
            self.model, self.preprocess = clip.load(model_name, device=device)

        self.model.eval()
        # Determine embedding dimensionality via dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            self.embed_dim: int = self.model.encode_image(dummy).shape[-1]

        logger.info(f"CLIP ready. Embedding dim = {self.embed_dim}.")

    # ── Image embedding ───────────────────────────────────────────────────────

    def embed_images(
        self,
        image_paths: list[str | Path],
        batch_size: int = config.CLIP_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Generate L2-normalised image embeddings for a list of image paths.

        Invalid / missing images are replaced with zero vectors.

        Args:
            image_paths: Ordered list of image file paths.
            batch_size:  Number of images processed per GPU/CPU batch.

        Returns:
            Float32 numpy array of shape ``(N, embed_dim)``, L2-normalised.
        """
        n = len(image_paths)
        embeddings = np.zeros((n, self.embed_dim), dtype=np.float32)

        logger.info(f"Embedding {n} images in batches of {batch_size} …")

        with utils.timer("Image embedding generation", logger):
            for batch_start in range(0, n, batch_size):
                batch_paths = image_paths[batch_start: batch_start + batch_size]
                batch_tensors: list[Tensor] = []
                valid_indices: list[int] = []

                for local_idx, path in enumerate(batch_paths):
                    img = utils.load_image_safe(path)
                    if img is None:
                        logger.debug(f"Skipping unreadable image: {path}")
                        continue
                    try:
                        tensor = self.preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)
                        batch_tensors.append(tensor)
                        valid_indices.append(local_idx)
                    except Exception as exc:
                        logger.debug(f"Preprocess error for {path}: {exc}")

                if not batch_tensors:
                    continue

                batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)  # (B, 3, 224, 224)

                with torch.no_grad():
                    batch_emb = self.model.encode_image(batch_tensor)   # (B, D)
                    batch_emb = batch_emb.cpu().float().numpy()

                # Write into the output array at the right global indices
                for local_i, global_i in enumerate(valid_indices):
                    global_idx = batch_start + global_i
                    embeddings[global_idx] = batch_emb[local_i]

                processed = min(batch_start + batch_size, n)
                if processed % 200 == 0 or processed == n:
                    logger.info(f"  … {processed}/{n} images embedded.")

        # L2-normalise (skip zero vectors)
        utils.l2_normalize(embeddings)
        return embeddings

    def embed_single_image(self, path: str | Path) -> Optional[np.ndarray]:
        """
        Generate a normalised embedding for one image.

        Args:
            path: Path to image file.

        Returns:
            Shape ``(embed_dim,)`` float32 array, or ``None`` on error.
        """
        img = utils.load_image_safe(path)
        if img is None:
            return None
        try:
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(tensor).cpu().float().numpy()[0]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            return emb
        except Exception as exc:
            logger.warning(f"embed_single_image failed for {path}: {exc}")
            return None

    # ── Text embedding ────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate a normalised embedding for a text query.

        Args:
            text: Free-form search query string.

        Returns:
            Shape ``(embed_dim,)`` float32 array, L2-normalised.

        Raises:
            ValueError: If *text* is empty.
        """
        if not text or not text.strip():
            raise ValueError("Query text must not be empty.")

        tokens = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens).cpu().float().numpy()[0]

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def embed_texts_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate normalised embeddings for multiple text strings.

        Args:
            texts: List of query strings.

        Returns:
            Shape ``(N, embed_dim)`` float32 array, L2-normalised.
        """
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens).cpu().float().numpy()
        utils.l2_normalize(emb)
        return emb


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Manages saving and loading of precomputed embeddings to / from disk.

    File layout::

        embeddings/
          image_embeddings.npy   # float32 (N, D)
          product_ids.npy        # int64   (N,)
          metadata.pkl           # pickled DataFrame
    """

    @staticmethod
    def exists() -> bool:
        """Return True if all three cache files exist."""
        return (
            config.IMAGE_EMBEDDINGS_FILE.exists()
            and config.PRODUCT_IDS_FILE.exists()
            and config.METADATA_CACHE_FILE.exists()
        )

    @staticmethod
    def save(
        embeddings: np.ndarray,
        product_ids: np.ndarray,
        metadata: pd.DataFrame,
    ) -> None:
        """
        Persist embeddings, IDs, and metadata to disk.

        Args:
            embeddings:  Shape ``(N, D)`` float32 array.
            product_ids: Shape ``(N,)`` int64 array.
            metadata:    Full cleaned DataFrame.
        """
        config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.IMAGE_EMBEDDINGS_FILE, embeddings)
        np.save(config.PRODUCT_IDS_FILE, product_ids)
        utils.save_pickle(metadata, config.METADATA_CACHE_FILE)
        logger.info(
            f"Saved embeddings: {embeddings.shape} | "
            f"product_ids: {product_ids.shape} | "
            f"metadata: {len(metadata)} rows."
        )

    @staticmethod
    def load() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load cached embeddings from disk.

        Returns:
            Tuple ``(embeddings, product_ids, metadata_df)``.

        Raises:
            FileNotFoundError: If cache is missing.
        """
        if not EmbeddingCache.exists():
            raise FileNotFoundError(
                "Embedding cache not found. Run `python embedding.py` first."
            )
        embeddings = np.load(config.IMAGE_EMBEDDINGS_FILE)
        product_ids = np.load(config.PRODUCT_IDS_FILE)
        metadata = utils.load_pickle(config.METADATA_CACHE_FILE)
        logger.info(
            f"Loaded embeddings from cache: "
            f"shape={embeddings.shape}, products={len(metadata)}."
        )
        return embeddings, product_ids, metadata


# ─────────────────────────────────────────────────────────────────────────────
# High-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_cache_embeddings(
    df: pd.DataFrame,
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate CLIP image embeddings for every product in *df* and cache them.

    If a valid cache exists and *force_recompute* is False, the cached
    version is loaded and returned immediately.

    Args:
        df:              Cleaned metadata DataFrame with ``image_path`` column.
        force_recompute: If True, always recompute even if cache exists.

    Returns:
        Tuple ``(embeddings, product_ids, df)`` where

        * ``embeddings`` – float32 ``(N, D)`` array
        * ``product_ids`` – int64 ``(N,)`` array
        * ``df``          – metadata DataFrame aligned row-for-row
    """
    if not force_recompute and EmbeddingCache.exists():
        logger.info("Embedding cache found — loading from disk (skip recompute).")
        return EmbeddingCache.load()

    logger.info("No cache found — computing embeddings from scratch.")
    embedder = get_embedder()

    image_paths: list[str] = df["image_path"].tolist()
    product_ids: np.ndarray = df["id"].to_numpy(dtype=np.int64)

    embeddings = embedder.embed_images(image_paths)
    EmbeddingCache.save(embeddings, product_ids, df)
    return embeddings, product_ids, df


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_embedder_instance: Optional[CLIPEmbedder] = None


def get_embedder() -> CLIPEmbedder:
    """
    Return the module-level :class:`CLIPEmbedder` singleton.

    Instantiates on first call; subsequent calls return the cached instance.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = CLIPEmbedder()
    return _embedder_instance


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import DatasetLoader

    loader = DatasetLoader(max_products=200).load()
    embeddings, ids, meta = generate_and_cache_embeddings(loader.df, force_recompute=True)

    print(f"\nEmbedding shape : {embeddings.shape}")
    print(f"Product IDs     : {ids[:5]} …")
    print(f"Norm of emb[0]  : {np.linalg.norm(embeddings[0]):.4f}  (should be ~1.0)")

    # Text embedding smoke-test
    embedder = get_embedder()
    txt_emb = embedder.embed_text("blue casual shirt")
    print(f"Text emb shape  : {txt_emb.shape}")
    print(f"Text emb norm   : {np.linalg.norm(txt_emb):.4f}")
