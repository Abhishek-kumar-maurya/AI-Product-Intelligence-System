"""
utils.py
========
Shared utilities used across all modules:
  - Centralised logging factory
  - Timer context manager
  - Safe image loading with error handling
  - File / directory helpers
  - Tensor normalisation helper
"""

from __future__ import annotations

import logging
import time
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Any, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

import config


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a named logger with the project's standard format.

    Args:
        name: Logger name, usually ``__name__`` from the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# TIMER
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None) -> Generator:
    """
    Context manager that logs the wall-clock time of a block.

    Example::

        with timer("FAISS index build"):
            build_index(...)

    Args:
        label:  Human-readable description of the timed block.
        logger: Logger to write the timing message.  Uses root logger if None.
    """
    _log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _log.info(f"[{label}] completed in {elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_image_safe(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None,
) -> Optional[Image.Image]:
    """
    Load a PIL image, returning *None* on any error rather than raising.

    Handles:
      - Missing files
      - Corrupted / truncated images
      - Non-image files mis-labelled with an image extension

    Args:
        path: Filesystem path to the image file.
        size: Optional ``(width, height)`` to resize to after loading.

    Returns:
        :class:`PIL.Image.Image` in RGB mode, or ``None`` on failure.
    """
    try:
        img = Image.open(path)
        img.verify()                   # catch truncated files early
        img = Image.open(path)        # re-open after verify (verify closes)
        img = img.convert("RGB")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
        return img
    except (FileNotFoundError, OSError, UnidentifiedImageError, Exception):
        return None


def load_image_strict(path: Union[str, Path], size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Load a PIL image and raise a descriptive error on failure.

    Args:
        path: Filesystem path to the image file.
        size: Optional ``(width, height)`` to resize to.

    Returns:
        :class:`PIL.Image.Image` in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = load_image_safe(p, size)
    if img is None:
        raise ValueError(f"Cannot decode image: {p}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY / EMBEDDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalise a batch of vectors in-place (modifies array).

    After normalisation every row has unit norm, making dot product
    equivalent to cosine similarity.

    Args:
        vectors: Shape ``(N, D)`` float array.

    Returns:
        Normalised array (same object, modified in-place).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    vectors /= norms
    return vectors


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two L2-normalised matrices.

    Args:
        a: Shape ``(M, D)``, L2-normalised.
        b: Shape ``(N, D)``, L2-normalised.

    Returns:
        Shape ``(M, N)`` similarity matrix.
    """
    return a @ b.T


# ─────────────────────────────────────────────────────────────────────────────
# PICKLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_pickle(obj: Any, path: Path) -> None:
    """Serialise *obj* to *path* using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Deserialise and return an object from a pickle file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_image_path(product_id: int | str) -> Optional[Path]:
    """
    Resolve the filesystem path for a product's image.

    Searches ``config.IMAGES_DIR`` for a file whose stem matches
    *product_id*, trying all supported extensions.

    Args:
        product_id: Numeric or string product identifier.

    Returns:
        :class:`pathlib.Path` if found, else ``None``.
    """
    stem = str(product_id)
    for ext in config.IMAGE_EXTENSIONS:
        candidate = config.IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def ensure_dir(path: Path) -> Path:
    """Create *path* and all parents if they do not exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def truncate_str(text: str, max_len: int = 40) -> str:
    """Truncate *text* to *max_len* characters, appending '…' if needed."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def format_score(score: float) -> str:
    """Format a similarity score as a percentage string."""
    return f"{score * 100:.1f}%"
