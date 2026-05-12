"""
Embedder plug-in interface for engram.

Implement ``BaseEmbedder`` with any embedding backend (sentence-transformers,
OpenAI, Cohere, local ONNX models, etc.) and pass it to
``Engram.set_embedder()`` to activate semantic search.

When ``NullEmbedder`` is active (the default), ``semantic_score`` is always
0.0 and the keyword / temporal signals carry the full retrieval weight for
that slot.  Swap in a real embedder to get hybrid keyword + semantic ranking.

Example::

    class SentenceTransformerEmbedder(BaseEmbedder):
        def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name

        @property
        def model_name(self) -> str:
            return self._model_name

        def embed(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts, normalize_embeddings=True).tolist()
"""
from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Contract for text embedding backends.

    All vectors produced by a single instance must share the same dimension.
    The ``embed`` method accepts a *batch* of texts for efficiency.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Unique identifier for this embedding model.

        Used as a cache key in the embeddings table so that embeddings
        computed with different models are stored separately.
        """
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            Parallel list of float vectors, all with identical length.
        """
        ...


class NullEmbedder(BaseEmbedder):
    """
    Default embedder that returns empty vectors.

    Semantic scores remain 0.0 until a real embedder is registered via
    ``Engram.set_embedder()``.
    """

    @property
    def model_name(self) -> str:
        return "null"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


# ---------------------------------------------------------------------------
# Pure-Python utilities (no numpy required)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two float vectors.

    Returns 0.0 for empty, mismatched, or zero-norm inputs so callers
    never need to guard against NaN or ZeroDivisionError.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    if not (math.isfinite(norm_a) and math.isfinite(norm_b)):
        return 0.0
    result = dot / (norm_a * norm_b)
    return result if math.isfinite(result) else 0.0


def vec_to_blob(v: list[float]) -> bytes:
    """Pack a float vector to a compact binary blob (IEEE 754 float32)."""
    return struct.pack(f"{len(v)}f", *v)


def blob_to_vec(b: bytes) -> list[float]:
    """Unpack a binary blob (IEEE 754 float32) back to a float vector."""
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))
