"""Stage 3: EmbeddingGemma 2, text-only, as the model card prescribes.

* bf16 where the device supports it, else f32 - never f16, whose range EmbeddingGemma 2 overflows
  into NaNs or silently degraded vectors;
* documents are formatted ``title: {title} | text: {text}`` by hand (``prompt_name="Document"``
  would insert ``title: none``); queries use ``prompt_name="SearchQuery"``;
* Matryoshka truncation through ``truncate_dim``, then re-normalized in f32, since bf16 inference
  leaves norms a few parts per thousand off 1.
"""

from __future__ import annotations

import os

import numpy as np

MODEL = "google/embeddinggemma-2"


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    def __init__(self, device: str = "auto", dtype: str = "auto", max_seq_length: int = 512, model: str = MODEL):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = pick_device(device)
        if dtype == "auto":
            bf16 = (self.device == "cuda" and torch.cuda.is_bf16_supported()) or self.device == "mps"
            dtype = "bfloat16" if bf16 else "float32"
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("EmbeddingGemma 2 must run in bfloat16 or float32, never float16")
        self.dtype = dtype
        self.model = SentenceTransformer(
            model,
            device=self.device,
            config_kwargs={"vision_config": None, "audio_config": None},  # the 270M text model only
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
        )
        self.model.max_seq_length = max_seq_length
        self.name = model

    @staticmethod
    def document(title: str, text: str) -> str:
        return f"title: {title} | text: {text}"

    def _finish(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms == 0).any():
            raise FloatingPointError("the encoder produced NaN, Inf or zero vectors")
        return vectors / norms

    def encode_documents(self, cards: list[tuple[str, str]], dims: int, batch_size: int = 64) -> np.ndarray:
        texts = [self.document(title, text) for title, text in cards]
        vectors = self.model.encode(
            texts, batch_size=batch_size, truncate_dim=dims, normalize_embeddings=True, convert_to_numpy=True
        )
        return self._finish(vectors)

    def encode_query(self, query: str, dims: int) -> np.ndarray:
        vector = self.model.encode(
            [query], prompt_name="SearchQuery", truncate_dim=dims, normalize_embeddings=True, convert_to_numpy=True
        )
        return self._finish(vector)[0]


def truncate(vector: np.ndarray, dims: int) -> np.ndarray:
    """Matryoshka slice of a stored vector to a smaller index's dimension, re-normalized - how a
    768d concept vector queries a 512d item index."""
    sliced = np.asarray(vector[:dims], dtype=np.float32)
    return sliced / np.linalg.norm(sliced)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
