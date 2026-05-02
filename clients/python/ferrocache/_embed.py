"""Default embedding helper for the SDK middleware.

Uses sentence-transformers if available; raises a clear ImportError otherwise.
Users can always provide their own `embed_fn` to skip this dependency.
"""

from __future__ import annotations

from typing import Callable


def default_embed_fn(model_name: str = "all-MiniLM-L6-v2") -> Callable[[str], list[float]]:
    """Return a callable that embeds a single string into a unit-norm float vector.

    Loads the sentence-transformers model lazily on first construction.
    """
    embed, _model_id = get_default_embed(model_name)
    return embed


def get_default_embed(
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[Callable[[str], list[float]], str]:
    """Return `(embed_fn, model_id)` for `model_name`.

    `model_id` follows the convention `"<model_name>::<dimension>"` so that
    namespaces in the cache stay disjoint across embedding models even when
    they happen to produce the same dimension.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Default embedding requires sentence-transformers. "
            "Install it with `pip install sentence-transformers`, "
            "or pass your own embed_fn (and model_id) to the wrapper."
        ) from e

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    def embed(text: str) -> list[float]:
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    return embed, f"{model_name}::{dim}"
