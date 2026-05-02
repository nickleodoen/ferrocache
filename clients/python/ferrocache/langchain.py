"""LangChain `BaseCache` backend for ferrocache.

Usage:
    from langchain.globals import set_llm_cache
    from ferrocache.langchain import FerrocacheCache

    set_llm_cache(FerrocacheCache(embed_fn=my_embed))

Every LLM call in the user's chain now checks ferrocache first.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from ferrocache.client import FerrocacheClient, FerrocacheError

log = logging.getLogger("ferrocache.langchain")

try:
    from langchain_core.caches import BaseCache
    from langchain_core.outputs import Generation

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    _HAS_LANGCHAIN = False
    BaseCache = object  # type: ignore[assignment,misc]
    Generation = None  # type: ignore[assignment]


def _missing_langchain() -> ImportError:
    return ImportError(
        "LangChain integration requires langchain-core. "
        "Install it with `pip install langchain-core`."
    )


class FerrocacheCache(BaseCache):  # type: ignore[misc]
    """ferrocache-backed LangChain cache.

    On `lookup`, embeds the prompt and queries ferrocache; returns a single
    `Generation` on hit, `None` on miss. On `update`, embeds the prompt and
    inserts the cached completion. `clear` is a no-op (ferrocache is
    append-only — there is no bulk-delete primitive).

    Errors are swallowed when `fail_open=True` (the default) so a cache
    outage degrades gracefully into "cache always misses".
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        cache_url: str | None = None,
        threshold: float | None = None,
        model_id: str | None = None,
        fail_open: bool = True,
        auth_token: str | None = None,
    ) -> None:
        if not _HAS_LANGCHAIN:
            raise _missing_langchain()

        from ferrocache.middleware import (
            _resolve_embed_and_model_id,
            _resolve_threshold,
            _resolve_url,
        )

        self._client = FerrocacheClient(_resolve_url(cache_url), auth_token=auth_token)
        self._threshold = _resolve_threshold(threshold)
        self._fail_open = fail_open

        embed_fn, model_id = _resolve_embed_and_model_id(embed_fn, model_id)
        self._embed_fn = embed_fn
        self._model_id = model_id

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        try:
            embedding = self._embed_fn(prompt)
        except Exception as e:
            log.warning("embed_fn failed during lookup: %s", e)
            return None

        try:
            result = self._client.query(
                embedding=embedding,
                threshold=self._threshold,
                model_id=self._model_id,
            )
        except FerrocacheError as e:
            if not self._fail_open:
                raise
            log.warning("ferrocache lookup failed (%s); treating as miss", e)
            return None

        if not result.get("hit"):
            return None

        text = result.get("response") or ""
        log.info(
            "cache hit (sim=%.3f): %s",
            float(result.get("similarity") or 0.0),
            llm_string[:40],
        )
        return [Generation(text=text)]

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: Sequence[Generation],
    ) -> None:
        if not return_val:
            return
        text = getattr(return_val[0], "text", None)
        if not text:
            return

        try:
            embedding = self._embed_fn(prompt)
        except Exception as e:
            log.warning("embed_fn failed during update: %s", e)
            return

        try:
            self._client.insert(
                embedding=embedding,
                response=text,
                query_text=prompt,
                model_id=self._model_id,
            )
        except FerrocacheError as e:
            if not self._fail_open:
                raise
            log.warning("ferrocache insert failed: %s", e)

    def clear(self, **kwargs: Any) -> None:
        log.warning(
            "FerrocacheCache.clear() is a no-op — ferrocache is append-only. "
            "Restart the server with a fresh WAL to drop entries."
        )
