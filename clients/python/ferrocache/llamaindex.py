"""LlamaIndex LLM wrapper that consults ferrocache before delegating.

Usage:
    from llama_index.llms.openai import OpenAI
    from ferrocache.llamaindex import FerrocacheLLM

    llm = FerrocacheLLM(inner=OpenAI(model="gpt-4o-mini"))
    # use `llm` anywhere a LlamaIndex LLM is expected.

Subclasses `CustomLLM` so it satisfies LlamaIndex's `LLM` interface
(metadata, complete, stream_complete, chat). Cache misses delegate to the
inner LLM; hits return a synthetic `CompletionResponse` / `ChatResponse`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from ferrocache.client import FerrocacheClient, FerrocacheError

log = logging.getLogger("ferrocache.llamaindex")

try:
    from llama_index.core.llms import (
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        CustomLLM,
        LLMMetadata,
        MessageRole,
    )
    from pydantic import PrivateAttr

    _HAS_LLAMAINDEX = True
except ImportError:  # pragma: no cover
    _HAS_LLAMAINDEX = False
    CustomLLM = object  # type: ignore[assignment,misc]
    PrivateAttr = None  # type: ignore[assignment]


def _missing_llamaindex() -> ImportError:
    return ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install it with `pip install llama-index-core`."
    )


def _last_user_text(messages: Sequence[Any]) -> str | None:
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        if role and str(role).lower().endswith("user"):
            content = getattr(msg, "content", None)
            if callable(content):  # ChatMessage.content is a method in newer versions
                try:
                    content = content()
                except Exception:
                    content = None
            if isinstance(content, str) and content:
                return content
    return None


if _HAS_LLAMAINDEX:

    class FerrocacheLLM(CustomLLM):  # type: ignore[misc]
        """LlamaIndex LLM that checks ferrocache before delegating to `inner`."""

        _inner: Any = PrivateAttr()
        _client: FerrocacheClient = PrivateAttr()
        _embed_fn: Callable[[str], list[float]] = PrivateAttr()
        _model_id: str = PrivateAttr()
        _threshold: float = PrivateAttr()
        _fail_open: bool = PrivateAttr()

        def __init__(
            self,
            inner: Any,
            embed_fn: Callable[[str], list[float]] | None = None,
            cache_url: str | None = None,
            threshold: float | None = None,
            model_id: str | None = None,
            fail_open: bool = True,
            **pydantic_kwargs: Any,
        ) -> None:
            super().__init__(**pydantic_kwargs)

            from ferrocache.middleware import (
                _resolve_embed_and_model_id,
                _resolve_threshold,
                _resolve_url,
            )

            self._inner = inner
            self._client = FerrocacheClient(_resolve_url(cache_url))
            self._threshold = _resolve_threshold(threshold)
            self._fail_open = fail_open

            embed_fn, model_id = _resolve_embed_and_model_id(embed_fn, model_id)
            self._embed_fn = embed_fn
            self._model_id = model_id

        @property
        def metadata(self) -> LLMMetadata:
            inner_meta = getattr(self._inner, "metadata", None)
            if isinstance(inner_meta, LLMMetadata):
                return inner_meta
            return LLMMetadata()

        # ----- shared cache logic ---------------------------------------------------

        def _lookup(self, prompt: str) -> tuple[bool, str, float]:
            """Returns (hit, text, similarity). Suppresses errors on fail_open."""
            try:
                embedding = self._embed_fn(prompt)
            except Exception as e:
                log.warning("embed_fn failed: %s", e)
                return False, "", 0.0
            try:
                r = self._client.query(
                    embedding=embedding,
                    threshold=self._threshold,
                    model_id=self._model_id,
                )
            except FerrocacheError as e:
                if not self._fail_open:
                    raise
                log.warning("ferrocache unreachable (%s); treating as miss", e)
                return False, "", 0.0
            if not r.get("hit"):
                return False, "", 0.0
            return True, (r.get("response") or ""), float(r.get("similarity") or 0.0)

        def _store(self, prompt: str, text: str) -> None:
            try:
                embedding = self._embed_fn(prompt)
                self._client.insert(
                    embedding=embedding,
                    response=text,
                    query_text=prompt,
                    model_id=self._model_id,
                )
            except (FerrocacheError, Exception) as e:
                if not self._fail_open and isinstance(e, FerrocacheError):
                    raise
                log.warning("ferrocache insert failed: %s", e)

        # ----- LlamaIndex LLM interface --------------------------------------------

        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            hit, text, sim = self._lookup(prompt)
            if hit:
                log.info("cache hit (sim=%.3f)", sim)
                return CompletionResponse(text=text, additional_kwargs={"ferrocache_hit": True, "ferrocache_similarity": sim})
            log.info("cache miss; calling inner LLM")
            resp = self._inner.complete(prompt, formatted=formatted, **kwargs)
            self._store(prompt, getattr(resp, "text", "") or "")
            return resp

        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            # Streaming bypasses the cache (we'd have to buffer the full output to
            # cache it, which defeats the point of streaming). Delegate directly.
            return self._inner.stream_complete(prompt, formatted=formatted, **kwargs)

        def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
            prompt = _last_user_text(messages)
            if prompt is None:
                return self._inner.chat(messages, **kwargs)

            hit, text, sim = self._lookup(prompt)
            if hit:
                log.info("cache hit on chat (sim=%.3f)", sim)
                return ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
                    additional_kwargs={"ferrocache_hit": True, "ferrocache_similarity": sim},
                )
            log.info("cache miss on chat; calling inner LLM")
            resp = self._inner.chat(messages, **kwargs)
            cached_text = ""
            inner_msg = getattr(resp, "message", None)
            if inner_msg is not None:
                content = getattr(inner_msg, "content", None)
                if callable(content):
                    try:
                        content = content()
                    except Exception:
                        content = None
                if isinstance(content, str):
                    cached_text = content
            if cached_text:
                self._store(prompt, cached_text)
            return resp

        @classmethod
        def class_name(cls) -> str:
            return "FerrocacheLLM"

else:

    class FerrocacheLLM:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _missing_llamaindex()
