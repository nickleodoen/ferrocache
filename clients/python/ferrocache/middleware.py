"""Drop-in semantic caching middleware for the OpenAI and Anthropic SDKs.

Usage:
    from openai import OpenAI
    from ferrocache.middleware import wrap_openai

    client = wrap_openai(OpenAI())  # all chat.completions.create calls now check cache

The wrappers proxy attribute access to the real SDK client; only the chat
completion / message creation method is intercepted. A cache outage with
fail_open=True falls through to the real API — caching never breaks the app.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable
from uuid import uuid4

from ferrocache.client import FerrocacheClient, FerrocacheError

log = logging.getLogger("ferrocache.middleware")

DEFAULT_URL = "http://localhost:3000"
DEFAULT_THRESHOLD = 0.92


def _resolve_url(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return os.environ.get("FERROCACHE_URL", DEFAULT_URL)


def _resolve_threshold(explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    raw = os.environ.get("FERROCACHE_THRESHOLD")
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        log.warning("invalid FERROCACHE_THRESHOLD=%r; using %s", raw, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD


def _last_user_text(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            # OpenAI/Anthropic also accept content as a list of parts
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                joined = "\n".join(p for p in parts if p)
                if joined:
                    return joined
            return None
    return None


def _embed_safely(embed_fn: Callable[[str], list[float]], text: str) -> list[float] | None:
    try:
        return embed_fn(text)
    except Exception as e:
        log.warning("embed_fn failed: %s — falling through to API", e)
        return None


# ---------------------------------------------------------------------------
# Synthetic response objects (structurally compatible with the SDKs)
# ---------------------------------------------------------------------------


class _SimpleNamespace:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _build_openai_completion(content: str, model: str, similarity: float) -> Any:
    msg = _SimpleNamespace(role="assistant", content=content)
    choice = _SimpleNamespace(index=0, message=msg, finish_reason="stop")
    usage = _SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return _SimpleNamespace(
        id=f"ferrocache-{uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
        _ferrocache_hit=True,
        _ferrocache_similarity=similarity,
    )


def _build_anthropic_message(content: str, model: str, similarity: float) -> Any:
    block = _SimpleNamespace(type="text", text=content)
    usage = _SimpleNamespace(input_tokens=0, output_tokens=0)
    return _SimpleNamespace(
        id=f"ferrocache-{uuid4().hex[:8]}",
        type="message",
        role="assistant",
        model=model,
        content=[block],
        stop_reason="end_turn",
        usage=usage,
        _ferrocache_hit=True,
        _ferrocache_similarity=similarity,
    )


def _set_attr_safely(obj: Any, name: str, value: Any) -> None:
    """Try to set an attr on an SDK response. SDK objects are pydantic models
    that may reject unknown attrs in __setattr__; fall back to __dict__."""
    try:
        setattr(obj, name, value)
    except Exception:
        try:
            obj.__dict__[name] = value
        except Exception:
            pass  # best-effort; caller can still use the response


# ---------------------------------------------------------------------------
# Shared interception logic
# ---------------------------------------------------------------------------


class _ProviderHooks:
    """Provider-specific extraction + response building."""

    def __init__(
        self,
        extract_response_text: Callable[[Any], str],
        build_cached_response: Callable[[str, str, float], Any],
    ) -> None:
        self.extract = extract_response_text
        self.build = build_cached_response


def _intercept(
    real_call: Callable[..., Any],
    provider: _ProviderHooks,
    cache: FerrocacheClient,
    embed_fn: Callable[[str], list[float]],
    model_id: str,
    threshold: float,
    fail_open: bool,
    kwargs: dict[str, Any],
) -> Any:
    messages = kwargs.get("messages") or []
    model = kwargs.get("model", "unknown")
    prompt = _last_user_text(messages) if isinstance(messages, list) else None

    if prompt is None:
        log.debug("no user prompt found in messages; bypassing cache")
        return real_call(**kwargs)

    embedding = _embed_safely(embed_fn, prompt)
    if embedding is None:
        return real_call(**kwargs)

    log.debug(
        "ferrocache lookup: dim=%d, threshold=%s, model_id=%s",
        len(embedding),
        threshold,
        model_id,
    )

    try:
        result = cache.query(embedding=embedding, threshold=threshold, model_id=model_id)
    except FerrocacheError as e:
        if not fail_open:
            raise
        log.warning("ferrocache unreachable (%s); calling API directly", e)
        resp = real_call(**kwargs)
        _set_attr_safely(resp, "_ferrocache_hit", None)
        return resp

    if result.get("hit"):
        sim = float(result.get("similarity") or 0.0)
        content = result.get("response") or ""
        log.info("cache hit: similarity=%.3f", sim)
        return provider.build(content, model, sim)

    log.info("cache miss: calling API")
    resp = real_call(**kwargs)
    _set_attr_safely(resp, "_ferrocache_hit", False)

    try:
        text = provider.extract(resp)
    except Exception as e:
        log.warning("could not extract response text for caching: %s", e)
        return resp

    try:
        cache.insert(
            embedding=embedding,
            response=text,
            query_text=prompt,
            model_id=model_id,
        )
    except FerrocacheError as e:
        if not fail_open:
            raise
        log.warning("ferrocache insert failed: %s", e)

    return resp


def _resolve_embed_and_model_id(
    embed_fn: Callable[[str], list[float]] | None,
    model_id: str | None,
) -> tuple[Callable[[str], list[float]], str]:
    """Embed-function + model_id resolution shared by all integrations.

    - both None  → load the default sentence-transformers model + auto-derive model_id
    - embed_fn given without model_id → ValueError (we won't guess)
    - both given → use as-is
    """
    if embed_fn is None and model_id is None:
        from ferrocache._embed import get_default_embed

        return get_default_embed()
    if embed_fn is not None and model_id is None:
        raise ValueError(
            "When providing a custom embed_fn, you must also provide model_id "
            "(e.g. model_id='my-model::768'). Cross-model cache hits would otherwise "
            "be silently incorrect."
        )
    if embed_fn is None and model_id is not None:
        from ferrocache._embed import get_default_embed

        embed_fn, _ = get_default_embed()
    return embed_fn, model_id  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# OpenAI wrapper
# ---------------------------------------------------------------------------


def _extract_openai_text(resp: Any) -> str:
    return resp.choices[0].message.content or ""


_openai_hooks = _ProviderHooks(_extract_openai_text, _build_openai_completion)


class _WrappedOpenAICompletions:
    def __init__(
        self,
        real: Any,
        cache: FerrocacheClient,
        embed_fn: Callable[[str], list[float]],
        model_id: str,
        threshold: float,
        fail_open: bool,
    ) -> None:
        self._real = real
        self._cache = cache
        self._embed_fn = embed_fn
        self._model_id = model_id
        self._threshold = threshold
        self._fail_open = fail_open

    def create(self, **kwargs: Any) -> Any:
        return _intercept(
            self._real.create,
            _openai_hooks,
            self._cache,
            self._embed_fn,
            self._model_id,
            self._threshold,
            self._fail_open,
            kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _WrappedOpenAIChat:
    def __init__(self, real: Any, **kwargs: Any) -> None:
        self._real = real
        self.completions = _WrappedOpenAICompletions(real.completions, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class WrappedOpenAIClient:
    def __init__(
        self,
        real: Any,
        cache: FerrocacheClient,
        embed_fn: Callable[[str], list[float]],
        model_id: str,
        threshold: float,
        fail_open: bool,
    ) -> None:
        self._real = real
        self.chat = _WrappedOpenAIChat(
            real.chat,
            cache=cache,
            embed_fn=embed_fn,
            model_id=model_id,
            threshold=threshold,
            fail_open=fail_open,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_openai(
    client: Any,
    cache_url: str | None = None,
    threshold: float | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    model_id: str | None = None,
    fail_open: bool = True,
) -> WrappedOpenAIClient:
    """Wrap an `openai.OpenAI()` client so chat.completions.create checks ferrocache first."""
    cache = FerrocacheClient(_resolve_url(cache_url))
    embed_fn, model_id = _resolve_embed_and_model_id(embed_fn, model_id)
    return WrappedOpenAIClient(
        client,
        cache=cache,
        embed_fn=embed_fn,
        model_id=model_id,
        threshold=_resolve_threshold(threshold),
        fail_open=fail_open,
    )


# ---------------------------------------------------------------------------
# Anthropic wrapper
# ---------------------------------------------------------------------------


def _extract_anthropic_text(resp: Any) -> str:
    parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts)


_anthropic_hooks = _ProviderHooks(_extract_anthropic_text, _build_anthropic_message)


class _WrappedAnthropicMessages:
    def __init__(
        self,
        real: Any,
        cache: FerrocacheClient,
        embed_fn: Callable[[str], list[float]],
        model_id: str,
        threshold: float,
        fail_open: bool,
    ) -> None:
        self._real = real
        self._cache = cache
        self._embed_fn = embed_fn
        self._model_id = model_id
        self._threshold = threshold
        self._fail_open = fail_open

    def create(self, **kwargs: Any) -> Any:
        return _intercept(
            self._real.create,
            _anthropic_hooks,
            self._cache,
            self._embed_fn,
            self._model_id,
            self._threshold,
            self._fail_open,
            kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class WrappedAnthropicClient:
    def __init__(
        self,
        real: Any,
        cache: FerrocacheClient,
        embed_fn: Callable[[str], list[float]],
        model_id: str,
        threshold: float,
        fail_open: bool,
    ) -> None:
        self._real = real
        self.messages = _WrappedAnthropicMessages(
            real.messages,
            cache=cache,
            embed_fn=embed_fn,
            model_id=model_id,
            threshold=threshold,
            fail_open=fail_open,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_anthropic(
    client: Any,
    cache_url: str | None = None,
    threshold: float | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    model_id: str | None = None,
    fail_open: bool = True,
) -> WrappedAnthropicClient:
    """Wrap an `anthropic.Anthropic()` client so messages.create checks ferrocache first."""
    cache = FerrocacheClient(_resolve_url(cache_url))
    embed_fn, model_id = _resolve_embed_and_model_id(embed_fn, model_id)
    return WrappedAnthropicClient(
        client,
        cache=cache,
        embed_fn=embed_fn,
        model_id=model_id,
        threshold=_resolve_threshold(threshold),
        fail_open=fail_open,
    )


__all__ = ["wrap_openai", "wrap_anthropic"]
