"""ferrocache — minimal Python client (stdlib only).

Usage:
    from ferrocache import FerrocacheClient

    client = FerrocacheClient("http://localhost:3000")
    result = client.insert(
        embedding=[0.1, 0.2, 0.3, ...],
        response="The answer is 42",
        query_text="What is the meaning of life?",
    )
    print(result["id"])

    hit = client.query(embedding=[0.1, 0.2, 0.3, ...], threshold=0.92)
    if hit["hit"]:
        print(hit["response"])
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

DEFAULT_TIMEOUT = 10.0


class FerrocacheError(RuntimeError):
    """Raised on non-2xx responses or transport errors."""


class FerrocacheClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # If not passed explicitly, fall back to env. Empty string disables auth.
        if auth_token is None:
            auth_token = os.environ.get("FERROCACHE_AUTH_TOKEN") or None
        self.auth_token = auth_token

    def insert(
        self,
        embedding: list[float],
        response: str,
        query_text: str,
        model_id: str,
        ttl_seconds: int | None = None,
        cache_scope: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "embedding": embedding,
            "response": response,
            "query_text": query_text,
            "model_id": model_id,
        }
        if ttl_seconds is not None:
            payload["ttl_seconds"] = int(ttl_seconds)
        # M28: optional tenant/scope isolation. Sent only when explicitly
        # provided so older servers (pre-M28) just ignore the field.
        if cache_scope is not None:
            payload["cache_scope"] = cache_scope
        # M29: optional conversation id. Inserts with a conversation_id go
        # to the conversation-scoped namespace ONLY (not the global one).
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        return self._post("/insert", payload)

    def delete_entry(self, uuid: str) -> dict[str, Any]:
        """Delete a specific cache entry by UUID (M26)."""
        if not uuid:
            raise ValueError("uuid is required")
        return self._delete(f"/entry/{uuid}")

    def invalidate(
        self,
        embedding: list[float],
        threshold: float,
        model_id: str,
        cache_scope: str | None = None,
    ) -> dict[str, Any]:
        """Evict all entries with cosine similarity >= threshold (M26)."""
        body: dict[str, Any] = {
            "embedding": embedding,
            "threshold": threshold,
            "model_id": model_id,
        }
        if cache_scope is not None:
            body["cache_scope"] = cache_scope
        return self._post("/admin/invalidate", body)

    def query(
        self,
        embedding: list[float],
        threshold: float = 0.92,
        model_id: str | None = None,
        query_text: str | None = None,
        cache_scope: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise ValueError("model_id is required")
        body: dict[str, Any] = {
            "embedding": embedding,
            "threshold": threshold,
            "model_id": model_id,
        }
        # M27: optional exact-match pre-filter input. Sent only when set
        # so older servers (pre-M27) just ignore the field.
        if query_text is not None:
            body["query_text"] = query_text
        # M28: optional cache scope. Sent only when set so older servers
        # (pre-M28) just ignore the field.
        if cache_scope is not None:
            body["cache_scope"] = cache_scope
        # M29: optional conversation id. Triggers two-level lookup —
        # conversation namespace first, base namespace as fallback.
        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        return self._post("/query", body)

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def stats(self) -> dict[str, Any]:
        return self._get("/stats")

    def cluster_status(self) -> dict[str, Any]:
        return self._get("/cluster/status")

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        h: dict[str, str] = {}
        if content_type:
            h["Content-Type"] = content_type
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def _get(self, path: str) -> dict[str, Any]:
        req = request.Request(
            self.base_url + path,
            headers=self._headers(),
            method="GET",
        )
        return self._send(req)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=body,
            headers=self._headers("application/json"),
            method="POST",
        )
        return self._send(req)

    def _delete(self, path: str) -> dict[str, Any]:
        req = request.Request(
            self.base_url + path,
            headers=self._headers(),
            method="DELETE",
        )
        return self._send(req)

    def _send(self, req: request.Request) -> dict[str, Any]:
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise FerrocacheError(f"{e.code} {e.reason}: {body}") from e
        except error.URLError as e:
            raise FerrocacheError(f"transport error: {e.reason}") from e
