"""ferrocache MCP server — exposes semantic caching as tools for AI agents.

Run via:
    python -m ferrocache.mcp_server

Speaks JSON-RPC over stdio (the standard transport for Claude Desktop and
Claude Code). Three tools are exposed: `semantic_cache_lookup`,
`semantic_cache_store`, and `cache_status`. The MCP layer handles embedding
internally — agents send text, not vectors.

Environment:
    FERROCACHE_URL          (default: http://localhost:3000)
    FERROCACHE_THRESHOLD    (default: 0.92)
    FERROCACHE_EMBED_MODEL  (default: all-MiniLM-L6-v2)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable

from ferrocache.client import FerrocacheClient, FerrocacheError

log = logging.getLogger("ferrocache.mcp_server")

DEFAULT_URL = "http://localhost:3000"
DEFAULT_THRESHOLD = 0.92
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Pure tool dispatch — decoupled from the MCP transport so it's easily testable.
# ---------------------------------------------------------------------------


class FerrocacheTools:
    """Async tool dispatch backed by a `FerrocacheClient` and an embed function.

    Every tool method returns a JSON-serializable dict. Errors are converted
    into `{"error": "..."}` payloads so the MCP server keeps running across
    transient failures.
    """

    def __init__(
        self,
        client: FerrocacheClient,
        embed_fn: Callable[[str], list[float]],
        model_id: str = "all-MiniLM-L6-v2::384",
        default_threshold: float = DEFAULT_THRESHOLD,
        auth_token: str | None = None,
    ) -> None:
        # auth_token is accepted for symmetry; the client carries the actual
        # auth header. If both are passed, the client's takes precedence.
        if auth_token is not None and not client.auth_token:
            client.auth_token = auth_token
        self.client = client
        self.embed_fn = embed_fn
        self.model_id = model_id
        self.default_threshold = default_threshold

    async def lookup(
        self,
        query_text: str,
        threshold: float | None = None,
        cache_scope: str | None = None,
    ) -> dict[str, Any]:
        if not query_text:
            return {"error": "query_text is required"}
        t = threshold if threshold is not None else self.default_threshold
        try:
            embedding = await asyncio.to_thread(self.embed_fn, query_text)
        except Exception as e:
            log.warning("embed_fn failed: %s", e)
            return {"error": f"embedding failed: {e}"}
        try:
            result = await asyncio.to_thread(
                self.client.query,
                embedding=embedding,
                threshold=t,
                model_id=self.model_id,
                query_text=query_text,  # M27 exact-match pre-filter
                cache_scope=cache_scope,
            )
        except FerrocacheError as e:
            log.warning("ferrocache lookup failed: %s", e)
            return {"error": f"ferrocache unreachable: {e}"}
        return result

    async def store(
        self,
        query_text: str,
        response: str,
        cache_scope: str | None = None,
    ) -> dict[str, Any]:
        if not query_text:
            return {"error": "query_text is required"}
        if not response:
            return {"error": "response is required"}
        try:
            embedding = await asyncio.to_thread(self.embed_fn, query_text)
        except Exception as e:
            log.warning("embed_fn failed: %s", e)
            return {"error": f"embedding failed: {e}"}
        try:
            result = await asyncio.to_thread(
                self.client.insert,
                embedding=embedding,
                response=response,
                query_text=query_text,
                model_id=self.model_id,
                cache_scope=cache_scope,
            )
        except FerrocacheError as e:
            log.warning("ferrocache insert failed: %s", e)
            return {"error": f"ferrocache unreachable: {e}"}
        return result

    async def status(self) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self.client.health)
        except FerrocacheError as e:
            log.warning("ferrocache health failed: %s", e)
            return {"error": f"ferrocache unreachable: {e}"}


# ---------------------------------------------------------------------------
# Tool catalog — used by both the MCP server and the test suite.
# ---------------------------------------------------------------------------


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "semantic_cache_lookup",
        "description": (
            "Search the semantic cache for a previously cached response that is "
            "similar to the given query. Returns the cached response if "
            "similarity exceeds the threshold, otherwise returns a miss. Use "
            "this BEFORE making expensive LLM or API calls to check if a "
            "similar query has already been answered."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The query to search for in the cache",
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold 0.0-1.0 (default: 0.92)",
                },
                "cache_scope": {
                    "type": "string",
                    "description": (
                        "Optional cache scope (M28) — isolates the lookup by "
                        "tenant, user, system prompt, or any user-defined key."
                    ),
                },
            },
            "required": ["query_text"],
        },
    },
    {
        "name": "semantic_cache_store",
        "description": (
            "Store a query-response pair in the semantic cache so future "
            "similar queries can be answered without re-computation. Call "
            "this AFTER getting a response from an expensive operation (LLM "
            "call, API call, database query, etc.) to cache the result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The original query",
                },
                "response": {
                    "type": "string",
                    "description": "The response to cache",
                },
                "cache_scope": {
                    "type": "string",
                    "description": (
                        "Optional cache scope (M28) — must match the scope "
                        "used at lookup time for the entry to be visible."
                    ),
                },
            },
            "required": ["query_text", "response"],
        },
    },
    {
        "name": "cache_status",
        "description": "Check the health and statistics of the ferrocache server.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def list_tool_names() -> list[str]:
    return [t["name"] for t in TOOL_DEFINITIONS]


# ---------------------------------------------------------------------------
# MCP server wiring (only loaded when actually starting the server).
# ---------------------------------------------------------------------------


async def _dispatch_tool(
    tools: FerrocacheTools, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if name == "semantic_cache_lookup":
        return await tools.lookup(
            query_text=arguments.get("query_text", ""),
            threshold=arguments.get("threshold"),
            cache_scope=arguments.get("cache_scope"),
        )
    if name == "semantic_cache_store":
        return await tools.store(
            query_text=arguments.get("query_text", ""),
            response=arguments.get("response", ""),
            cache_scope=arguments.get("cache_scope"),
        )
    if name == "cache_status":
        return await tools.status()
    return {"error": f"unknown tool: {name}"}


def _build_tools_from_env() -> FerrocacheTools:
    url = os.environ.get("FERROCACHE_URL", DEFAULT_URL)
    threshold_raw = os.environ.get("FERROCACHE_THRESHOLD")
    try:
        threshold = float(threshold_raw) if threshold_raw is not None else DEFAULT_THRESHOLD
    except ValueError:
        log.warning("invalid FERROCACHE_THRESHOLD=%r; using %s", threshold_raw, DEFAULT_THRESHOLD)
        threshold = DEFAULT_THRESHOLD
    embed_model = os.environ.get("FERROCACHE_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    from ferrocache._embed import get_default_embed

    embed_fn, model_id = get_default_embed(embed_model)
    # FerrocacheClient picks up FERROCACHE_AUTH_TOKEN automatically.
    client = FerrocacheClient(url)
    return FerrocacheTools(
        client=client,
        embed_fn=embed_fn,
        model_id=model_id,
        default_threshold=threshold,
    )


async def _run_server() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError as e:
        raise ImportError(
            "MCP server requires the `mcp` package. Install it with `pip install mcp`."
        ) from e

    tools = _build_tools_from_env()
    server: Server = Server("ferrocache")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [Tool(**t) for t in TOOL_DEFINITIONS]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = await _dispatch_tool(tools, name, arguments)
        return [TextContent(type="text", text=json.dumps(result))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
