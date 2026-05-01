"""Tests for the MCP server's tool dispatch. No real ferrocache or mcp transport required."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from ferrocache.client import FerrocacheError  # noqa: E402
from ferrocache.mcp_server import (  # noqa: E402
    TOOL_DEFINITIONS,
    FerrocacheTools,
    _dispatch_tool,
)


def fake_embed(_: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def make_tools(
    query_return: dict | None = None,
    insert_return: dict | None = None,
    health_return: dict | None = None,
    query_raises: Exception | None = None,
    insert_raises: Exception | None = None,
) -> tuple[FerrocacheTools, MagicMock]:
    client = MagicMock()
    if query_raises is not None:
        client.query.side_effect = query_raises
    else:
        client.query.return_value = query_return or {"hit": False}
    if insert_raises is not None:
        client.insert.side_effect = insert_raises
    else:
        client.insert.return_value = insert_return or {"id": "u-1", "status": "ok"}
    client.health.return_value = health_return or {
        "status": "ok",
        "node_id": "test-node",
        "entry_count": 0,
    }
    return FerrocacheTools(client=client, embed_fn=fake_embed), client


class McpToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_hit(self) -> None:
        tools, client = make_tools(
            query_return={"hit": True, "id": "u-1", "response": "cached", "similarity": 0.95}
        )
        result = await tools.lookup("the query")
        self.assertTrue(result["hit"])
        self.assertEqual(result["response"], "cached")
        self.assertEqual(client.query.call_args.kwargs["embedding"], fake_embed(""))

    async def test_lookup_miss(self) -> None:
        tools, _ = make_tools(query_return={"hit": False})
        result = await tools.lookup("the query")
        self.assertEqual(result, {"hit": False})

    async def test_store_success(self) -> None:
        tools, client = make_tools(
            insert_return={"id": "abc-123", "status": "ok"}
        )
        result = await tools.store("the query", "the response")
        self.assertEqual(result, {"id": "abc-123", "status": "ok"})
        kwargs = client.insert.call_args.kwargs
        self.assertEqual(kwargs["embedding"], fake_embed(""))
        self.assertEqual(kwargs["response"], "the response")
        self.assertEqual(kwargs["query_text"], "the query")

    async def test_cache_status(self) -> None:
        tools, _ = make_tools(
            health_return={"status": "ok", "node_id": "n", "entry_count": 42}
        )
        result = await tools.status()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["entry_count"], 42)

    async def test_lookup_ferrocache_unreachable(self) -> None:
        tools, _ = make_tools(query_raises=FerrocacheError("connection refused"))
        result = await tools.lookup("q")
        self.assertIn("error", result)
        self.assertIn("ferrocache unreachable", result["error"])

    async def test_store_ferrocache_unreachable(self) -> None:
        tools, _ = make_tools(insert_raises=FerrocacheError("down"))
        result = await tools.store("q", "r")
        self.assertIn("error", result)
        self.assertIn("ferrocache unreachable", result["error"])

    async def test_dispatch_unknown_tool(self) -> None:
        tools, _ = make_tools()
        result = await _dispatch_tool(tools, "not_a_tool", {})
        self.assertIn("error", result)
        self.assertIn("unknown tool", result["error"])


class ToolCatalogTests(unittest.TestCase):
    def test_list_tools_returns_three(self) -> None:
        self.assertEqual(len(TOOL_DEFINITIONS), 3)
        names = [t["name"] for t in TOOL_DEFINITIONS]
        self.assertEqual(
            sorted(names),
            ["cache_status", "semantic_cache_lookup", "semantic_cache_store"],
        )
        for tool in TOOL_DEFINITIONS:
            self.assertIn("description", tool)
            self.assertIn("inputSchema", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")


if __name__ == "__main__":
    unittest.main()
