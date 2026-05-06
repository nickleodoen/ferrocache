# MCP Server (Claude Desktop / Claude Code)

FerroCache ships an MCP server that exposes semantic caching as tools to any MCP-capable agent. Tools accept text — the MCP layer embeds locally before talking to FerroCache.

## Install

```bash
pip install ferrocache[mcp]
# or, for development from a clone:
pip install -r clients/python/mcp_requirements.txt
```

This installs the official Anthropic `mcp` SDK and `sentence-transformers` (for local embedding).

## Run the server

```bash
python3 -m ferrocache.mcp_server
```

The server speaks JSON-RPC over stdio. You don't run it directly in normal use — your MCP client (Claude Desktop, Claude Code) launches it.

## Configure Claude Desktop

Edit your config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

Add a `ferrocache` server entry. Replace `/path/to/ferrocache` with the absolute path to your clone (or omit `PYTHONPATH` if you `pip install ferrocache[mcp]` globally):

```json
{
  "mcpServers": {
    "ferrocache": {
      "command": "python3",
      "args": ["-m", "ferrocache.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/ferrocache/clients/python",
        "FERROCACHE_URL": "http://localhost:3000",
        "FERROCACHE_THRESHOLD": "0.92",
        "FERROCACHE_EMBED_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

Restart Claude Desktop. The three FerroCache tools will appear in the tool picker.

## Configure Claude Code

```bash
claude mcp add ferrocache -- python3 -m ferrocache.mcp_server
```

Then export the same environment variables in your shell profile:

```bash
export PYTHONPATH="/path/to/ferrocache/clients/python"
export FERROCACHE_URL="http://localhost:3000"
export FERROCACHE_THRESHOLD="0.92"
```

## Tools

| Tool | Purpose |
|---|---|
| `semantic_cache_lookup` | Search by query text. Optional `cache_scope`, `conversation_id`. |
| `semantic_cache_store` | Store a query-response pair. Optional `cache_scope`, `conversation_id`. |
| `cache_status` | Server health + entry count. |

## How it works

1. Claude calls `semantic_cache_lookup` with `query_text="..."`.
2. The MCP server embeds the query locally with `all-MiniLM-L6-v2` (384-dim).
3. It hits FerroCache `/query` with the float vector and threshold.
4. On a hit, Claude gets the cached response back without making the expensive call.
5. On a miss, Claude proceeds to its normal answer path, then calls `semantic_cache_store` to remember the result for next time.

The MCP layer absorbs FerroCache outages — every error becomes `{"error": "..."}` in the tool response, so the conversation keeps moving.

## Things to ask after wiring it up

- "Check the FerroCache status."
- "Look up if there's a cached answer for 'What is the company vacation policy?'"
- "Remember this answer in the cache for next time."

## Troubleshooting

- **Tools don't appear:** check `~/Library/Logs/Claude/mcp*.log` for stderr from the server. Most often the path to `python3` or the package isn't right.
- **Every call returns `{"error": ...}`:** FerroCache itself isn't running, or `FERROCACHE_URL` is wrong. Hit `/health` directly to confirm.
- **Slow first call:** `sentence-transformers` lazily downloads model weights on first use. Pre-warm with `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`.
