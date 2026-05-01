# Setting up ferrocache with Claude Desktop / Claude Code

This guide wires the `ferrocache.mcp_server` into an MCP-capable client so the agent can call `semantic_cache_lookup`, `semantic_cache_store`, and `cache_status` as tools during conversations.

## Prerequisites

- ferrocache server running: `cargo run --release` or `docker compose up -d --build`
- Python 3.9+ with `pip`

## Install the MCP server dependencies

From the repo root:

```bash
pip install -r clients/python/mcp_requirements.txt
```

This pulls in `mcp` (the official Anthropic MCP SDK) and `sentence-transformers` (for local embedding — the MCP layer accepts text and embeds it before talking to ferrocache).

## Configure Claude Desktop

Edit your config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

Add a `ferrocache` server entry. Replace `/path/to/ferrocache` with the absolute path to your clone:

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

Restart Claude Desktop. The three ferrocache tools should appear in the tool list.

## Configure Claude Code

```bash
claude mcp add ferrocache -- python3 -m ferrocache.mcp_server
```

Then export the same environment variables in your shell profile (or pass them via `--env` flags Claude Code may support):

```bash
export PYTHONPATH="/path/to/ferrocache/clients/python"
export FERROCACHE_URL="http://localhost:3000"
export FERROCACHE_THRESHOLD="0.92"
```

## Tools exposed

| Tool                     | Purpose                                                    |
|--------------------------|------------------------------------------------------------|
| `semantic_cache_lookup`  | Search for a cached response similar to a given query     |
| `semantic_cache_store`   | Store a query-response pair so future similar asks hit    |
| `cache_status`           | Health + entry count from `/health`                       |

Things to ask Claude after wiring it up:

- "Check the ferrocache status."
- "Look up if there's a cached answer for 'What is the company vacation policy?'"
- "Remember this answer in the cache for next time."

## How it works

1. Claude calls `semantic_cache_lookup` with `query_text="..."`.
2. The MCP server embeds the query locally with `all-MiniLM-L6-v2` (384-dim).
3. It hits ferrocache's `/query` endpoint with the float vector and threshold.
4. On a hit, Claude gets the cached response back without making the expensive call.
5. On a miss, Claude proceeds to whatever tool/API would normally answer the question, then calls `semantic_cache_store` to remember the result.

The MCP layer absorbs ferrocache outages — every error becomes `{"error": "..."}` in the tool response, so the conversation keeps moving even if the cache is down.
