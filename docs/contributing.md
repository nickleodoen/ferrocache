# Contributing

FerroCache is actively developed and welcomes contributions.

## Development setup

```bash
git clone https://github.com/nickleodoen/ferrocache
cd ferrocache

# Rust build
cargo build
cargo test                                # ~222 unit tests
cargo clippy --all-targets -- -D warnings
cargo fmt --check

# Python client tests
cd clients/python
python3 -m unittest discover tests        # ~59 tests
cd ../..

# Cluster integration (requires Docker)
make cluster-test                         # spins up 3 nodes, runs 44 assertions
```

Recommended toolchain: Rust 1.75+ via `rustup`, Python 3.10+ for the client.

## Running benchmarks

```bash
cargo bench                               # criterion microbenchmarks
make bench-concurrent                     # concurrent HTTP throughput
make benchmark-vs-gptcache                # FerroCache vs GPTCache (Python ≤ 3.12)
make simulate                             # realistic FAQ workload
```

## Three high-impact contribution areas

### 1. Embedding model integrations

FerroCache is embedding-agnostic by design — the client computes the vector. But most users want a default that just works. Adding first-class support for **Voyage AI**, **Cohere**, and local **Ollama** models to the Python client's auto-embed path would lower the barrier to adoption significantly.

Good first issue: add `ferrocache[voyage]` extra with a Voyage AI `embed_fn`.

### 2. Async Python client

The Python client and all middleware wrappers are synchronous. Modern Python LLM applications are async-native (LangChain LCEL, the async Anthropic client, FastAPI). An `AsyncFerrocacheClient` built on `httpx.AsyncClient` would unblock this entire class of users.

Good first issue: implement `AsyncFerrocacheClient` mirroring the sync client's API.

### 3. Load testing and real-world benchmarks

The current benchmarks run on synthetic FAQ workloads. Real-world hit rate data on production query distributions (MS MARCO, customer support logs, coding assistant queries) would help users calibrate their threshold and make the project more credible to evaluators.

Good first issue: publish a benchmark notebook using the MS MARCO dataset.

## Code style

- **Rust**: `cargo fmt` (default rustfmt) + `cargo clippy --all-targets -- -D warnings`. No allows without a `// SAFETY:` or `// NOTE:` justification.
- **Python**: PEP 8, no formatter enforced. Match the surrounding file. Type hints encouraged but not required.
- **No new dependencies without discussion.** The Python client is stdlib-only; the Rust binary keeps a small dep tree on purpose.

## PR process

1. Fork and branch off `main`.
2. Run the full test matrix locally:
   ```bash
   cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --check
   cd clients/python && python3 -m unittest discover tests
   ```
3. For changes that affect the cluster code path, also run `make cluster-test`.
4. Open a PR with:
   - One-paragraph description of the *why*.
   - List of modules touched.
   - Any benchmark numbers if it's a perf change.
5. CI runs `check`/`test`/`clippy`/`fmt` plus the docker-compose cluster integration on every push.

## Issue labels

- [`good first issue`](https://github.com/nickleodoen/ferrocache/issues?q=label%3A%22good+first+issue%22) — scoped, well-defined, no architectural decisions.
- [`help wanted`](https://github.com/nickleodoen/ferrocache/issues?q=label%3A%22help+wanted%22) — needs more thought; design discussion welcome.
- [`bug`](https://github.com/nickleodoen/ferrocache/issues?q=label%3Abug) / [`enhancement`](https://github.com/nickleodoen/ferrocache/issues?q=label%3Aenhancement) — self-explanatory.

## Filing bugs

Include:

- FerroCache version (`docker image inspect ghcr.io/nickleodoen/ferrocache:latest | grep -i version` or `cargo pkgid`).
- Cluster topology (`/cluster/status` output).
- Minimal reproduction (curl commands, embedding dimension, threshold).
- WAL inspection if relevant: `head -50 ferrocache.wal`.

Don't include real cached responses (they may contain sensitive content) — synthesise a repro.

## License

MIT. By contributing you agree your changes are licensed under the same.
