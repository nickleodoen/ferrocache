# Contributing to FerroCache

See the full contributing guide at the documentation site:
https://nickleodoen.github.io/ferrocache/contributing/

## Quick start

```bash
git clone https://github.com/nickleodoen/ferrocache
cd ferrocache
cargo build --release
cargo test
cargo clippy --all-targets -- -D warnings
```

## Three areas where contributions have the most impact

1. Embedding model integrations (Voyage AI, Cohere, Ollama)
2. Async Python client (httpx-based AsyncFerrocacheClient)
3. Real-world benchmarks on MS MARCO or production query distributions

Open issues labeled `good first issue` are good starting points.
PRs welcome — please run `cargo fmt` and `cargo clippy` before submitting.
