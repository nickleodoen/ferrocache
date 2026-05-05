#!/usr/bin/env python3
"""
ferrocache vs GPTCache benchmark.

Runs the same workload against a single-node ferrocache (assumed to be
listening on localhost:3000) and an in-process GPTCache, then reports a
markdown comparison table to stdout.

Workload:
- 200 seed query/answer pairs (generated synthetically — no external data)
- 3 paraphrase variations per seed → 600 expected-hit queries
- 50 unrelated queries → expected misses
- ~750 queries total against ~200 inserted entries

Skips gracefully when optional deps are missing:
    pip install sentence-transformers gptcache psutil

Usage:
    # Start ferrocache first:
    cargo run --release &
    sleep 2
    # Then:
    python3 tests/benchmark_vs_gptcache.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))


def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


SENTENCE_TRANSFORMERS = _try_import("sentence_transformers")
GPTCACHE = _try_import("gptcache")
PSUTIL = _try_import("psutil")


def percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def gen_workload(seed_count: int = 200, variations: int = 3, unrelated: int = 50):
    """Synthetic workload — deterministic, no external data."""
    seeds: list[tuple[str, str]] = []
    for i in range(seed_count):
        q = f"What is the capital of region {i}?"
        a = f"The capital of region {i} is City{i}."
        seeds.append((q, a))

    # Three paraphrases per seed. Crude but consistent across systems.
    queries: list[tuple[str, str]] = []
    for q, a in seeds:
        queries.append((q, a))  # exact
        queries.append((f"Tell me {q.lower()}", a))
        queries.append((f"{q} I'd like to know.", a))

    unrelated_qs = [
        f"What is the recipe for dish number {i}?" for i in range(unrelated)
    ]
    return seeds, queries, unrelated_qs


def measure_rss(pid: int) -> int:
    """Resident-set size in bytes for the given pid. 0 on failure."""
    if PSUTIL is not None:
        try:
            return PSUTIL.Process(pid).memory_info().rss
        except Exception:
            return 0
    # Fallback: /proc on Linux. macOS has no /proc, so this returns 0.
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def find_ferrocache_pid() -> int:
    """Best-effort search for a `ferrocache` binary owned by this user."""
    if PSUTIL is None:
        return 0
    for p in PSUTIL.process_iter(["name", "exe"]):
        try:
            name = p.info.get("name") or ""
            if "ferrocache" in name.lower():
                return p.pid
        except Exception:
            continue
    return 0


# ----------------------------------------------------------------------------
# ferrocache benchmark
# ----------------------------------------------------------------------------


def run_ferrocache(model, seeds, queries, unrelated_qs, threshold: float):
    """Returns dict of metrics, or None if ferrocache is unreachable."""
    from ferrocache import FerrocacheClient, FerrocacheError  # type: ignore

    client = FerrocacheClient("http://localhost:3000")
    try:
        client.health()
    except FerrocacheError as e:
        print(f"ferrocache unreachable: {e}", file=sys.stderr)
        return None

    dim = model.get_sentence_embedding_dimension()
    model_id = f"bench-vs-gptcache::{dim}"

    seed_texts = [q for q, _ in seeds]
    seed_vecs = [v.tolist() for v in model.encode(seed_texts, show_progress_bar=False)]
    answers = [a for _, a in seeds]

    insert_latencies: list[float] = []
    for vec, q, a in zip(seed_vecs, seed_texts, answers):
        t0 = time.perf_counter()
        client.insert(embedding=vec, response=a, query_text=q, model_id=model_id)
        insert_latencies.append((time.perf_counter() - t0) * 1000)

    # Hit-path queries
    hit_latencies: list[float] = []
    hits = 0
    query_texts = [q for q, _ in queries]
    query_vecs = [v.tolist() for v in model.encode(query_texts, show_progress_bar=False)]
    for vec, q in zip(query_vecs, query_texts):
        t0 = time.perf_counter()
        r = client.query(
            embedding=vec, threshold=threshold, model_id=model_id, query_text=q
        )
        hit_latencies.append((time.perf_counter() - t0) * 1000)
        if r.get("hit"):
            hits += 1
    hit_rate = hits / len(queries) if queries else 0.0

    # Miss-path queries
    unrelated_vecs = [v.tolist() for v in model.encode(unrelated_qs, show_progress_bar=False)]
    miss_latencies: list[float] = []
    false_hits = 0
    for vec, q in zip(unrelated_vecs, unrelated_qs):
        t0 = time.perf_counter()
        r = client.query(
            embedding=vec, threshold=threshold, model_id=model_id, query_text=q
        )
        miss_latencies.append((time.perf_counter() - t0) * 1000)
        if r.get("hit"):
            false_hits += 1

    rss = measure_rss(find_ferrocache_pid())

    # Concurrent insert throughput at concurrency 50.
    conc = 50
    duration = 3.0
    stop_at = time.perf_counter() + duration
    counter = {"n": 0}

    def insert_loop(seed_offset: int):
        c = FerrocacheClient("http://localhost:3000")
        local = 0
        while time.perf_counter() < stop_at:
            i = counter["n"] + seed_offset
            vec = seed_vecs[i % len(seed_vecs)]
            try:
                c.insert(
                    embedding=vec,
                    response=f"thr-{i}",
                    query_text=f"thr-q-{i}",
                    model_id=f"throughput::{dim}",
                )
                local += 1
            except FerrocacheError:
                pass
        counter["n"] += local
        return local

    with ThreadPoolExecutor(max_workers=conc) as ex:
        futures = [ex.submit(insert_loop, k) for k in range(conc)]
        total = sum(f.result() for f in futures)
    insert_throughput = total / duration

    return {
        "hit_rate": hit_rate,
        "false_hits": false_hits,
        "query_p50": percentile(hit_latencies + miss_latencies, 50),
        "query_p99": percentile(hit_latencies + miss_latencies, 99),
        "insert_p50": percentile(insert_latencies, 50),
        "insert_p99": percentile(insert_latencies, 99),
        "rss_bytes": rss,
        "insert_throughput": insert_throughput,
    }


# ----------------------------------------------------------------------------
# GPTCache benchmark
# ----------------------------------------------------------------------------


def run_gptcache_subprocess(seeds, queries, unrelated_qs, threshold: float):
    """Run the GPTCache benchmark in a child process so a segfault (faiss /
    sqlite on newer Pythons) doesn't take down the main process. Returns
    None on any failure."""
    import json as _json
    import subprocess

    payload = _json.dumps(
        {
            "seeds": seeds,
            "queries": queries,
            "unrelated": unrelated_qs,
            "threshold": threshold,
        }
    )
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--gptcache-worker"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("GPTCache subprocess timed out", file=sys.stderr)
        return None
    if proc.returncode != 0:
        # Likely SIGSEGV on Python 3.13+ where faiss-cpu wheels lag behind.
        print(
            f"GPTCache subprocess exited with code {proc.returncode}; "
            "treating as N/A.",
            file=sys.stderr,
        )
        if proc.stderr:
            print(proc.stderr.splitlines()[-1] if proc.stderr.splitlines() else "", file=sys.stderr)
        return None
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return _json.loads(last_line)
    except Exception as e:
        print(f"GPTCache subprocess output unparseable: {e}", file=sys.stderr)
        return None


def run_gptcache(model, seeds, queries, unrelated_qs, threshold: float):
    """Returns dict of metrics, or None if GPTCache isn't installed.

    GPTCache's `init_similar_cache` defaults to ONNX which won't install on
    Python 3.14. Construct the cache manually with the shared
    sentence-transformers embeddings instead, so both systems compare against
    identical vectors.
    """
    if GPTCACHE is None:
        return None

    try:
        from gptcache import Cache  # type: ignore
        from gptcache.adapter.api import get as cache_get  # type: ignore
        from gptcache.adapter.api import put as cache_put  # type: ignore
        from gptcache.embedding.base import BaseEmbedding  # type: ignore
        from gptcache.manager import CacheBase, VectorBase, get_data_manager  # type: ignore
        from gptcache.similarity_evaluation.distance import (  # type: ignore
            SearchDistanceEvaluation,
        )
    except Exception as e:
        print(f"GPTCache imports failed: {e}", file=sys.stderr)
        return None

    dim = model.get_sentence_embedding_dimension()

    class STEmbedding(BaseEmbedding):
        """Adapter so GPTCache uses the same sentence-transformers model
        as ferrocache. Single-text only (matches GPTCache's `to_embeddings`
        contract — it embeds the prompt one at a time)."""

        def to_embeddings(self, data, **_kwargs):
            return model.encode([data], show_progress_bar=False)[0]

        def dimension(self):
            return dim

    embedding = STEmbedding()
    data_manager = get_data_manager(
        CacheBase("sqlite"), VectorBase("faiss", dimension=dim)
    )

    cache = Cache()
    # GPTCache's default `pre_embedding_func` (`last_content`) expects an
    # OpenAI-style `{"messages": [...]}` dict. The benchmark passes plain
    # strings via `cache_put(prompt, answer)`, so override `pre_embedding_func`
    # to return the prompt directly (the kwargs pipeline forwards the user
    # data verbatim under the `prompt` key).
    def pre_func(data, **_kwargs):
        return data["prompt"]

    try:
        cache.init(
            pre_embedding_func=pre_func,
            embedding_func=embedding.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
        )
    except Exception as e:
        print(f"GPTCache init failed: {e}", file=sys.stderr)
        return None

    seed_texts = [q for q, _ in seeds]
    answers = [a for _, a in seeds]

    insert_latencies: list[float] = []
    for q, a in zip(seed_texts, answers):
        t0 = time.perf_counter()
        try:
            cache_put(q, a, cache_obj=cache)
        except Exception as e:
            print(f"GPTCache put failed: {e}", file=sys.stderr)
            return None
        insert_latencies.append((time.perf_counter() - t0) * 1000)

    # Hit-path queries
    hit_latencies: list[float] = []
    hits = 0
    for q, _ in queries:
        t0 = time.perf_counter()
        try:
            r = cache_get(q, cache_obj=cache)
        except Exception:
            r = None
        hit_latencies.append((time.perf_counter() - t0) * 1000)
        if r is not None:
            hits += 1
    hit_rate = hits / len(queries) if queries else 0.0

    # Miss-path queries
    miss_latencies: list[float] = []
    false_hits = 0
    for q in unrelated_qs:
        t0 = time.perf_counter()
        try:
            r = cache_get(q, cache_obj=cache)
        except Exception:
            r = None
        miss_latencies.append((time.perf_counter() - t0) * 1000)
        if r is not None:
            false_hits += 1

    rss = measure_rss(os.getpid())

    return {
        "hit_rate": hit_rate,
        "false_hits": false_hits,
        "query_p50": percentile(hit_latencies + miss_latencies, 50),
        "query_p99": percentile(hit_latencies + miss_latencies, 99),
        "insert_p50": percentile(insert_latencies, 50),
        "insert_p99": percentile(insert_latencies, 99),
        "rss_bytes": rss,
        "insert_throughput": None,  # GPTCache is single-threaded
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def fmt_ms(v):
    return "N/A" if v is None else f"{v:.2f}ms"


def fmt_pct(v):
    return "N/A" if v is None else f"{v * 100:.1f}%"


def fmt_mb(b):
    return "N/A" if not b else f"{b / 1024 / 1024:.1f} MB"


def fmt_ops(v):
    return "N/A" if v is None else f"{v:.0f}/s"


def report(fc, gc):
    rows = [
        ("Hit rate (threshold 0.90)", fmt_pct(fc and fc["hit_rate"]), fmt_pct(gc and gc["hit_rate"])),
        ("False hits on unrelated", str(fc and fc["false_hits"] or 0), str(gc and gc["false_hits"] or 0)),
        ("Query latency p50", fmt_ms(fc and fc["query_p50"]), fmt_ms(gc and gc["query_p50"])),
        ("Query latency p99", fmt_ms(fc and fc["query_p99"]), fmt_ms(gc and gc["query_p99"])),
        ("Insert latency p50", fmt_ms(fc and fc["insert_p50"]), fmt_ms(gc and gc["insert_p50"])),
        ("Insert latency p99", fmt_ms(fc and fc["insert_p99"]), fmt_ms(gc and gc["insert_p99"])),
        ("RSS after seed inserts", fmt_mb(fc and fc["rss_bytes"]), fmt_mb(gc and gc["rss_bytes"])),
        (
            "Insert throughput (concurrency 50)",
            fmt_ops(fc and fc["insert_throughput"]),
            fmt_ops(gc and gc["insert_throughput"]),
        ),
    ]
    print()
    print("| Metric | ferrocache | GPTCache |")
    print("|---|---|---|")
    for name, a, b in rows:
        print(f"| {name} | {a} | {b} |")
    print()
    print("Notes:")
    print("- ferrocache query latency includes HTTP round-trip; GPTCache is in-process.")
    print("- ferrocache insert includes WAL fsync (durable); GPTCache is in-memory.")
    print("- Both use sentence-transformers `all-MiniLM-L6-v2` (384-dim) for embeddings.")
    print("- GPTCache is single-threaded; ferrocache throughput row exercises 50 concurrent clients.")


def gptcache_worker_main():
    """Subprocess entry point — reads a JSON payload from stdin, runs the
    GPTCache benchmark, prints a JSON result on the last stdout line. Any
    segfault crashes ONLY this child process; the parent treats the empty
    output as N/A."""
    import json as _json

    if SENTENCE_TRANSFORMERS is None or GPTCACHE is None:
        return 1
    from sentence_transformers import SentenceTransformer  # type: ignore

    payload = _json.loads(sys.stdin.read())
    seeds = [tuple(p) for p in payload["seeds"]]
    queries = [tuple(p) for p in payload["queries"]]
    unrelated = list(payload["unrelated"])
    threshold = float(payload["threshold"])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    result = run_gptcache(model, seeds, queries, unrelated, threshold)
    if result is None:
        return 2
    print(_json.dumps(result))
    return 0


def main() -> int:
    if "--gptcache-worker" in sys.argv:
        return gptcache_worker_main()

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--variations", type=int, default=3)
    parser.add_argument("--unrelated", type=int, default=50)
    args = parser.parse_args()

    if SENTENCE_TRANSFORMERS is None:
        print(
            "Skipping benchmark — sentence-transformers not installed.\n"
            "Install with: pip install sentence-transformers gptcache psutil",
            file=sys.stderr,
        )
        return 0

    from sentence_transformers import SentenceTransformer  # type: ignore

    print("Loading sentence-transformers model...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Generating workload...", file=sys.stderr)
    seeds, queries, unrelated_qs = gen_workload(
        seed_count=args.seeds, variations=args.variations, unrelated=args.unrelated
    )

    print(f"Workload: {len(seeds)} inserts, {len(queries)} expected-hit queries, "
          f"{len(unrelated_qs)} unrelated", file=sys.stderr)

    print("\nRunning ferrocache...", file=sys.stderr)
    fc = run_ferrocache(model, seeds, queries, unrelated_qs, args.threshold)
    if fc is None:
        print(
            "Skipping ferrocache — server unreachable on localhost:3000.\n"
            "Start it first with: cargo run --release",
            file=sys.stderr,
        )

    print("Running GPTCache (in subprocess to isolate native crashes)...", file=sys.stderr)
    gc = run_gptcache_subprocess(seeds, queries, unrelated_qs, args.threshold)
    if gc is None:
        print(
            "GPTCache run skipped — install `gptcache` and ensure your Python "
            "is compatible with `faiss-cpu` (Python <= 3.12 recommended).",
            file=sys.stderr,
        )

    if fc is None and gc is None:
        return 1

    report(fc, gc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
