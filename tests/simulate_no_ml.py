"""ferrocache latency-only simulation (no ML deps).

Generates random 384-dim vectors instead of real embeddings — so the
hit rate is meaningless, but the insert/query path is exercised at the
same load as the full simulation. Use this when sentence-transformers
or PyTorch can't be installed.

Run:
    cargo run --release         # in another terminal
    python3 tests/simulate_no_ml.py
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from ferrocache import FerrocacheClient, FerrocacheError  # noqa: E402

DIM = 384
NUM_INSERTS = 50
NUM_QUERIES = 100
MODEL_ID = f"random::{DIM}"


def random_unit_vector(rng: random.Random) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt_ms(ms: float) -> str:
    return f"{ms:.2f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="ferrocache latency-only simulation")
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--inserts", type=int, default=NUM_INSERTS)
    parser.add_argument("--queries", type=int, default=NUM_QUERIES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        "NOTE: Using random vectors — hit rate is not meaningful. "
        "Use simulate.py with sentence-transformers for semantic testing.\n"
    )

    client = FerrocacheClient(args.url)
    try:
        health = client.health()
    except FerrocacheError as e:
        print(f"ERROR: ferrocache not reachable at {args.url}: {e}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    insert_latencies: list[float] = []
    query_latencies: list[float] = []

    print(f"Phase 1: inserting {args.inserts} random vectors...")
    for i in range(args.inserts):
        vec = random_unit_vector(rng)
        t0 = time.perf_counter()
        client.insert(
            embedding=vec, response=f"resp-{i}", query_text=f"q-{i}", model_id=MODEL_ID
        )
        insert_latencies.append((time.perf_counter() - t0) * 1000)

    print(f"Phase 2: running {args.queries} random queries...")
    hits = 0
    for _ in range(args.queries):
        vec = random_unit_vector(rng)
        t0 = time.perf_counter()
        result = client.query(embedding=vec, threshold=args.threshold, model_id=MODEL_ID)
        query_latencies.append((time.perf_counter() - t0) * 1000)
        if result.get("hit"):
            hits += 1

    health = client.health()
    line = "═" * 63
    print()
    print("ferrocache latency-only simulation results")
    print(line)
    print(f"Target:             {args.url}")
    print(f"Vector dim:         {DIM} (random unit vectors)")
    print(f"Threshold:          {args.threshold}")
    print()
    print("Workload")
    print(f"  Inserts:            {len(insert_latencies)}")
    print(f"  Queries:            {len(query_latencies)}")
    print(f"  Random hits:        {hits} (incidental — not semantically meaningful)")
    print()
    print("Latency (ferrocache round-trip)")
    print(
        f"  Insert:             p50={fmt_ms(percentile(insert_latencies, 0.5))}  "
        f"p99={fmt_ms(percentile(insert_latencies, 0.99))}   "
        f"mean={fmt_ms(statistics.mean(insert_latencies))}"
    )
    print(
        f"  Query:              p50={fmt_ms(percentile(query_latencies, 0.5))}  "
        f"p99={fmt_ms(percentile(query_latencies, 0.99))}   "
        f"mean={fmt_ms(statistics.mean(query_latencies))}"
    )
    print()
    print("Cache State")
    print(f"  Entry count:        {health.get('entry_count')}")
    print(f"  Node:               {health.get('node_id')}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
