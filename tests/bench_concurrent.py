#!/usr/bin/env python3
"""
Concurrent load benchmark for ferrocache.

Runs insert and query workloads at various concurrency levels through the
existing Python client (HTTP) and reports throughput (ops/sec) and latency
percentiles. Designed to demonstrate the WAL serialization bottleneck and
the group-commit speedup.

Usage:
    python3 tests/bench_concurrent.py [--url URL] [--concurrency 1,10,50,100] [--duration 10]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event

# Make the in-tree client importable without installing.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "clients", "python"))
from ferrocache import FerrocacheClient, FerrocacheError  # noqa: E402


def random_unit_vector(dim: int, rng: random.Random) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1e-9
    return [x / norm for x in v]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


class Workload:
    name: str

    def setup(self, client: FerrocacheClient) -> None:
        pass

    def call(self, client: FerrocacheClient, worker_id: int, iter_id: int) -> None:
        raise NotImplementedError


class InsertWorkload(Workload):
    name = "INSERT"

    def __init__(self, dim: int, model_id: str, pool_size: int = 4096) -> None:
        self.dim = dim
        self.model_id = model_id
        rng = random.Random(0xFE7700)
        self.vectors = [random_unit_vector(dim, rng) for _ in range(pool_size)]

    def call(self, client: FerrocacheClient, worker_id: int, iter_id: int) -> None:
        # Each worker pulls a deterministic-but-distinct vector from the pool;
        # adding worker offset prevents two workers writing identical vectors.
        v = self.vectors[(worker_id * 7919 + iter_id) % len(self.vectors)]
        client._post(
            "/insert",
            {
                "embedding": v,
                "response": "bench-r",
                "query_text": "bench-q",
                "model_id": self.model_id,
                "uuid": _uuid.uuid4().hex,
            },
        )


class QueryHitWorkload(Workload):
    name = "QUERY (hit)"

    def __init__(self, dim: int, model_id: str, n_prepop: int = 1000) -> None:
        self.dim = dim
        self.model_id = model_id
        self.n_prepop = n_prepop
        rng = random.Random(0xCAFE)
        self.vectors = [random_unit_vector(dim, rng) for _ in range(n_prepop)]

    def setup(self, client: FerrocacheClient) -> None:
        # Pre-populate sequentially so the workload that follows is read-only.
        for v in self.vectors:
            client._post(
                "/insert",
                {
                    "embedding": v,
                    "response": "pre",
                    "query_text": "pre",
                    "model_id": self.model_id,
                    "uuid": _uuid.uuid4().hex,
                },
            )

    def call(self, client: FerrocacheClient, worker_id: int, iter_id: int) -> None:
        v = self.vectors[(worker_id * 31 + iter_id) % len(self.vectors)]
        client.query(embedding=v, threshold=0.85, model_id=self.model_id)


class QueryMissWorkload(Workload):
    name = "QUERY (miss)"

    def __init__(self, dim: int, model_id: str) -> None:
        self.dim = dim
        self.model_id = model_id
        rng = random.Random(0xDEAD)
        # Random vectors with extreme threshold → guaranteed miss path.
        self.vectors = [random_unit_vector(dim, rng) for _ in range(2048)]

    def call(self, client: FerrocacheClient, worker_id: int, iter_id: int) -> None:
        v = self.vectors[(worker_id * 13 + iter_id) % len(self.vectors)]
        client.query(embedding=v, threshold=0.999, model_id=self.model_id)


def run_workload(
    workload: Workload,
    url: str,
    auth_token: str | None,
    concurrency: int,
    duration: float,
) -> tuple[int, list[float], int]:
    """Drive `workload` from `concurrency` threads for `duration` seconds.

    Returns (ops_completed, per_op_latencies_ms, errors).
    """
    stop = Event()
    latencies: list[list[float]] = [[] for _ in range(concurrency)]
    errors = [0] * concurrency

    def worker(worker_id: int) -> None:
        client = FerrocacheClient(url, auth_token=auth_token)
        local_lat = latencies[worker_id]
        local_err = 0
        i = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                workload.call(client, worker_id, i)
            except FerrocacheError:
                local_err += 1
            except Exception:
                local_err += 1
            local_lat.append((time.perf_counter() - t0) * 1000.0)
            i += 1
        errors[worker_id] = local_err

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(concurrency)]
        time.sleep(duration)
        stop.set()
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start

    flat = [x for sub in latencies for x in sub]
    err_total = sum(errors)
    # ops_completed counts everything we measured a latency for (including errors).
    # Throughput excludes errors.
    return (len(flat) - err_total, flat, err_total)


def fmt_row(parts: list[str]) -> str:
    widths = [13, 12, 12, 12, 8]
    return "  " + "".join(p.ljust(w) for p, w in zip(parts, widths))


def print_table(name: str, rows: list[tuple[int, int, float, float, int]]) -> None:
    print(f"\n{name} throughput")
    print(fmt_row(["Concurrency", "Ops/sec", "p50 (ms)", "p99 (ms)", "Errors"]))
    print("  " + "-" * 56)
    for c, ops_sec, p50, p99, errs in rows:
        print(
            fmt_row(
                [
                    str(c),
                    f"{ops_sec:.0f}",
                    f"{p50:.2f}",
                    f"{p99:.2f}",
                    str(errs),
                ]
            )
        )


def diagnose(rows: list[tuple[int, int, float, float, int]]) -> str:
    """A blunt heuristic: if highest-concurrency throughput is <1.5× the
    single-thread throughput AND p50 grows roughly linearly with concurrency,
    call out WAL serialization."""
    if len(rows) < 2:
        return ""
    base_ops = rows[0][1] or 1
    top_ops = rows[-1][1]
    base_p50 = rows[0][2] or 0.01
    top_p50 = rows[-1][2]
    top_c = rows[-1][0]
    if top_ops < base_ops * 1.5 and top_p50 > base_p50 * (top_c * 0.5):
        return (
            "DIAGNOSIS: Insert throughput is flat across concurrency levels\n"
            "           (WAL mutex serializes writers — fsync is the bottleneck)"
        )
    if top_ops > base_ops * 3:
        return (
            "DIAGNOSIS: Throughput scales with concurrency — group-commit (or\n"
            "           a non-fsync path) is amortizing the per-op cost."
        )
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--concurrency", default="1,10,50,100")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--model-id", default="bench-concurrent::384")
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("FERROCACHE_AUTH_TOKEN"),
        help="Bearer token; defaults to $FERROCACHE_AUTH_TOKEN",
    )
    parser.add_argument(
        "--workloads",
        default="insert,query_hit,query_miss",
        help="Comma list: insert, query_hit, query_miss",
    )
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    workloads_to_run = [w.strip() for w in args.workloads.split(",") if w.strip()]

    # Smoke check the server.
    probe = FerrocacheClient(args.url, auth_token=args.auth_token)
    try:
        probe.health()
    except FerrocacheError as e:
        print(f"ferrocache at {args.url} is not reachable: {e}", file=sys.stderr)
        return 2

    print("ferrocache concurrent benchmark")
    print("=" * 67)
    print(f"Target:     {args.url}")
    print(f"Duration:   {args.duration:g}s per test")
    print(f"Model ID:   {args.model_id}")
    print(f"Embed dim:  {args.dim}")

    workloads: list[Workload] = []
    for name in workloads_to_run:
        if name == "insert":
            workloads.append(InsertWorkload(args.dim, args.model_id + "::insert"))
        elif name == "query_hit":
            workloads.append(QueryHitWorkload(args.dim, args.model_id + "::qh"))
        elif name == "query_miss":
            workloads.append(QueryMissWorkload(args.dim, args.model_id + "::qm"))
        else:
            print(f"unknown workload: {name}", file=sys.stderr)
            return 2

    all_rows: list[tuple[str, list[tuple[int, int, float, float, int]]]] = []
    for wl in workloads:
        # One-shot setup before the timed runs.
        wl.setup(probe)
        rows: list[tuple[int, int, float, float, int]] = []
        for c in levels:
            ops, lat_ms, errs = run_workload(
                wl, args.url, args.auth_token, c, args.duration
            )
            ops_sec = ops / args.duration if args.duration else 0.0
            p50 = percentile(lat_ms, 0.5)
            p99 = percentile(lat_ms, 0.99)
            rows.append((c, int(ops_sec), p50, p99, errs))
        print_table(wl.name, rows)
        all_rows.append((wl.name, rows))

    # Diagnosis specifically for insert (the M20 motivation).
    for name, rows in all_rows:
        if name.startswith("INSERT"):
            note = diagnose(rows)
            if note:
                print()
                print(note)
            break

    print("=" * 67)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
