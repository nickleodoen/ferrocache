"""ferrocache realistic simulation harness.

Generates an FAQ-style workload with semantic variations, runs it against a
live ferrocache instance, and prints latency + hit-rate metrics. Embeddings
are computed locally with sentence-transformers — no API keys.

Run:
    cargo run --release         # in another terminal
    pip install -r tests/requirements.txt
    python3 tests/simulate.py

For latency-only benchmarking without PyTorch, see simulate_no_ml.py.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from ferrocache import FerrocacheClient, FerrocacheError  # noqa: E402

SEED_QUESTIONS: list[tuple[str, str]] = [
    (
        "What is the company vacation policy?",
        "Employees receive 20 days of PTO per year, accruing monthly.",
    ),
    (
        "How do I reset my password?",
        "Go to Settings > Security > Reset Password, or contact IT at help@company.com.",
    ),
    (
        "What are the office hours?",
        "The office is open Monday-Friday, 8am to 6pm. Core hours are 10am-4pm.",
    ),
    (
        "How do I submit an expense report?",
        "Use the Concur app. Submit within 30 days of the expense with receipts attached.",
    ),
    (
        "What is the parental leave policy?",
        "12 weeks paid for primary caregivers, 6 weeks for secondary caregivers.",
    ),
    (
        "Where can I find the employee handbook?",
        "Available on the company intranet under HR > Documents > Employee Handbook.",
    ),
    (
        "How do I request time off?",
        "Submit a request through Workday at least 2 weeks in advance for planned PTO.",
    ),
    (
        "What health insurance plans are available?",
        "Three tiers: Basic ($0/mo), Standard ($150/mo), and Premium ($300/mo).",
    ),
    (
        "How do I set up direct deposit?",
        "Go to Workday > Pay > Payment Elections and add your bank routing and account numbers.",
    ),
    (
        "What is the remote work policy?",
        "Hybrid model: 3 days in-office (Tue-Thu), 2 days remote (Mon, Fri).",
    ),
    (
        "How do I enroll in the 401k?",
        "Enroll through Fidelity NetBenefits. Company matches 50% up to 6% of salary.",
    ),
    (
        "What is the dress code?",
        "Business casual Monday-Thursday, casual Friday. No open-toed shoes in the lab.",
    ),
    (
        "How do I book a conference room?",
        "Use Google Calendar. Rooms are prefixed with the floor number (e.g., 3-Sequoia).",
    ),
    (
        "What is the guest WiFi password?",
        "Network: CompanyGuest, Password: Welcome2024. Changes quarterly.",
    ),
    (
        "How do I order office supplies?",
        "Submit a request through the Facilities portal. Standard items ship within 2 business days.",
    ),
]

UNRELATED_QUERIES: list[str] = [
    "What's the weather forecast for tomorrow?",
    "Recipe for chocolate chip cookies",
    "Best hiking trails near Yosemite",
    "Who won the 2022 World Cup?",
    "How does photosynthesis work?",
    "Translate hello to Japanese",
    "What is the capital of Mongolia?",
    "Explain quantum entanglement briefly",
    "Latest stock price for TSLA",
    "How tall is Mount Everest in feet?",
]


def generate_variations(question: str) -> list[str]:
    base = question.rstrip("?").lower()
    variations: list[str] = [
        f"Can you tell me {base}?",
        f"I need to know {base}",
        f"Please explain {base}",
    ]
    if "how" in base:
        variations.append(base.replace("how do i", "what's the process to"))
    if "what" in base:
        variations.append(base.replace("what is", "tell me about"))
    return variations[:3]


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
    return f"{ms:.1f}ms"


def load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: sentence-transformers is not installed.\n"
            "  pip install -r tests/requirements.txt\n"
            "  (or use simulate_no_ml.py for latency-only testing)",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Loading model {model_name} (downloads on first run)...")
    return SentenceTransformer(model_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="ferrocache realistic simulation")
    parser.add_argument("--url", default="http://localhost:3000")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    client = FerrocacheClient(args.url)
    try:
        health = client.health()
    except FerrocacheError as e:
        print(f"ERROR: ferrocache not reachable at {args.url}: {e}", file=sys.stderr)
        return 1

    model = load_model(args.model)

    def embed(texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in model.encode(texts, show_progress_bar=False)]

    insert_latencies: list[float] = []
    hit_query_latencies: list[float] = []
    miss_query_latencies: list[float] = []
    embed_latencies: list[float] = []

    # Phase 1 — populate
    print("\nPhase 1: populating cache...")
    seed_texts = [q for q, _ in SEED_QUESTIONS]
    t0 = time.perf_counter()
    seed_vecs = embed(seed_texts)
    embed_latencies.append((time.perf_counter() - t0) * 1000 / len(seed_texts))

    for (question, answer), vec in zip(SEED_QUESTIONS, seed_vecs, strict=True):
        t0 = time.perf_counter()
        client.insert(embedding=vec, response=answer, query_text=question)
        insert_latencies.append((time.perf_counter() - t0) * 1000)

    # Phase 2 — expected-hit queries (variations of seeds)
    print("Phase 2: querying with semantic variations (expected hits)...")
    expected_hit_pairs: list[tuple[str, str]] = []
    for question, answer in SEED_QUESTIONS:
        for v in generate_variations(question):
            expected_hit_pairs.append((v, answer))

    hit_count = 0
    false_misses = 0
    for variation, expected_answer in expected_hit_pairs:
        t0 = time.perf_counter()
        vec = embed([variation])[0]
        embed_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        result = client.query(embedding=vec, threshold=args.threshold)
        hit_query_latencies.append((time.perf_counter() - t0) * 1000)

        if result.get("hit"):
            hit_count += 1
        else:
            false_misses += 1

    # Phase 3 — expected-miss queries (unrelated)
    print("Phase 3: querying unrelated topics (expected misses)...")
    true_misses = 0
    false_hits = 0
    for q in UNRELATED_QUERIES:
        t0 = time.perf_counter()
        vec = embed([q])[0]
        embed_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        result = client.query(embedding=vec, threshold=args.threshold)
        miss_query_latencies.append((time.perf_counter() - t0) * 1000)

        if result.get("hit"):
            false_hits += 1
        else:
            true_misses += 1

    # Phase 4 — repeated queries on a warm cache (latency only)
    print("Phase 4: repeated queries (warm-cache latency)...")
    sample = random.Random(42).sample(expected_hit_pairs, min(15, len(expected_hit_pairs)))
    for variation, _ in sample:
        vec = embed([variation])[0]
        t0 = time.perf_counter()
        client.query(embedding=vec, threshold=args.threshold)
        hit_query_latencies.append((time.perf_counter() - t0) * 1000)

    health = client.health()
    total_hit_q = len(expected_hit_pairs)
    hit_rate = (hit_count / total_hit_q * 100) if total_hit_q else 0.0

    line = "═" * 63
    print()
    print("ferrocache simulation results")
    print(line)
    print(f"Model:           {args.model} (384-dim)")
    print(f"Target:          {args.url}")
    print(f"Threshold:       {args.threshold}")
    print()
    print("Workload")
    print(f"  Seed questions:     {len(SEED_QUESTIONS)}")
    print(f"  Variations/seed:    3")
    print(f"  Total inserts:      {len(insert_latencies)}")
    print(
        f"  Total queries:      {len(hit_query_latencies) + len(miss_query_latencies)} "
        f"({total_hit_q} expected-hit + {len(UNRELATED_QUERIES)} expected-miss)"
    )
    print()
    print("Cache Performance")
    print(f"  Hit rate:           {hit_rate:.1f}% ({hit_count}/{total_hit_q} expected-hit queries matched)")
    print(f"  False misses:       {false_misses} (variations too different at threshold {args.threshold})")
    print(f"  True misses:        {true_misses}/{len(UNRELATED_QUERIES)} (unrelated queries correctly missed)")
    if false_hits:
        print(f"  False hits:         {false_hits} (unrelated queries that matched — threshold too low)")
    print()
    print("Latency (ferrocache only, excludes embedding time)")
    print(
        f"  Insert:             p50={fmt_ms(percentile(insert_latencies, 0.5))}  "
        f"p99={fmt_ms(percentile(insert_latencies, 0.99))}   "
        f"mean={fmt_ms(statistics.mean(insert_latencies))}"
    )
    print(
        f"  Query (hit):        p50={fmt_ms(percentile(hit_query_latencies, 0.5))}  "
        f"p99={fmt_ms(percentile(hit_query_latencies, 0.99))}   "
        f"mean={fmt_ms(statistics.mean(hit_query_latencies))}"
    )
    print(
        f"  Query (miss):       p50={fmt_ms(percentile(miss_query_latencies, 0.5))}  "
        f"p99={fmt_ms(percentile(miss_query_latencies, 0.99))}   "
        f"mean={fmt_ms(statistics.mean(miss_query_latencies))}"
    )
    print()
    print("Embedding (local model, for reference)")
    print(
        f"  Embed time/query:   p50={fmt_ms(percentile(embed_latencies, 0.5))}   "
        f"p99={fmt_ms(percentile(embed_latencies, 0.99))}    "
        f"mean={fmt_ms(statistics.mean(embed_latencies))}"
    )
    print()
    print("Cache State")
    print(f"  Entry count:        {health.get('entry_count')}")
    print(f"  Node:               {health.get('node_id')}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
