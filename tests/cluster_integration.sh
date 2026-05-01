#!/usr/bin/env bash
# Integration test for the 3-node ferrocache cluster.
# Prereq: docker compose up -d --build (and converged); curl + jq installed.
set -euo pipefail

BASE1="http://localhost:3001"
BASE2="http://localhost:3002"
BASE3="http://localhost:3003"

PASS=0
FAIL=0

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc (expected=$expected, actual=$actual)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Waiting for cluster convergence ==="
CONVERGED=0
for i in $(seq 1 60); do
    COUNT=$(curl -s --max-time 2 "$BASE1/cluster/status" | jq -r '.node_count // 0' 2>/dev/null || echo 0)
    if [ "$COUNT" = "3" ]; then
        echo "  Cluster converged after ${i}s"
        CONVERGED=1
        break
    fi
    sleep 1
done
if [ "$CONVERGED" = "0" ]; then
    echo "  FAIL: cluster did not converge within 60s"
    curl -s "$BASE1/cluster/status" || true
    exit 1
fi

echo ""
echo "=== Test 1: Cluster status ==="
for PORT in 3001 3002 3003; do
    COUNT=$(curl -s "http://localhost:$PORT/cluster/status" | jq -r '.node_count')
    assert_eq "node on port $PORT sees 3 nodes" "3" "$COUNT"
done

echo ""
echo "=== Test 2: Insert on node1, query on all nodes ==="
EMBEDDING='[1.0, 0.0, 0.0, 0.0]'
INSERT_RESP=$(curl -s -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING, \"response\": \"test-response-1\", \"query_text\": \"test query\"}")
UUID=$(echo "$INSERT_RESP" | jq -r '.id')
STATUS=$(echo "$INSERT_RESP" | jq -r '.status')
assert_eq "insert on node1 returns ok" "ok" "$STATUS"
echo "  UUID: $UUID"

for PORT in 3001 3002 3003; do
    QUERY_RESP=$(curl -s -X POST "http://localhost:$PORT/query" \
        -H "Content-Type: application/json" \
        -d "{\"embedding\": $EMBEDDING, \"threshold\": 0.90}")
    HIT=$(echo "$QUERY_RESP" | jq -r '.hit')
    RESP=$(echo "$QUERY_RESP" | jq -r '.response // empty')
    assert_eq "query on port $PORT returns hit" "true" "$HIT"
    assert_eq "query on port $PORT returns correct response" "test-response-1" "$RESP"
done

echo ""
echo "=== Test 3: Insert on node2, query routes correctly from node3 ==="
EMBEDDING2='[0.0, 1.0, 0.0, 0.0]'
INSERT_RESP2=$(curl -s -X POST "$BASE2/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING2, \"response\": \"test-response-2\", \"query_text\": \"second query\"}")
STATUS2=$(echo "$INSERT_RESP2" | jq -r '.status')
assert_eq "insert on node2 returns ok" "ok" "$STATUS2"

QUERY_RESP2=$(curl -s -X POST "$BASE3/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING2, \"threshold\": 0.90}")
HIT2=$(echo "$QUERY_RESP2" | jq -r '.hit')
RESP2=$(echo "$QUERY_RESP2" | jq -r '.response // empty')
assert_eq "query on node3 for embedding2 returns hit" "true" "$HIT2"
assert_eq "query on node3 returns correct response" "test-response-2" "$RESP2"

echo ""
echo "=== Test 4: Health and stats ==="
for PORT in 3001 3002 3003; do
    HEALTH=$(curl -s "http://localhost:$PORT/health" | jq -r '.status')
    assert_eq "health on port $PORT" "ok" "$HEALTH"
done

echo ""
echo "=== Test 5: local=true scopes the operation ==="
EMBEDDING3='[0.0, 0.0, 1.0, 0.0]'
curl -s -X POST "$BASE1/insert?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING3, \"response\": \"local-only\", \"query_text\": \"local test\"}" > /dev/null

LOCAL_HIT=$(curl -s -X POST "$BASE1/query?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING3, \"threshold\": 0.90}" | jq -r '.hit')
assert_eq "local query on node1 hits" "true" "$LOCAL_HIT"

LOCAL_MISS=$(curl -s -X POST "$BASE2/query?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING3, \"threshold\": 0.90}" | jq -r '.hit')
assert_eq "local query on node2 misses (not replicated)" "false" "$LOCAL_MISS"

echo ""
echo "=== Results ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "SOME TESTS FAILED"
    exit 1
else
    echo "ALL TESTS PASSED"
    exit 0
fi
