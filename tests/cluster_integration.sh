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

MODEL_ID="test-model::4"

echo ""
echo "=== Test 2: Insert on node1, query on all nodes ==="
EMBEDDING='[1.0, 0.0, 0.0, 0.0]'
INSERT_RESP=$(curl -s -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING, \"response\": \"test-response-1\", \"query_text\": \"test query\", \"model_id\": \"$MODEL_ID\"}")
UUID=$(echo "$INSERT_RESP" | jq -r '.id')
STATUS=$(echo "$INSERT_RESP" | jq -r '.status')
assert_eq "insert on node1 returns ok" "ok" "$STATUS"
echo "  UUID: $UUID"

for PORT in 3001 3002 3003; do
    QUERY_RESP=$(curl -s -X POST "http://localhost:$PORT/query" \
        -H "Content-Type: application/json" \
        -d "{\"embedding\": $EMBEDDING, \"threshold\": 0.90, \"model_id\": \"$MODEL_ID\"}")
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
    -d "{\"embedding\": $EMBEDDING2, \"response\": \"test-response-2\", \"query_text\": \"second query\", \"model_id\": \"$MODEL_ID\"}")
STATUS2=$(echo "$INSERT_RESP2" | jq -r '.status')
assert_eq "insert on node2 returns ok" "ok" "$STATUS2"

QUERY_RESP2=$(curl -s -X POST "$BASE3/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING2, \"threshold\": 0.90, \"model_id\": \"$MODEL_ID\"}")
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
    -d "{\"embedding\": $EMBEDDING3, \"response\": \"local-only\", \"query_text\": \"local test\", \"model_id\": \"$MODEL_ID\"}" > /dev/null

LOCAL_HIT=$(curl -s -X POST "$BASE1/query?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING3, \"threshold\": 0.90, \"model_id\": \"$MODEL_ID\"}" | jq -r '.hit')
assert_eq "local query on node1 hits" "true" "$LOCAL_HIT"

LOCAL_MISS=$(curl -s -X POST "$BASE2/query?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING3, \"threshold\": 0.90, \"model_id\": \"$MODEL_ID\"}" | jq -r '.hit')
assert_eq "local query on node2 misses (not replicated)" "false" "$LOCAL_MISS"

echo ""
echo "=== Test 6: Auth (sanity — health is always public) ==="
# /health and /metrics must never require auth. The cluster compose file
# leaves FERROCACHE_AUTH_TOKEN unset by default, so the data routes also
# accept unauthenticated traffic. Auth-on integration testing requires a
# separate compose run with the env var exported on every node.
HEALTH=$(curl -s "$BASE1/health" | jq -r '.status')
assert_eq "health always works without auth" "ok" "$HEALTH"
METRICS_CT=$(curl -s -o /dev/null -w "%{content_type}" "$BASE1/metrics")
assert_eq "metrics always works without auth" "text/plain; version=0.0.4; charset=utf-8" "$METRICS_CT"

echo ""
echo "=== Test 7: Node failure and ring reassignment (M22) ==="
# Insert via node1 — replication_factor=2 puts a copy on the ring's
# clockwise successor. After we kill node3 the cluster must keep serving
# without 502s. Phi-accrual + chitchat detection together take a while,
# so this test is timing-tolerant: the load-bearing assertion is
# "non-502", not "miss vs hit" or precise dead_nodes contents.
EMBEDDING_FAIL='[0.5, 0.5, 0.0, 0.0]'
INSERT_RESP_FAIL=$(curl -s -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING_FAIL, \"response\": \"failover-test\", \"query_text\": \"failover\", \"model_id\": \"test-model::4\"}")
STATUS_FAIL=$(echo "$INSERT_RESP_FAIL" | jq -r '.status')
assert_eq "insert before failover" "ok" "$STATUS_FAIL"

echo "  Stopping ferrocache-node3..."
docker stop ferrocache-node3 > /dev/null

# Failure detection latency = chitchat dead-time + phi-accrual rise.
# Default config can take 30+ seconds; we wait 35 to give it slack.
echo "  Waiting 35s for failure detection to fire..."
sleep 35

# /cluster/status must still respond (the cluster is degraded, not down).
STATUS_CODE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE1/cluster/status")
assert_eq "cluster/status still 200 after node3 dies" "200" "$STATUS_CODE_STATUS"

# The critical assertion: queries return 200 (miss-or-hit), never 502.
# A 502 would mean the cluster is still routing to the dead node and
# returning upstream-unavailable. After this many seconds, either
# chitchat or phi-accrual (or both) should have removed node3 from the
# ring or M21's soft skip should fire — both paths produce 200.
STATUS_CODE_QUERY=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE1/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING_FAIL, \"threshold\": 0.90, \"model_id\": \"test-model::4\"}")
assert_eq "query with dead node returns 200, not 502" "200" "$STATUS_CODE_QUERY"

# Inserts during degraded replication should still succeed (warn-logged,
# replication factor degrades silently).
STATUS_CODE_INSERT=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.6, 0.6, 0.0, 0.0], \"response\": \"degraded-write\", \"query_text\": \"degraded\", \"model_id\": \"test-model::4\"}")
assert_eq "insert during failure returns 200, not 502" "200" "$STATUS_CODE_INSERT"

echo "  Restarting ferrocache-node3..."
docker start ferrocache-node3 > /dev/null
echo "  Waiting 15s for re-join..."
sleep 15

# After restart, the cluster should stay healthy. Exact node count
# depends on gossip convergence timing, so just assert /cluster/status
# is still answering.
STATUS_CODE_REJOIN=$(curl -s -o /dev/null -w "%{http_code}" "$BASE1/cluster/status")
assert_eq "cluster/status 200 after node3 restarts" "200" "$STATUS_CODE_REJOIN"

echo ""
echo "=== Test 8: Read repair (M23) ==="
# Insert directly on node1 with ?local=true so node2/node3 don't learn
# about it via replication. Then query the same embedding via node2 — if
# node2 is the ring owner, the local miss triggers a fan-out to replicas;
# node1 (where the entry is) returns the hit. The behaviour is gated on
# ring layout and `read_repair_enabled = true` (default).
# Use a fresh model_id namespace for this test so prior tests' entries
# (which may share dims and partial directions) don't trigger false hits
# at the 0.90 threshold.
RR_MODEL="m23-rr::4"
EMBEDDING_RR='[0.1, 0.2, 0.7, 0.6]'
RR_INSERT=$(curl -s -X POST "$BASE1/insert?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING_RR, \"response\": \"repair-test\", \"query_text\": \"repair\", \"model_id\": \"$RR_MODEL\", \"uuid\": \"rr-target\"}" | jq -r '.status')
assert_eq "local-only insert on node1" "ok" "$RR_INSERT"

# Verify locality of the seed: node1 has it, node2 doesn't (yet).
LOCAL_HIT_N1=$(curl -s -X POST "$BASE1/query?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING_RR, \"threshold\": 0.90, \"model_id\": \"$RR_MODEL\"}" | jq -r '.hit')
assert_eq "node1 has the entry locally" "true" "$LOCAL_HIT_N1"

LOCAL_HIT_N2_BEFORE=$(curl -s -X POST "$BASE2/query?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING_RR, \"threshold\": 0.90, \"model_id\": \"$RR_MODEL\"}" | jq -r '.hit')
assert_eq "node2 does NOT have the entry locally yet" "false" "$LOCAL_HIT_N2_BEFORE"

# /internal/entry/{uuid} on node1 should return the full entry, on node2 a 404.
ENTRY_N1_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE1/internal/entry/rr-target?local=true")
assert_eq "/internal/entry on node1 returns 200" "200" "$ENTRY_N1_CODE"
ENTRY_N2_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE2/internal/entry/rr-target?local=true")
assert_eq "/internal/entry on node2 returns 404" "404" "$ENTRY_N2_CODE"

# Routed query through node2 — non-502 is the load-bearing assertion.
# Whether it's a hit (read repair found it on node1) or miss (replica
# walk skipped node1) depends on the ring; both produce 200.
ROUTED_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE2/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": $EMBEDDING_RR, \"threshold\": 0.90, \"model_id\": \"$RR_MODEL\"}")
assert_eq "routed query through node2 returns 200" "200" "$ROUTED_CODE"

# read_repair_enabled flag visible in /cluster/status.
RR_FLAG=$(curl -s "$BASE1/cluster/status" | jq -r '.read_repair_enabled')
assert_eq "/cluster/status reports read_repair_enabled" "true" "$RR_FLAG"

echo ""
echo "=== Test 9: Phase 7 feature integration (M24-M29) ==="

PHASE7_MODEL="m30::4"

# 1. Insert with TTL (3s deadline)
TTL_INSERT=$(curl -s -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.1, 0.2, 0.3, 0.4], \"response\": \"ttl-entry\", \"query_text\": \"ttl test\", \"model_id\": \"$PHASE7_MODEL\", \"ttl_seconds\": 3}")
TTL_STATUS=$(echo "$TTL_INSERT" | jq -r '.status')
TTL_UUID=$(echo "$TTL_INSERT" | jq -r '.id')
assert_eq "TTL insert returns ok" "ok" "$TTL_STATUS"
TTL_UUID_NONEMPTY="no"
if [ -n "$TTL_UUID" ] && [ "$TTL_UUID" != "null" ]; then TTL_UUID_NONEMPTY="yes"; fi
assert_eq "TTL insert returns non-empty UUID" "yes" "$TTL_UUID_NONEMPTY"

# 2. Insert with cache_scope=tenant_a
curl -s -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.5, 0.5, 0.5, 0.5], \"response\": \"tenant-a-answer\", \"query_text\": \"scoped question\", \"model_id\": \"$PHASE7_MODEL\", \"cache_scope\": \"tenant_a\"}" > /dev/null

# 3. Cross-tenant query must miss (different cache_scope)
SCOPE_MISS=$(curl -s -X POST "$BASE2/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.5, 0.5, 0.5, 0.5], \"threshold\": 0.90, \"model_id\": \"$PHASE7_MODEL\", \"cache_scope\": \"tenant_b\"}" \
    | jq -r '.hit')
assert_eq "different cache_scope -> miss" "false" "$SCOPE_MISS"

# 4. Same-tenant query must hit
SCOPE_HIT=$(curl -s -X POST "$BASE2/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.5, 0.5, 0.5, 0.5], \"threshold\": 0.90, \"model_id\": \"$PHASE7_MODEL\", \"cache_scope\": \"tenant_a\"}" \
    | jq -r '.hit')
assert_eq "same cache_scope -> hit" "true" "$SCOPE_HIT"

# 5. Insert with conversation_id
curl -s -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.9, 0.1, 0.0, 0.0], \"response\": \"conv-answer\", \"query_text\": \"conv question\", \"model_id\": \"$PHASE7_MODEL\", \"conversation_id\": \"conv_test_1\"}" > /dev/null

# 6. Two-level lookup: conversation hit (scope=conversation)
CONV_SCOPE=$(curl -s -X POST "$BASE2/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.9, 0.1, 0.0, 0.0], \"threshold\": 0.90, \"model_id\": \"$PHASE7_MODEL\", \"conversation_id\": \"conv_test_1\"}" \
    | jq -r '.scope')
assert_eq "conversation hit reports scope=conversation" "conversation" "$CONV_SCOPE"

# 7. Two-level lookup: global fallback (different conversation, hit base namespace via tenant scope)
GLOBAL_SCOPE=$(curl -s -X POST "$BASE2/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.5, 0.5, 0.5, 0.5], \"threshold\": 0.90, \"model_id\": \"$PHASE7_MODEL\", \"conversation_id\": \"conv_test_other\", \"cache_scope\": \"tenant_a\"}" \
    | jq -r '.scope')
assert_eq "fallback hit reports scope=global" "global" "$GLOBAL_SCOPE"

# 8. DELETE /entry/:uuid
DEL_UUID=$(curl -s -X POST "$BASE1/insert" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.0, 0.0, 0.0, 1.0], \"response\": \"delete-me\", \"query_text\": \"deletable\", \"model_id\": \"$PHASE7_MODEL\"}" \
    | jq -r '.id')
DEL_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE1/entry/$DEL_UUID")
assert_eq "DELETE /entry/:uuid returns 200" "200" "$DEL_STATUS"
DEL_MISS=$(curl -s -X POST "$BASE2/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.0, 0.0, 0.0, 1.0], \"threshold\": 0.90, \"model_id\": \"$PHASE7_MODEL\"}" \
    | jq -r '.hit')
assert_eq "deleted entry -> miss across cluster" "false" "$DEL_MISS"

# 9. Semantic invalidation. Use `?local=true` on insert so the entry is
#    deterministically on node1; otherwise ring routing might land it on
#    node2/node3 only and node1's local invalidate sweep would find 0.
curl -s -X POST "$BASE1/insert?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [1.0, 0.0, 0.0, 0.0], \"response\": \"inv-1\", \"query_text\": \"inv one\", \"model_id\": \"$PHASE7_MODEL\", \"uuid\": \"inv-uuid-1\"}" > /dev/null
INV_COUNT=$(curl -s -X POST "$BASE1/admin/invalidate?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [1.0, 0.0, 0.0, 0.0], \"threshold\": 0.95, \"model_id\": \"$PHASE7_MODEL\"}" \
    | jq -r '.invalidated_count')
assert_eq "invalidate found radius match" "1" "$INV_COUNT"

# 10. Exact-match pre-filter. Insert + query both `?local=true` on the same
#     node so the test doesn't depend on which replicas got the entry.
curl -s -X POST "$BASE1/insert?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.3, 0.3, 0.3, 0.3], \"response\": \"exact-answer\", \"query_text\": \"Exact Match Test\", \"model_id\": \"$PHASE7_MODEL\", \"uuid\": \"em-uuid-1\"}" > /dev/null
EM_FLAG=$(curl -s -X POST "$BASE1/query?local=true" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.0, 0.0, 0.0, 1.0], \"threshold\": 0.99, \"model_id\": \"$PHASE7_MODEL\", \"query_text\": \"exact match test\"}" \
    | jq -r '.exact_match')
assert_eq "exact match pre-filter fires (case/space-insensitive)" "true" "$EM_FLAG"

# 11. TTL expiry. Sleep past the 3s TTL; the inline expiry check on /query
#     filters expired neighbours regardless of reaper-tick interval. Use a
#     tight threshold (0.99) so the assertion isn't accidentally satisfied
#     by another entry from earlier in this test block.
echo "  Waiting 8s for TTL expiry..."
sleep 8
TTL_MISS=$(curl -s -X POST "$BASE2/query" \
    -H "Content-Type: application/json" \
    -d "{\"embedding\": [0.1, 0.2, 0.3, 0.4], \"threshold\": 0.99, \"model_id\": \"$PHASE7_MODEL\"}" \
    | jq -r '.hit')
assert_eq "TTL-expired entry -> miss" "false" "$TTL_MISS"

# 12. /admin/entry-stats responds with the namespace breakdown
ENTRY_STATS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE1/admin/entry-stats")
assert_eq "GET /admin/entry-stats returns 200" "200" "$ENTRY_STATS_CODE"

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
