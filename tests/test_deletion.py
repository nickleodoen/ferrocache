"""Python client tests for the M26 deletion + invalidation surface."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from ferrocache.client import FerrocacheClient  # noqa: E402


def _fake_response(payload: dict) -> MagicMock:
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, exc_type, exc, tb: None
    return resp


def _captured_request(urlopen_mock: MagicMock):
    args, _ = urlopen_mock.call_args
    return args[0]


class DeleteEntryTests(unittest.TestCase):
    def test_delete_entry_method(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"deleted": True})
            result = client.delete_entry("uuid-123")
        self.assertEqual(result, {"deleted": True})
        req = _captured_request(urlopen)
        self.assertEqual(req.method, "DELETE")
        self.assertTrue(req.full_url.endswith("/entry/uuid-123"))

    def test_delete_entry_includes_auth(self) -> None:
        client = FerrocacheClient(
            "http://localhost:3000", auth_token="s3cr3t"
        )
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"deleted": True})
            client.delete_entry("any")
        req = _captured_request(urlopen)
        # urllib lowercases header names internally.
        self.assertEqual(
            req.headers.get("Authorization"), "Bearer s3cr3t"
        )

    def test_delete_entry_rejects_empty_uuid(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with self.assertRaises(ValueError):
            client.delete_entry("")


class InsertTtlTests(unittest.TestCase):
    def test_insert_includes_ttl_when_set(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"id": "u", "status": "ok"})
            client.insert(
                embedding=[0.1, 0.2, 0.3],
                response="r",
                query_text="q",
                model_id="m::3",
                ttl_seconds=120,
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["ttl_seconds"], 120)

    def test_insert_omits_ttl_by_default(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"id": "u", "status": "ok"})
            client.insert(
                embedding=[0.1, 0.2, 0.3],
                response="r",
                query_text="q",
                model_id="m::3",
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("ttl_seconds", body)


class InvalidateTests(unittest.TestCase):
    def test_invalidate_posts_correct_payload(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response(
                {"invalidated_count": 3, "uuids": ["a", "b", "c"]}
            )
            result = client.invalidate(
                embedding=[1.0, 0.0, 0.0],
                threshold=0.9,
                model_id="m::3",
            )
        self.assertEqual(result["invalidated_count"], 3)
        req = _captured_request(urlopen)
        self.assertEqual(req.method, "POST")
        self.assertTrue(req.full_url.endswith("/admin/invalidate"))
        body = json.loads(req.data.decode("utf-8"))
        self.assertAlmostEqual(body["threshold"], 0.9)
        self.assertEqual(body["model_id"], "m::3")


if __name__ == "__main__":
    unittest.main()
