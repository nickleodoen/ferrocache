"""Tests for the bearer-token auth path in the Python client + integrations."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from ferrocache.client import FerrocacheClient  # noqa: E402
from ferrocache.mcp_server import FerrocacheTools  # noqa: E402
from ferrocache.middleware import wrap_openai  # noqa: E402


def _fake_response(payload: dict) -> MagicMock:
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, exc_type, exc, tb: None
    return resp


def _captured_request(urlopen_mock: MagicMock):
    # urlopen(req, timeout=...) → first positional is the Request
    args, _ = urlopen_mock.call_args
    return args[0]


class ClientAuthTests(unittest.TestCase):
    def test_client_sends_auth_header(self) -> None:
        client = FerrocacheClient("http://localhost:3000", auth_token="test-token")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response(
                {"status": "ok", "node_id": "n", "entry_count": 0}
            )
            client.health()
        req = _captured_request(urlopen)
        # urllib lowercases header names internally.
        self.assertEqual(req.get_header("Authorization"), "Bearer test-token")

    def test_client_no_auth_header_without_token(self) -> None:
        # Force-clear env so the constructor falls through to None.
        with patch.dict(os.environ, {}, clear=True):
            client = FerrocacheClient("http://localhost:3000")
            with patch("ferrocache.client.request.urlopen") as urlopen:
                urlopen.return_value = _fake_response(
                    {"status": "ok", "node_id": "n", "entry_count": 0}
                )
                client.health()
            req = _captured_request(urlopen)
            self.assertIsNone(req.get_header("Authorization"))

    def test_client_reads_env_token(self) -> None:
        with patch.dict(os.environ, {"FERROCACHE_AUTH_TOKEN": "env-token"}, clear=False):
            client = FerrocacheClient("http://localhost:3000")
            with patch("ferrocache.client.request.urlopen") as urlopen:
                urlopen.return_value = _fake_response({"status": "ok"})
                client.health()
        req = _captured_request(urlopen)
        self.assertEqual(req.get_header("Authorization"), "Bearer env-token")

    def test_client_post_includes_auth_header(self) -> None:
        client = FerrocacheClient("http://localhost:3000", auth_token="t")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"hit": False})
            client.query(embedding=[0.1, 0.2], threshold=0.9, model_id="m::2")
        req = _captured_request(urlopen)
        self.assertEqual(req.get_header("Authorization"), "Bearer t")
        self.assertEqual(req.get_header("Content-type"), "application/json")


class MiddlewareAuthTests(unittest.TestCase):
    @patch("ferrocache.middleware.FerrocacheClient")
    def test_middleware_passes_auth_to_client(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        wrap_openai(
            MagicMock(),
            embed_fn=lambda _t: [0.1, 0.2, 0.3],
            model_id="m::3",
            cache_url="http://x",
            auth_token="middle-token",
        )
        # FerrocacheClient was constructed with the explicit auth_token.
        _, kwargs = MockCache.call_args
        self.assertEqual(kwargs.get("auth_token"), "middle-token")


class McpAuthTests(unittest.TestCase):
    def test_mcp_passes_auth_to_client(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        # No auth_token initially.
        self.assertIsNone(client.auth_token)
        FerrocacheTools(
            client=client,
            embed_fn=lambda _t: [0.1, 0.2, 0.3],
            auth_token="mcp-token",
        )
        # Tools constructor sets the token on the underlying client when it
        # was unset, so subsequent HTTP calls carry the bearer header.
        self.assertEqual(client.auth_token, "mcp-token")


if __name__ == "__main__":
    unittest.main()
