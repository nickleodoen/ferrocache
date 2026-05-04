"""Python client tests for the M28 cache_scope pass-through."""

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


class CacheScopeClientTests(unittest.TestCase):
    def test_insert_passes_cache_scope(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"id": "u", "status": "ok"})
            client.insert(
                embedding=[1.0, 0.0, 0.0],
                response="r",
                query_text="q",
                model_id="m::3",
                cache_scope="tenant_a",
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["cache_scope"], "tenant_a")

    def test_query_passes_cache_scope(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"hit": False})
            client.query(
                embedding=[1.0, 0.0, 0.0],
                threshold=0.9,
                model_id="m::3",
                cache_scope="tenant_a",
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["cache_scope"], "tenant_a")

    def test_insert_omits_scope_when_none(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"id": "u", "status": "ok"})
            client.insert(
                embedding=[1.0, 0.0, 0.0],
                response="r",
                query_text="q",
                model_id="m::3",
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("cache_scope", body)

    def test_query_omits_scope_when_none(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"hit": False})
            client.query(
                embedding=[1.0, 0.0, 0.0],
                threshold=0.9,
                model_id="m::3",
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("cache_scope", body)

    def test_invalidate_passes_cache_scope(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response(
                {"invalidated_count": 0, "uuids": []}
            )
            client.invalidate(
                embedding=[1.0, 0.0, 0.0],
                threshold=0.95,
                model_id="m::3",
                cache_scope="tenant_a",
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["cache_scope"], "tenant_a")


class MiddlewareScopeTests(unittest.TestCase):
    """The OpenAI middleware must forward `cache_scope` to every cache call
    when the wrapper was constructed with one."""

    def test_middleware_passes_cache_scope_on_query_and_insert(self) -> None:
        from ferrocache.middleware import WrappedOpenAIClient

        # Build a real OpenAI response object the middleware can extract from.
        from types import SimpleNamespace

        msg = SimpleNamespace(role="assistant", content="answer")
        choice = SimpleNamespace(index=0, message=msg, finish_reason="stop")
        mock_resp = SimpleNamespace(choices=[choice])

        real_client = MagicMock()
        real_client.chat.completions.create.return_value = mock_resp

        mock_cache = MagicMock()
        mock_cache.query.return_value = {"hit": False}

        wrapped = WrappedOpenAIClient(
            real_client,
            cache=mock_cache,
            embed_fn=lambda text: [0.1, 0.2, 0.3],
            model_id="m::3",
            threshold=0.9,
            fail_open=True,
            cache_scope="tenant_a",
        )
        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        mock_cache.query.assert_called_once()
        self.assertEqual(
            mock_cache.query.call_args.kwargs.get("cache_scope"),
            "tenant_a",
        )
        # Cache miss → middleware should also store with the same scope.
        mock_cache.insert.assert_called_once()
        self.assertEqual(
            mock_cache.insert.call_args.kwargs.get("cache_scope"),
            "tenant_a",
        )


if __name__ == "__main__":
    unittest.main()
