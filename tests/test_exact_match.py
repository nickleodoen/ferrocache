"""Python client tests for the M27 exact-match pre-filter pass-through."""

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


class QueryTextTests(unittest.TestCase):
    def test_query_passes_query_text(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"hit": False})
            client.query(
                embedding=[1.0, 0.0, 0.0],
                threshold=0.9,
                model_id="m::3",
                query_text="What is X?",
            )
        req = _captured_request(urlopen)
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["query_text"], "What is X?")

    def test_query_omits_query_text_by_default(self) -> None:
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
        self.assertNotIn("query_text", body)


class MiddlewarePassThroughTests(unittest.TestCase):
    """The OpenAI middleware must forward the user prompt as `query_text`
    so the server can short-circuit via the exact-match pre-filter."""

    def test_middleware_passes_query_text(self) -> None:
        # Drive the OpenAI wrapper class directly so we can inject a mock
        # FerrocacheClient. The public `wrap_openai()` constructs its own
        # client from env, which is hard to mock.
        from ferrocache.middleware import WrappedOpenAIClient

        mock_resp = MagicMock()
        mock_resp.choices = []
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
        )
        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello world"}],
        )
        mock_cache.query.assert_called_once()
        call_kwargs = mock_cache.query.call_args.kwargs
        self.assertEqual(call_kwargs.get("query_text"), "Hello world")


if __name__ == "__main__":
    unittest.main()
