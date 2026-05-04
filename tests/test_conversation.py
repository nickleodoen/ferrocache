"""Python client tests for the M29 conversation_id pass-through."""

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


class ConversationClientTests(unittest.TestCase):
    def test_insert_passes_conversation_id(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"id": "u", "status": "ok"})
            client.insert(
                embedding=[1.0, 0.0, 0.0],
                response="r",
                query_text="q",
                model_id="m::3",
                conversation_id="conv_1",
            )
        body = json.loads(_captured_request(urlopen).data.decode("utf-8"))
        self.assertEqual(body["conversation_id"], "conv_1")

    def test_query_passes_conversation_id(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"hit": False})
            client.query(
                embedding=[1.0, 0.0, 0.0],
                threshold=0.9,
                model_id="m::3",
                conversation_id="conv_1",
            )
        body = json.loads(_captured_request(urlopen).data.decode("utf-8"))
        self.assertEqual(body["conversation_id"], "conv_1")

    def test_insert_omits_conversation_id_when_none(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"id": "u", "status": "ok"})
            client.insert(
                embedding=[1.0, 0.0, 0.0],
                response="r",
                query_text="q",
                model_id="m::3",
            )
        body = json.loads(_captured_request(urlopen).data.decode("utf-8"))
        self.assertNotIn("conversation_id", body)

    def test_query_omits_conversation_id_when_none(self) -> None:
        client = FerrocacheClient("http://localhost:3000")
        with patch("ferrocache.client.request.urlopen") as urlopen:
            urlopen.return_value = _fake_response({"hit": False})
            client.query(
                embedding=[1.0, 0.0, 0.0],
                threshold=0.9,
                model_id="m::3",
            )
        body = json.loads(_captured_request(urlopen).data.decode("utf-8"))
        self.assertNotIn("conversation_id", body)


class ConversationMiddlewareTests(unittest.TestCase):
    def test_middleware_forwards_conversation_id(self) -> None:
        from types import SimpleNamespace

        from ferrocache.middleware import WrappedOpenAIClient

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
            conversation_id="conv_xyz",
        )
        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        self.assertEqual(
            mock_cache.query.call_args.kwargs.get("conversation_id"),
            "conv_xyz",
        )
        self.assertEqual(
            mock_cache.insert.call_args.kwargs.get("conversation_id"),
            "conv_xyz",
        )


if __name__ == "__main__":
    unittest.main()
