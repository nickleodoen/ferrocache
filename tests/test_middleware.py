"""Tests for the SDK middleware. No real API or ferrocache required."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from ferrocache.client import FerrocacheError  # noqa: E402
from ferrocache.middleware import wrap_anthropic, wrap_openai  # noqa: E402


def fake_embed(_: str) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


def fake_openai_response(text: str = "from openai") -> SimpleNamespace:
    return SimpleNamespace(
        id="real-id",
        object="chat.completion",
        created=0,
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def fake_anthropic_response(text: str = "from anthropic") -> SimpleNamespace:
    return SimpleNamespace(
        id="real-id",
        type="message",
        role="assistant",
        model="claude-haiku-4-5",
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=5, output_tokens=5),
    )


def fake_openai_client(create_return: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=create_return)
    client.models.list = MagicMock(return_value=["model-a", "model-b"])
    return client


def fake_anthropic_client(create_return: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.messages.create = MagicMock(return_value=create_return)
    return client


# ---------------------------------------------------------------------------
# OpenAI tests
# ---------------------------------------------------------------------------


class OpenAITests(unittest.TestCase):
    @patch("ferrocache.middleware.FerrocacheClient")
    def test_openai_cache_hit(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {
            "hit": True,
            "id": "u-1",
            "response": "cached answer",
            "similarity": 0.97,
        }
        real = fake_openai_client(fake_openai_response())
        wrapped = wrap_openai(real, embed_fn=fake_embed, cache_url="http://x")

        resp = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )

        real.chat.completions.create.assert_not_called()
        self.assertTrue(resp._ferrocache_hit)
        self.assertAlmostEqual(resp._ferrocache_similarity, 0.97)
        self.assertEqual(resp.choices[0].message.content, "cached answer")
        self.assertEqual(resp.choices[0].message.role, "assistant")

    @patch("ferrocache.middleware.FerrocacheClient")
    def test_openai_cache_miss(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        MockCache.return_value.insert.return_value = {"id": "new", "status": "ok"}
        real = fake_openai_client(fake_openai_response("real-api-output"))
        wrapped = wrap_openai(real, embed_fn=fake_embed)

        resp = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "what is x?"}],
        )

        real.chat.completions.create.assert_called_once()
        MockCache.return_value.insert.assert_called_once()
        insert_kwargs = MockCache.return_value.insert.call_args.kwargs
        self.assertEqual(insert_kwargs["embedding"], fake_embed(""))
        self.assertEqual(insert_kwargs["response"], "real-api-output")
        self.assertEqual(insert_kwargs["query_text"], "what is x?")
        self.assertEqual(resp._ferrocache_hit, False)
        self.assertEqual(resp.choices[0].message.content, "real-api-output")

    @patch("ferrocache.middleware.FerrocacheClient")
    def test_openai_fail_open(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.side_effect = FerrocacheError("connection refused")
        real = fake_openai_client(fake_openai_response("survived"))
        wrapped = wrap_openai(real, embed_fn=fake_embed, fail_open=True)

        resp = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )

        real.chat.completions.create.assert_called_once()
        self.assertIsNone(resp._ferrocache_hit)
        self.assertEqual(resp.choices[0].message.content, "survived")

    @patch("ferrocache.middleware.FerrocacheClient")
    def test_openai_passthrough(self, MockCache: MagicMock) -> None:
        real = fake_openai_client(fake_openai_response())
        wrapped = wrap_openai(real, embed_fn=fake_embed)

        result = wrapped.models.list()
        self.assertEqual(result, ["model-a", "model-b"])
        real.models.list.assert_called_once()


# ---------------------------------------------------------------------------
# Anthropic tests
# ---------------------------------------------------------------------------


class AnthropicTests(unittest.TestCase):
    @patch("ferrocache.middleware.FerrocacheClient")
    def test_anthropic_cache_hit(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {
            "hit": True,
            "id": "u-1",
            "response": "cached anthropic answer",
            "similarity": 0.94,
        }
        real = fake_anthropic_client(fake_anthropic_response())
        wrapped = wrap_anthropic(real, embed_fn=fake_embed)

        resp = wrapped.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "ping"}],
        )

        real.messages.create.assert_not_called()
        self.assertTrue(resp._ferrocache_hit)
        self.assertEqual(resp.content[0].text, "cached anthropic answer")
        self.assertEqual(resp.content[0].type, "text")
        self.assertEqual(resp.role, "assistant")

    @patch("ferrocache.middleware.FerrocacheClient")
    def test_anthropic_cache_miss(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        real = fake_anthropic_client(fake_anthropic_response("anthropic real"))
        wrapped = wrap_anthropic(real, embed_fn=fake_embed)

        resp = wrapped.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "the prompt"}],
        )

        real.messages.create.assert_called_once()
        MockCache.return_value.insert.assert_called_once()
        insert_kwargs = MockCache.return_value.insert.call_args.kwargs
        self.assertEqual(insert_kwargs["response"], "anthropic real")
        self.assertEqual(insert_kwargs["query_text"], "the prompt")
        self.assertEqual(resp._ferrocache_hit, False)

    @patch("ferrocache.middleware.FerrocacheClient")
    def test_anthropic_fail_open(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.side_effect = FerrocacheError("boom")
        real = fake_anthropic_client(fake_anthropic_response("served"))
        wrapped = wrap_anthropic(real, embed_fn=fake_embed)

        resp = wrapped.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
        )

        real.messages.create.assert_called_once()
        self.assertIsNone(resp._ferrocache_hit)
        self.assertEqual(resp.content[0].text, "served")


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    @patch("ferrocache.middleware.FerrocacheClient")
    def test_env_var_config(self, MockCache: MagicMock) -> None:
        with patch.dict(
            os.environ,
            {"FERROCACHE_URL": "http://env-host:9000", "FERROCACHE_THRESHOLD": "0.55"},
            clear=False,
        ):
            real = fake_openai_client(fake_openai_response())
            wrap_openai(real, embed_fn=fake_embed)
        MockCache.assert_called_with("http://env-host:9000")
        # threshold flows into the wrapper; verify by triggering a query
        MockCache.return_value.query.return_value = {"hit": False}

        with patch.dict(
            os.environ,
            {"FERROCACHE_URL": "http://env-host:9000", "FERROCACHE_THRESHOLD": "0.55"},
            clear=False,
        ):
            real = fake_openai_client(fake_openai_response())
            wrapped = wrap_openai(real, embed_fn=fake_embed)
            wrapped.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "p"}]
            )
        query_kwargs = MockCache.return_value.query.call_args.kwargs
        self.assertAlmostEqual(query_kwargs["threshold"], 0.55)

    @patch("ferrocache.middleware.FerrocacheClient")
    def test_explicit_kwargs_override_env(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        with patch.dict(
            os.environ,
            {"FERROCACHE_URL": "http://env:9000", "FERROCACHE_THRESHOLD": "0.55"},
            clear=False,
        ):
            real = fake_openai_client(fake_openai_response())
            wrapped = wrap_openai(
                real,
                cache_url="http://explicit:1234",
                threshold=0.88,
                embed_fn=fake_embed,
            )
            wrapped.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "p"}]
            )

        MockCache.assert_called_with("http://explicit:1234")
        query_kwargs = MockCache.return_value.query.call_args.kwargs
        self.assertAlmostEqual(query_kwargs["threshold"], 0.88)

    @patch("ferrocache.middleware.FerrocacheClient")
    def test_custom_embed_fn(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        captured: list[str] = []

        def my_embed(text: str) -> list[float]:
            captured.append(text)
            return [9.0, 8.0, 7.0]

        real = fake_openai_client(fake_openai_response())
        wrapped = wrap_openai(real, embed_fn=my_embed)
        wrapped.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "the prompt"}]
        )

        self.assertEqual(captured, ["the prompt"])
        query_kwargs = MockCache.return_value.query.call_args.kwargs
        self.assertEqual(query_kwargs["embedding"], [9.0, 8.0, 7.0])


if __name__ == "__main__":
    unittest.main()
