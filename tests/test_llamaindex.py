"""Tests for the LlamaIndex LLM wrapper. No real API or ferrocache required."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from llama_index.core.llms import (  # noqa: E402
    ChatMessage,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)

from ferrocache.client import FerrocacheError  # noqa: E402
from ferrocache.llamaindex import FerrocacheLLM  # noqa: E402


FAKE_MODEL_ID = "fake::3"


def fake_embed(_: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def fake_inner_llm(complete_text: str = "from inner") -> MagicMock:
    inner = MagicMock()
    inner.metadata = LLMMetadata()
    inner.complete = MagicMock(return_value=CompletionResponse(text=complete_text))
    return inner


class LlamaIndexLLMTests(unittest.TestCase):
    @patch("ferrocache.llamaindex.FerrocacheClient")
    def test_complete_cache_hit(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {
            "hit": True,
            "response": "cached output",
            "similarity": 0.95,
        }
        inner = fake_inner_llm()
        llm = FerrocacheLLM(inner=inner, embed_fn=fake_embed, model_id=FAKE_MODEL_ID)
        resp = llm.complete("prompt")
        inner.complete.assert_not_called()
        self.assertEqual(resp.text, "cached output")
        self.assertTrue(resp.additional_kwargs.get("ferrocache_hit"))

    @patch("ferrocache.llamaindex.FerrocacheClient")
    def test_complete_cache_miss(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        inner = fake_inner_llm("real answer")
        llm = FerrocacheLLM(inner=inner, embed_fn=fake_embed, model_id=FAKE_MODEL_ID)
        resp = llm.complete("prompt")
        inner.complete.assert_called_once()
        self.assertEqual(resp.text, "real answer")
        # Result should have been inserted.
        MockCache.return_value.insert.assert_called_once()
        kwargs = MockCache.return_value.insert.call_args.kwargs
        self.assertEqual(kwargs["response"], "real answer")
        self.assertEqual(kwargs["query_text"], "prompt")

    @patch("ferrocache.llamaindex.FerrocacheClient")
    def test_fail_open(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.side_effect = FerrocacheError("down")
        inner = fake_inner_llm("survived")
        llm = FerrocacheLLM(
            inner=inner, embed_fn=fake_embed, model_id=FAKE_MODEL_ID, fail_open=True
        )
        resp = llm.complete("prompt")
        inner.complete.assert_called_once()
        self.assertEqual(resp.text, "survived")

    @patch("ferrocache.llamaindex.FerrocacheClient")
    def test_custom_embed_fn(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        captured: list[str] = []

        def my_embed(text: str) -> list[float]:
            captured.append(text)
            return [9.0, 8.0, 7.0]

        inner = fake_inner_llm()
        llm = FerrocacheLLM(inner=inner, embed_fn=my_embed, model_id="custom::3")
        llm.complete("the prompt")
        self.assertIn("the prompt", captured)
        # The first lookup call should use our custom embedding.
        kwargs = MockCache.return_value.query.call_args.kwargs
        self.assertEqual(kwargs["embedding"], [9.0, 8.0, 7.0])

    @patch("ferrocache.llamaindex.FerrocacheClient")
    def test_chat_cache_hit(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {
            "hit": True,
            "response": "cached chat",
            "similarity": 0.97,
        }
        inner = fake_inner_llm()
        inner.chat = MagicMock()
        llm = FerrocacheLLM(inner=inner, embed_fn=fake_embed, model_id=FAKE_MODEL_ID)
        resp = llm.chat([ChatMessage(role=MessageRole.USER, content="hi there")])
        inner.chat.assert_not_called()
        self.assertEqual(resp.message.content, "cached chat")


if __name__ == "__main__":
    unittest.main()
