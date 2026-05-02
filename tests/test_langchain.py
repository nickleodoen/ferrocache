"""Tests for the LangChain BaseCache backend. No real API or ferrocache required."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "clients" / "python"))

from langchain_core.outputs import Generation  # noqa: E402

from ferrocache.client import FerrocacheError  # noqa: E402
from ferrocache.langchain import FerrocacheCache  # noqa: E402


FAKE_MODEL_ID = "fake::3"


def fake_embed(_: str) -> list[float]:
    return [0.1, 0.2, 0.3]


class LangchainCacheTests(unittest.TestCase):
    @patch("ferrocache.langchain.FerrocacheClient")
    def test_lookup_hit(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {
            "hit": True,
            "response": "cached answer",
            "similarity": 0.95,
        }
        cache = FerrocacheCache(embed_fn=fake_embed, model_id=FAKE_MODEL_ID, cache_url="http://x")
        result = cache.lookup("the prompt", "gpt-4o-mini")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "cached answer")

    @patch("ferrocache.langchain.FerrocacheClient")
    def test_lookup_miss(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        cache = FerrocacheCache(embed_fn=fake_embed, model_id=FAKE_MODEL_ID)
        self.assertIsNone(cache.lookup("the prompt", "gpt-4"))

    @patch("ferrocache.langchain.FerrocacheClient")
    def test_update_inserts(self, MockCache: MagicMock) -> None:
        cache = FerrocacheCache(embed_fn=fake_embed, model_id=FAKE_MODEL_ID)
        cache.update("the prompt", "gpt-4", [Generation(text="cached output")])
        MockCache.return_value.insert.assert_called_once()
        kwargs = MockCache.return_value.insert.call_args.kwargs
        self.assertEqual(kwargs["embedding"], fake_embed(""))
        self.assertEqual(kwargs["response"], "cached output")
        self.assertEqual(kwargs["query_text"], "the prompt")

    @patch("ferrocache.langchain.FerrocacheClient")
    def test_lookup_fail_open(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.side_effect = FerrocacheError("down")
        cache = FerrocacheCache(embed_fn=fake_embed, model_id=FAKE_MODEL_ID, fail_open=True)
        self.assertIsNone(cache.lookup("p", "gpt"))

    @patch("ferrocache.langchain.FerrocacheClient")
    def test_update_fail_open(self, MockCache: MagicMock) -> None:
        MockCache.return_value.insert.side_effect = FerrocacheError("down")
        cache = FerrocacheCache(embed_fn=fake_embed, model_id=FAKE_MODEL_ID, fail_open=True)
        # Should swallow the error, not raise.
        cache.update("p", "gpt", [Generation(text="x")])

    @patch("ferrocache.langchain.FerrocacheClient")
    def test_clear_is_noop(self, MockCache: MagicMock) -> None:
        cache = FerrocacheCache(embed_fn=fake_embed, model_id=FAKE_MODEL_ID)
        cache.clear()  # must not raise

    @patch("ferrocache.langchain.FerrocacheClient")
    def test_custom_embed_fn(self, MockCache: MagicMock) -> None:
        MockCache.return_value.query.return_value = {"hit": False}
        captured: list[str] = []

        def my_embed(text: str) -> list[float]:
            captured.append(text)
            return [9.0, 8.0, 7.0]

        cache = FerrocacheCache(embed_fn=my_embed, model_id="custom::3")
        cache.lookup("hello", "gpt")
        self.assertEqual(captured, ["hello"])
        kwargs = MockCache.return_value.query.call_args.kwargs
        self.assertEqual(kwargs["embedding"], [9.0, 8.0, 7.0])


if __name__ == "__main__":
    unittest.main()
