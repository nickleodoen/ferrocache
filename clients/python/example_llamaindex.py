"""Example: ferrocache wrapping a LlamaIndex LLM.

Requires:
    pip install llama-index-core llama-index-llms-openai sentence-transformers
    cargo run --release        # in another terminal
    OPENAI_API_KEY=sk-...

Usage:
    python3 clients/python/example_llamaindex.py

`FerrocacheLLM` subclasses `CustomLLM`, so it satisfies the LlamaIndex `LLM`
interface and can be used anywhere a LlamaIndex LLM is expected (queries,
agents, query engines, etc.).
"""

from llama_index.llms.openai import OpenAI

from ferrocache.llamaindex import FerrocacheLLM

llm = FerrocacheLLM(inner=OpenAI(model="gpt-4o-mini"))

r1 = llm.complete("What is the capital of France?")
print("First call:", r1.text)

r2 = llm.complete("Tell me the capital of France")
print("Second call:", r2.text)
print("Cache hit:", r2.additional_kwargs.get("ferrocache_hit"))
