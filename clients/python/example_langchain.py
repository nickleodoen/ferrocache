"""Example: ferrocache as a LangChain cache backend.

Requires:
    pip install langchain langchain-openai sentence-transformers
    cargo run --release        # in another terminal
    OPENAI_API_KEY=sk-...

Usage:
    python3 clients/python/example_langchain.py
"""

from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from ferrocache.langchain import FerrocacheCache

# One-line install: every LLM call now consults ferrocache first.
set_llm_cache(FerrocacheCache())

llm = ChatOpenAI(model="gpt-4o-mini")

r1 = llm.invoke("What is the capital of France?")
print("First call:", r1.content)

# Semantically similar — should hit the cache and skip the API.
r2 = llm.invoke("Tell me the capital of France")
print("Second call:", r2.content)
