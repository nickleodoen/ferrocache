"""Example: add semantic caching to an OpenAI script with one line.

Before:
    from openai import OpenAI
    client = OpenAI()

After:
    from openai import OpenAI
    from ferrocache.middleware import wrap_openai
    client = wrap_openai(OpenAI())

Requires: a running ferrocache (cargo run --release), pip install openai
sentence-transformers, and OPENAI_API_KEY in the environment.
"""

from openai import OpenAI

from ferrocache.middleware import wrap_openai

client = wrap_openai(OpenAI(), cache_url="http://localhost:3000", threshold=0.92)

# First call: cache miss → calls OpenAI, stores the response.
r1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(f"r1 hit={r1._ferrocache_hit}  text={r1.choices[0].message.content!r}")

# Second call: semantically similar → cache hit, no API call.
r2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me the capital city of France"}],
)
print(
    f"r2 hit={r2._ferrocache_hit}  "
    f"sim={getattr(r2, '_ferrocache_similarity', None)}  "
    f"text={r2.choices[0].message.content!r}"
)
