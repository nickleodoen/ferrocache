"""Example usage of the ferrocache Python client.

Prereq: a ferrocache node running on http://localhost:3000.
    cargo run --release
"""

from ferrocache import FerrocacheClient


def main() -> None:
    client = FerrocacheClient("http://localhost:3000")

    print("Health:", client.health())

    model_id = "demo::4"
    embeddings = [
        ([1.0, 0.0, 0.0, 0.0], "The vacation policy allows 20 days PTO per year."),
        ([0.0, 1.0, 0.0, 0.0], "The company was founded in 2015."),
        ([0.0, 0.0, 1.0, 0.0], "The engineering team uses Rust and Python."),
    ]
    for emb, response in embeddings:
        result = client.insert(
            embedding=emb, response=response, query_text="seed", model_id=model_id
        )
        print(f"Inserted id={result['id']}")

    hit = client.query(embedding=[1.0, 0.0, 0.0, 0.0], threshold=0.90, model_id=model_id)
    print(f"Exact match: hit={hit['hit']}, response={hit.get('response')}")

    miss = client.query(embedding=[0.5, 0.5, 0.5, 0.5], threshold=0.99, model_id=model_id)
    print(f"Different vector at high threshold: hit={miss['hit']}")

    print("Stats:", client.stats())


if __name__ == "__main__":
    main()
