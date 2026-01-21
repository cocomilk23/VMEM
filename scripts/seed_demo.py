import argparse

from vmem.cache import ValueCache
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.pipeline import MemoryPipeline
from vmem.settings import build_config

SAMPLE_TEXTS = [
    "The nurse called the doctor about an urgent lab result.",
    "A user reported a suspicious transaction on their credit card.",
    "The patient mentioned severe chest pain and shortness of breath.",
    "Alice met the CEO during the conference opening ceremony.",
    "The agent confirmed the customer's new address and phone number.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ValueMEM with demo memories")
    parser.add_argument("--config", help="Path to YAML config file")
    args = parser.parse_args()

    config = build_config(args.config)
    llm = LLMClient(config.llm)
    embedder = OpenAIEmbedder(config.embedding)
    graph = GraphStore(config.neo4j)
    cache = ValueCache(config.cache)
    graph.ensure_schema()

    pipeline = MemoryPipeline(llm=llm, embedder=embedder, graph=graph, cache=cache)
    for text in SAMPLE_TEXTS:
        pipeline.ingest_text(text, source="demo")
    print(f"Seeded {len(SAMPLE_TEXTS)} documents")


if __name__ == "__main__":
    main()
