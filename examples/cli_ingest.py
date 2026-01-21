import argparse

from vmem.cache import ValueCache
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.pipeline import MemoryPipeline
from vmem.settings import build_config


def main() -> None:
    parser = argparse.ArgumentParser(description="ValueMEM CLI ingest example")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--text", required=True, help="Memory text to ingest")
    args = parser.parse_args()

    config = build_config(args.config)
    llm = LLMClient(config.llm)
    embedder = OpenAIEmbedder(config.embedding)
    graph = GraphStore(config.neo4j)
    cache = ValueCache(config.cache)
    graph.ensure_schema()

    pipeline = MemoryPipeline(llm=llm, embedder=embedder, graph=graph, cache=cache)
    records = pipeline.ingest_text(args.text, source="example")
    print(f"Ingested {len(records)} memories")


if __name__ == "__main__":
    main()
