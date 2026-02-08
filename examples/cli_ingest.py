import argparse

from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.pipeline import MemoryPipeline
from vmem.settings import build_config


def main() -> None:
    parser = argparse.ArgumentParser(description="ValueMEM CLI ingest example")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--text", required=True, help="Memory text to ingest")
    parser.add_argument(
        "--add-graph",
        action="store_true",
        help="Also update entity-relationship graph (default: vector only)",
    )
    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Immediately extract memories from current buffer",
    )
    args = parser.parse_args()

    config = build_config(args.config)
    llm = LLMClient(config.llm)
    embedder = OpenAIEmbedder(config.embedding)
    graph = GraphStore(config.neo4j)
    graph.ensure_schema()

    pipeline = MemoryPipeline(
        llm=llm,
        embedder=embedder,
        graph=graph,
        value_threshold=config.retrieval.value_threshold,
        vector_index_name=config.retrieval.vector_index_name,
    )
    result = pipeline.ingest_text(
        args.text,
        source="example",
        add_vector=True,
        add_graph=args.add_graph,
        immediate=args.immediate,
    )
    result.extend(
        pipeline.flush_buffer(
            source="example",
            add_vector=True,
            add_graph=args.add_graph,
        )
    )
    total = len(result.vector_records) + len(result.graph_records)
    print(f"Ingested {total} memories")


if __name__ == "__main__":
    main()
