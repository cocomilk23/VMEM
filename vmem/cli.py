import argparse

from vmem.cache import ValueCache
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.logging_config import setup_logging
from vmem.memory.pipeline import MemoryPipeline
from vmem.retrieval.retriever import MemoryRetriever
from vmem.settings import build_config


def _build_components(config_path: str | None):
    config = build_config(config_path)
    llm = LLMClient(config.llm)
    embedder = OpenAIEmbedder(config.embedding)
    graph = GraphStore(config.neo4j)
    cache = ValueCache(config.cache)
    return config, llm, embedder, graph, cache


def main() -> None:
    parser = argparse.ArgumentParser(description="ValueMEM CLI")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init-schema", help="Initialize Neo4j schema and indexes")

    ingest_cmd = sub.add_parser("ingest", help="Ingest raw text into ValueMEM")
    ingest_cmd.add_argument("text", help="Raw input text")
    ingest_cmd.add_argument("--source", default="user", help="Memory source label")

    query_cmd = sub.add_parser("query", help="Query ValueMEM")
    query_cmd.add_argument("text", help="Query text")

    args = parser.parse_args()
    setup_logging(args.log_level)
    config, llm, embedder, graph, cache = _build_components(args.config)
    graph.ensure_schema()

    if args.command == "init-schema":
        print("Schema initialized")
        return

    if args.command == "ingest":
        pipeline = MemoryPipeline(llm=llm, embedder=embedder, graph=graph, cache=cache)
        records = pipeline.ingest_text(args.text, source=args.source)
        print(f"Ingested {len(records)} memories")
        return

    if args.command == "query":
        retriever = MemoryRetriever(graph=graph, embedder=embedder, config=config.retrieval)
        results = retriever.retrieve(args.text)
        for item in results:
            print(
                f'{item["subject"]} - {item["predicate"]} - {item["object"]} | '
                f'value={item["value_score"]:.2f} sim={item["similarity"]:.2f}'
            )


if __name__ == "__main__":
    main()
