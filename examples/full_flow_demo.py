import argparse
import os

from vmem.cache import ValueCache
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.pipeline import MemoryPipeline
from vmem.retrieval.retriever import MemoryRetriever
from vmem.settings import build_config


SAMPLE_TEXT = (
    "Alice met the CEO during the conference opening ceremony. "
    "The nurse called the doctor about an urgent lab result."
)
SAMPLE_QUERY = "who met the CEO"


def _check_env():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("NEO4J_URI"):
        missing.append("NEO4J_URI")
    if not os.getenv("NEO4J_USER"):
        missing.append("NEO4J_USER")
    if not os.getenv("NEO4J_PASSWORD"):
        missing.append("NEO4J_PASSWORD")
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ValueMEM full-flow demo")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--text", default=SAMPLE_TEXT, help="Memory text to ingest")
    parser.add_argument("--query", default=SAMPLE_QUERY, help="Query text")
    args = parser.parse_args()

    _check_env()
    config = build_config(args.config)

    llm = LLMClient(config.llm)
    embedder = OpenAIEmbedder(config.embedding)
    graph = GraphStore(config.neo4j)
    cache = ValueCache(config.cache)
    graph.ensure_schema()

    pipeline = MemoryPipeline(llm=llm, embedder=embedder, graph=graph, cache=cache)
    retriever = MemoryRetriever(graph=graph, embedder=embedder, config=config.retrieval)

    print("Ingesting...")
    records = pipeline.ingest_text(args.text, source="demo")
    print(f"Ingested {len(records)} memories")

    print("Querying...")
    results = retriever.retrieve(args.query)
    for item in results:
        print(
            f'{item["subject"]} - {item["predicate"]} - {item["object"]} | '
            f'value={item["value_score"]:.2f} sim={item.get("similarity", 0.0):.2f}'
        )

    memories = [item["memory_text"] for item in results]
    answer = llm.answer_question(args.query, memories) if memories else "insufficient"
    print("Answer:", answer)


if __name__ == "__main__":
    main()
