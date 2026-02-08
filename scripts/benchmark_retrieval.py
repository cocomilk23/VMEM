import argparse
import time

from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.pipeline import MemoryPipeline
from vmem.retrieval.retriever import MemoryRetriever
from vmem.settings import build_config

SAMPLE_QUERIES = [
    "urgent lab results",
    "credit card fraud",
    "who met the CEO",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ValueMEM retrieval latency")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
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
    pipeline.ingest_text(
        "Alice met the CEO during the conference opening ceremony.",
        source="bench",
        add_vector=True,
        add_graph=False,
        immediate=True,
    )
    pipeline.flush_buffer(source="bench", add_vector=True, add_graph=False)

    retriever = MemoryRetriever(graph=graph, embedder=embedder, llm=llm, config=config.retrieval)

    for _ in range(args.warmup):
        for query in SAMPLE_QUERIES:
            retriever.retrieve(query)

    total = 0.0
    for _ in range(args.runs):
        start = time.perf_counter()
        for query in SAMPLE_QUERIES:
            retriever.retrieve(query)
        total += time.perf_counter() - start

    avg = total / max(args.runs, 1)
    print(f"Average batch latency: {avg:.3f}s for {len(SAMPLE_QUERIES)} queries")


if __name__ == "__main__":
    main()
