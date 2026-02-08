import argparse

from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.retrieval.retriever import MemoryRetriever
from vmem.settings import build_config


def main() -> None:
    parser = argparse.ArgumentParser(description="ValueMEM CLI query example")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--text", required=True, help="Query text")
    args = parser.parse_args()

    config = build_config(args.config)
    llm = LLMClient(config.llm)
    embedder = OpenAIEmbedder(config.embedding)
    graph = GraphStore(config.neo4j)
    graph.ensure_schema()

    retriever = MemoryRetriever(graph=graph, embedder=embedder, llm=llm, config=config.retrieval)
    results = retriever.retrieve(args.text)
    for item in results:
        if item.get("record_type") == "vector":
            print(
                f'vector | {item["memory_text"]} | '
                f'value={item["value_score"]:.2f} sim={item.get("similarity", 0.0):.2f}'
            )
        else:
            print(
                f'{item["subject"]} - {item["predicate"]} - {item["object"]} | '
                f'value={item["value_score"]:.2f} sim={item.get("similarity", 0.0):.2f}'
            )


if __name__ == "__main__":
    main()
