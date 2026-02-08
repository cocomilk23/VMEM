import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from vmem.config import Neo4jConfig
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.embeddings.local_embedder import LocalEmbedder
from vmem.graph.falkordb_store import FalkorConfig, FalkorGraphStore
from vmem.graph.neo4j_store import GraphStore
from vmem.memory.models import MemoryRecord
from vmem.settings import build_config


def _build_backends():
    load_dotenv()
    config = build_config(os.getenv("VMEM_CONFIG_PATH"))
    neo4j = GraphStore(
        Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:8688"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "jjb123456"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
    )
    falkor = FalkorGraphStore(
        FalkorConfig(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
            graph=os.getenv("FALKORDB_GRAPH", "VMEM_BENCH"),
        )
    )
    neo4j.ensure_schema()
    falkor.ensure_schema()
    return neo4j, falkor, config


def _generate_random_graph_records(
    node_count: int,
    edge_count: int,
    seed: int,
    save_path: str | None,
) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    if node_count < 21:
        node_count = 21
    nodes = [f"User_{i}" for i in range(node_count)]
    relation_types = ["knows", "works_with", "likes", "visited", "owns", "follows"]
    records = []
    # Ensure random multi-hop paths exist: min length 2, max length 20.
    path_count = max(5, min(50, edge_count // 20))
    for _ in range(path_count):
        path_len = rng.randint(2, 20)
        path_nodes = [nodes[rng.randrange(node_count)] for _ in range(path_len + 1)]
        for i in range(path_len):
            subject = path_nodes[i]
            object_ = path_nodes[i + 1]
            if subject == object_:
                continue
            predicate = "path"
            value_score = rng.random()
            record = MemoryRecord(
                subject=subject,
                predicate=predicate,
                object=object_,
                memory_text=f"{subject} {predicate} {object_}",
                value_score=value_score,
                high_value=value_score >= 0.8,
                embedding=[round(rng.random(), 6) for _ in range(3)],
                created_at=now,
                source="bench",
                occurred_at=None,
            )
            subj_score = rng.random()
            obj_score = rng.random()
            records.append(
                {
                    "record": record,
                    "subject_score": subj_score,
                    "object_score": obj_score,
                    "subject_high_value": subj_score >= 0.7,
                    "object_high_value": obj_score >= 0.7,
                }
            )
    for i in range(edge_count):
        subject = nodes[rng.randrange(node_count)]
        object_ = nodes[rng.randrange(node_count)]
        if subject == object_:
            object_ = nodes[(nodes.index(subject) + 1) % node_count]
        predicate = relation_types[i % len(relation_types)]
        value_score = rng.random()
        record = MemoryRecord(
            subject=subject,
            predicate=predicate,
            object=object_,
            memory_text=f"{subject} {predicate} {object_}",
            value_score=value_score,
            high_value=value_score >= 0.8,
            embedding=[round(rng.random(), 6) for _ in range(3)],
            created_at=now,
            source="bench",
            occurred_at=None,
        )
        subj_score = rng.random()
        obj_score = rng.random()
        records.append(
            {
                "record": record,
                "subject_score": subj_score,
                "object_score": obj_score,
                "subject_high_value": subj_score >= 0.7,
                "object_high_value": obj_score >= 0.7,
            }
        )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "seed": seed,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "path_count": path_count,
                        "path_len_min": 2,
                        "path_len_max": 20,
                    }
                )
                + "\n"
            )
            for item in records:
                rec = item["record"]
                payload = {
                    "subject": rec.subject,
                    "predicate": rec.predicate,
                    "object": rec.object,
                    "memory_text": rec.memory_text,
                    "value_score": rec.value_score,
                    "high_value": rec.high_value,
                    "created_at": rec.created_at.isoformat(),
                    "subject_score": item["subject_score"],
                    "object_score": item["object_score"],
                    "subject_high_value": item["subject_high_value"],
                    "object_high_value": item["object_high_value"],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return records, nodes


def _insert_graph_records(graph, records) -> float:
    t0 = time.perf_counter()
    for item in records:
        graph.write_memory(
            item["record"],
            subject_score=item["subject_score"],
            object_score=item["object_score"],
            subject_high_value=item["subject_high_value"],
            object_high_value=item["object_high_value"],
        )
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare graph retrieval latency: Neo4j vs FalkorDB")
    parser.add_argument("--nodes", type=int, default=10000, help="Random graph node count")
    parser.add_argument("--edges", type=int, default=50000, help="Random graph edge count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default="results/random_graph.jsonl",
        help="Path to save generated graph data (JSONL)",
    )
    args = parser.parse_args()

    neo4j, falkor, config = _build_backends()
    try:
        print("Inserting graph memories...")
        records, nodes = _generate_random_graph_records(
            node_count=args.nodes,
            edge_count=args.edges,
            seed=args.seed,
            save_path=args.save_path,
        )
        neo4j_ms = _insert_graph_records(neo4j, records)
        falkor_ms = _insert_graph_records(falkor, records)
        print(f"Insert time (ms) | Neo4j={neo4j_ms:.1f} | FalkorDB={falkor_ms:.1f}")

        print("Insert complete.")
    finally:
        neo4j.close()
        falkor.close()


if __name__ == "__main__":
    main()
