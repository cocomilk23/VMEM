import argparse
import os
import time
from dotenv import load_dotenv

from vmem.config import Neo4jConfig
from vmem.graph.falkordb_store import FalkorConfig, FalkorGraphStore
from vmem.graph.neo4j_store import GraphStore
from vmem.settings import build_config

# Path recall between two entities (edit here, no CLI needed)
PATH_START = "User_360"
PATH_END = "User_519"
# If RANGE_MODE=True, use 1..PATH_MAX_HOPS; otherwise use exact PATH_HOPS.
RANGE_MODE = True
PATH_MIN_HOPS = 1
PATH_MAX_HOPS = 8
PATH_HOPS = 5
PATH_LIMIT = None  # None => no LIMIT (full recall)
PRINT_PATH_LIMIT = 10
PATH_REPEATS = 5

# Benchmark mode: "paths" (enumerate) or "count" (only count paths)
BENCH_MODE = "paths"

# FalkorDB query timeout (ms). Increase if queries are heavy.
FALKOR_TIMEOUT_MS = 60000


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


def _run_path_query_neo4j(graph, start_name: str, hops: int) -> None:
    cypher = (
        "MATCH (s:Entity {name: $name})-[:MEMORY*"
        + str(int(hops))
        + "]->(o:Entity) RETURN count(o) AS cnt"
    )
    with graph._driver.session(database=graph.config.database) as session:
        result = session.run(cypher, {"name": start_name})
        result.consume()


def _run_path_query_falkor(graph, start_name: str, hops: int) -> None:
    cypher = (
        "MATCH (s:Entity {name: '"
        + start_name.replace("'", "\\'")
        + "'})-[:MEMORY*"
        + str(int(hops))
        + "]->(o:Entity) RETURN count(o) AS cnt"
    )
    graph._graph.query(cypher)


def _count_paths_neo4j(graph, start_name: str, end_name: str, hop_pattern: str) -> int:
    cypher = (
        "MATCH p=(s:Entity {name: $start})-[:MEMORY"
        + hop_pattern
        + "]->(o:Entity {name: $end}) "
        "WHERE all(n IN nodes(p) WHERE single(m IN nodes(p) WHERE m = n)) "
        "RETURN count(p) AS cnt"
    )
    with graph._driver.session(database=graph.config.database) as session:
        record = session.run(cypher, {"start": start_name, "end": end_name}).single()
        return int(record["cnt"]) if record is not None else 0


def _count_paths_falkor(graph, start_name: str, end_name: str, hop_pattern: str) -> int:
    cypher = (
        "MATCH p=(s:Entity {name: '"
        + start_name.replace("'", "\\'")
        + "'})-[:MEMORY"
        + hop_pattern
        + "]->(o:Entity {name: '"
        + end_name.replace("'", "\\'")
        + "'}) "
        "WHERE all(n IN nodes(p) WHERE single(m IN nodes(p) WHERE m = n)) "
        "RETURN count(p) AS cnt"
    )
    res = graph._graph.query(cypher)
    return int(res.result_set[0][0]) if res.result_set else 0


def _time_count_paths(fn, start_name: str, end_name: str, hop_pattern: str, repeats: int):
    total_ms = 0.0
    last_count = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_count = fn(start_name, end_name, hop_pattern)
        total_ms += (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / max(1, repeats)
    return avg_ms, last_count


def _exact_paths_neo4j(
    graph, start_name: str, end_name: str, hop_pattern: str, limit: int | None
) -> list[tuple[list[str], list[str]]]:
    cypher = (
        "MATCH p=(s:Entity {name: $start})-[:MEMORY"
        + hop_pattern
        + "]->(o:Entity {name: $end}) "
        "WHERE all(n IN nodes(p) WHERE single(m IN nodes(p) WHERE m = n)) "
        "RETURN [n IN nodes(p) | n.name] AS nodes, "
        "[r IN relationships(p) | r.predicate] AS rels "
    )
    if limit is not None:
        cypher += "LIMIT " + str(int(limit))
    with graph._driver.session(database=graph.config.database) as session:
        records = session.run(cypher, {"start": start_name, "end": end_name})
        rows = [(record["nodes"], record["rels"]) for record in records]
        records.consume()
        return rows


def _exact_paths_falkor(
    graph, start_name: str, end_name: str, hop_pattern: str, limit: int | None
) -> list[tuple[list[str], list[str]]]:
    cypher = (
        "MATCH p=(s:Entity {name: '"
        + start_name.replace("'", "\\'")
        + "'})-[:MEMORY"
        + hop_pattern
        + "]->(o:Entity {name: '"
        + end_name.replace("'", "\\'")
        + "'}) "
        "WHERE all(n IN nodes(p) WHERE single(m IN nodes(p) WHERE m = n)) "
        "RETURN [n IN nodes(p) | n.name] AS nodes, "
        "[r IN relationships(p) | r.predicate] AS rels "
    )
    if limit is not None:
        cypher += "LIMIT " + str(int(limit))
    res = graph._graph.query(cypher, timeout=FALKOR_TIMEOUT_MS)
    return [(row[0], row[1]) for row in res.result_set]


def _time_exact_paths(fn, start_name: str, end_name: str, hop_pattern: str, limit: int | None, repeats: int):
    total_ms = 0.0
    last_paths = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_paths = fn(start_name, end_name, hop_pattern, limit)
        total_ms += (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / max(1, repeats)
    return avg_ms, last_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact H-hop path recall/count: Neo4j vs FalkorDB")
    args = parser.parse_args()

    neo4j, falkor, config = _build_backends()
    try:
        hop_pattern = (
            f"*{PATH_MIN_HOPS}..{PATH_MAX_HOPS}" if RANGE_MODE else f"*{PATH_HOPS}"
        )
        if BENCH_MODE == "count":
            neo4j_ms, neo4j_count = _time_count_paths(
                lambda start, end, hp: _count_paths_neo4j(neo4j, start, end, hp),
                PATH_START,
                PATH_END,
                hop_pattern,
                PATH_REPEATS,
            )
            falkor_ms, falkor_count = _time_count_paths(
                lambda start, end, hp: _count_paths_falkor(falkor, start, end, hp),
                PATH_START,
                PATH_END,
                hop_pattern,
                PATH_REPEATS,
            )
            print(
                f"Path COUNT {hop_pattern} from {PATH_START} to {PATH_END} "
                f"(avg over {PATH_REPEATS} runs)"
            )
            print(f"  Neo4j: count={neo4j_count} avg={neo4j_ms:.2f} ms")
            print(f"  FalkorDB: count={falkor_count} avg={falkor_ms:.2f} ms")
        else:
            # Exact H-hop paths recall
            neo4j_ms, neo4j_paths = _time_exact_paths(
                lambda start, end, hp, limit: _exact_paths_neo4j(neo4j, start, end, hp, limit),
                PATH_START,
                PATH_END,
                hop_pattern,
                PATH_LIMIT,
                PATH_REPEATS,
            )
            falkor_ms, falkor_paths = _time_exact_paths(
                lambda start, end, hp, limit: _exact_paths_falkor(falkor, start, end, hp, limit),
                PATH_START,
                PATH_END,
                hop_pattern,
                PATH_LIMIT,
                PATH_REPEATS,
            )

            limit_label = "none" if PATH_LIMIT is None else str(PATH_LIMIT)
            print(
                f"Paths {hop_pattern} from {PATH_START} to {PATH_END} "
                f"(limit={limit_label})"
            )
            # def _format_path(nodes, rels):
            #     parts = []
            #     for i, node in enumerate(nodes):
            #         parts.append(node)
            #         if i < len(rels):
            #             parts.append(rels[i])
            #     return " --- ".join(parts)

            print(f"  Neo4j: {len(neo4j_paths)} paths avg {neo4j_ms:.2f} ms over {PATH_REPEATS} runs")
            # for nodes, rels in neo4j_paths[:PRINT_PATH_LIMIT]:
            #     print("  - " + _format_path(nodes, rels))
            print(f"  FalkorDB: {len(falkor_paths)} paths avg {falkor_ms:.2f} ms over {PATH_REPEATS} runs")
            # for nodes, rels in falkor_paths[:PRINT_PATH_LIMIT]:
            #     print("  - " + _format_path(nodes, rels))
    finally:
        neo4j.close()
        falkor.close()


if __name__ == "__main__":
    main()
