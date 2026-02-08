from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from typing import Any

from falkordb import FalkorDB

from vmem.config import Neo4jConfig
from vmem.utils import cosine_similarity


@dataclass
class FalkorConfig:
    host: str = "localhost"
    port: int = 6379
    graph: str = "VMEM"


def _cypher_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return "'" + escaped + "'"


def _bool_literal(value: bool | None) -> str:
    return "true" if value else "false"


def _encode_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding)


def _decode_embedding(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except json.JSONDecodeError:
            return []
    return []


@dataclass
class FalkorGraphStore:
    config: FalkorConfig

    def __post_init__(self) -> None:
        self._db = FalkorDB(host=self.config.host, port=self.config.port)
        self._graph = self._db.select_graph(self.config.graph)

    def close(self) -> None:
        # FalkorDB client does not require explicit close for graph objects.
        return None

    def ensure_schema(self) -> None:
        logger = logging.getLogger(__name__)
        statements = [
            "CREATE INDEX ON :Entity(name)",
            "CREATE INDEX ON :Entity(high_value)",
            "CREATE INDEX ON :Entity(value_score)",
            "CREATE INDEX ON :VectorMemory(value_score)",
            "CREATE INDEX ON :VectorMemory(high_value)",
            "CREATE INDEX ON :UserProfile(value_score)",
            "CREATE INDEX ON :UserProfile(high_value)",
        ]
        for stmt in statements:
            try:
                self._graph.query(stmt)
            except Exception as exc:
                logger.debug("Skipping schema statement in FalkorDB: %s (%s)", stmt, exc)

    def ensure_vector_index(self, index_name: str, dimensions: int) -> None:
        # FalkorDB does not expose a Neo4j-compatible vector index API.
        return None

    def ensure_profile_vector_index(self, index_name: str, dimensions: int) -> None:
        return None

    def write_memory(
        self,
        record: Any,
        subject_score: float | None = None,
        object_score: float | None = None,
        subject_high_value: bool | None = None,
        object_high_value: bool | None = None,
    ) -> None:
        created_at = record.created_at.astimezone(timezone.utc).isoformat()
        cypher = (
            "MERGE (s:Entity {name: " + _cypher_str(record.subject) + "}) "
            "MERGE (o:Entity {name: " + _cypher_str(record.object) + "}) "
            "MERGE (s)-[r:MEMORY {predicate: " + _cypher_str(record.predicate) + ", "
            "memory_text: " + _cypher_str(record.memory_text) + "}]->(o) "
            "SET s.high_value = coalesce(s.high_value, false) OR "
            + _bool_literal(subject_high_value)
            + ", "
            "o.high_value = coalesce(o.high_value, false) OR "
            + _bool_literal(object_high_value)
            + ", "
            "s.value_score = CASE "
            "WHEN s.value_score IS NULL THEN " + (str(subject_score) if subject_score is not None else "NULL") + " "
            "WHEN " + (str(subject_score) if subject_score is not None else "NULL") + " IS NULL THEN s.value_score "
            "WHEN s.value_score >= " + (str(subject_score) if subject_score is not None else "0") + " "
            "THEN s.value_score ELSE " + (str(subject_score) if subject_score is not None else "0") + " END, "
            "o.value_score = CASE "
            "WHEN o.value_score IS NULL THEN " + (str(object_score) if object_score is not None else "NULL") + " "
            "WHEN " + (str(object_score) if object_score is not None else "NULL") + " IS NULL THEN o.value_score "
            "WHEN o.value_score >= " + (str(object_score) if object_score is not None else "0") + " "
            "THEN o.value_score ELSE " + (str(object_score) if object_score is not None else "0") + " END, "
            "r.value_score = " + str(float(record.value_score)) + ", "
            "r.high_value = " + _bool_literal(bool(record.high_value)) + ", "
            "r.embedding = " + _cypher_str(_encode_embedding(record.embedding)) + ", "
            "r.occurred_at = " + (_cypher_str(record.occurred_at) if record.occurred_at else "NULL") + ", "
            "r.created_at = " + _cypher_str(created_at) + ", "
            "r.access_count = coalesce(r.access_count, 0), "
            "r.source = " + _cypher_str(record.source)
        )
        self._graph.query(cypher)

    def write_vector_memory(
        self,
        memory_text: str,
        value_score: float,
        embedding: list[float],
        created_at: datetime,
        source: str,
        high_value: bool,
        occurred_at: str | None = None,
    ) -> None:
        cypher = (
            "CREATE (m:VectorMemory {"
            "memory_text: " + _cypher_str(memory_text) + ", "
            "value_score: " + str(float(value_score)) + ", "
            "high_value: " + _bool_literal(high_value) + ", "
            "embedding: " + _cypher_str(_encode_embedding(embedding)) + ", "
            "occurred_at: " + (_cypher_str(occurred_at) if occurred_at else "NULL") + ", "
            "created_at: " + _cypher_str(created_at.astimezone(timezone.utc).isoformat()) + ", "
            "source: " + _cypher_str(source) + ", "
            "access_count: 0"
            "})"
        )
        self._graph.query(cypher)

    def write_profile_memory(
        self,
        memory_text: str,
        value_score: float,
        embedding: list[float],
        created_at: datetime,
        source: str,
        high_value: bool,
    ) -> None:
        cypher = (
            "CREATE (p:UserProfile {"
            "memory_text: " + _cypher_str(memory_text) + ", "
            "value_score: " + str(float(value_score)) + ", "
            "high_value: " + _bool_literal(high_value) + ", "
            "embedding: " + _cypher_str(_encode_embedding(embedding)) + ", "
            "created_at: " + _cypher_str(created_at.astimezone(timezone.utc).isoformat()) + ", "
            "source: " + _cypher_str(source) + ", "
            "access_count: 0"
            "})"
        )
        self._graph.query(cypher)

    def query_by_entities(
        self, entities: list[str], value_threshold: float | None, limit: int
    ) -> list[dict]:
        if not entities:
            return []
        ent_list = "[" + ", ".join(_cypher_str(e) for e in entities) + "]"
        where_clause = f"WHERE s.name IN {ent_list} OR o.name IN {ent_list}"
        if value_threshold is not None:
            where_clause += f" AND r.value_score >= {float(value_threshold)}"
        cypher = (
            "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
            + where_clause
            + " RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
            "r.memory_text AS memory_text, r.value_score AS value_score, r.high_value AS high_value, "
            "r.embedding AS embedding "
            "ORDER BY r.value_score DESC LIMIT "
            + str(int(limit))
        )
        res = self._graph.query(cypher)
        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "memory_text": row[3],
                "value_score": row[4],
                "high_value": row[5],
                "embedding": _decode_embedding(row[6]),
            }
            for row in res.result_set
        ]

    def query_entity_names_fuzzy(
        self, entities: list[str], index_name: str, limit: int
    ) -> list[str]:
        # FalkorDB fulltext index syntax may differ; keep a safe fallback.
        return []

    def query_vector(
        self,
        embedding: list[float],
        limit: int,
        index_name: str,
        value_threshold: float | None = None,
        only_low_value: bool = False,
    ) -> list[dict]:
        candidates = self.query_vector_memories(value_threshold, limit, only_low_value=only_low_value)
        scored = []
        for item in candidates:
            similarity = cosine_similarity(embedding, item.get("embedding") or [])
            item = dict(item)
            item["similarity"] = similarity
            scored.append(item)
        scored.sort(key=lambda x: float(x.get("similarity") or 0.0), reverse=True)
        return scored

    def query_vector_memories(
        self, value_threshold: float | None, limit: int, only_low_value: bool = False
    ) -> list[dict]:
        where_parts = []
        if value_threshold is not None:
            if only_low_value:
                where_parts.append(f"m.value_score < {float(value_threshold)}")
            else:
                where_parts.append(f"m.value_score >= {float(value_threshold)}")
        elif only_low_value:
            where_parts.append("m.high_value = false")
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        cypher = (
            "MATCH (m:VectorMemory) "
            + where_clause
            + " RETURN id(m) AS node_id, m.memory_text AS memory_text, "
            "m.value_score AS value_score, m.high_value AS high_value, "
            "m.embedding AS embedding, m.source AS source "
            "ORDER BY m.value_score DESC LIMIT "
            + str(int(limit))
        )
        res = self._graph.query(cypher)
        out = []
        for row in res.result_set:
            out.append(
                {
                    "node_id": row[0],
                    "memory_text": row[1],
                    "value_score": row[2],
                    "high_value": row[3],
                    "embedding": _decode_embedding(row[4]),
                    "source": row[5],
                }
            )
        return out

    def expand_neighbors(self, node_names: list[str], hops: int, limit: int) -> list[dict]:
        if not node_names:
            return []
        ent_list = "[" + ", ".join(_cypher_str(e) for e in node_names) + "]"
        if hops <= 1:
            cypher = (
                "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
                f"WHERE s.name IN {ent_list} OR o.name IN {ent_list} "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding "
                "LIMIT "
                + str(int(limit))
            )
        else:
            cypher = (
                "MATCH p=(a:Entity)-[:MEMORY*1.."
                + str(int(hops))
                + "]-(b:Entity) "
                f"WHERE a.name IN {ent_list} OR b.name IN {ent_list} "
                "UNWIND relationships(p) AS r "
                "WITH DISTINCT r "
                "MATCH (s:Entity)-[r]->(o:Entity) "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding "
                "LIMIT "
                + str(int(limit))
            )
        res = self._graph.query(cypher)
        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "memory_text": row[3],
                "value_score": row[4],
                "embedding": _decode_embedding(row[5]),
            }
            for row in res.result_set
        ]

    def bump_access(self, subject: str, predicate: str, object_: str) -> None:
        cypher = (
            "MATCH (s:Entity {name: "
            + _cypher_str(subject)
            + "})-[r:MEMORY {predicate: "
            + _cypher_str(predicate)
            + "}]->(o:Entity {name: "
            + _cypher_str(object_)
            + "}) "
            "SET r.access_count = coalesce(r.access_count, 0) + 1, "
            "r.last_accessed = "
            + _cypher_str(datetime.now(timezone.utc).isoformat())
        )
        self._graph.query(cypher)

    def bump_vector_access(self, node_id: str) -> None:
        cypher = (
            "MATCH (m:VectorMemory) WHERE id(m) = "
            + str(node_id)
            + " SET m.access_count = coalesce(m.access_count, 0) + 1, "
            "m.last_accessed = "
            + _cypher_str(datetime.now(timezone.utc).isoformat())
        )
        self._graph.query(cypher)

    def get_entity_scores(self, names: list[str]) -> dict[str, float | None]:
        if not names:
            return {}
        ent_list = "[" + ", ".join(_cypher_str(e) for e in names) + "]"
        cypher = (
            "MATCH (e:Entity) WHERE e.name IN "
            + ent_list
            + " RETURN e.name AS name, e.value_score AS value_score"
        )
        res = self._graph.query(cypher)
        out: dict[str, float | None] = {}
        for row in res.result_set:
            name = row[0]
            value = row[1]
            try:
                out[str(name)] = float(value) if value is not None else None
            except (TypeError, ValueError):
                out[str(name)] = None
        return out


def build_falkor_config_from_neo4j(neo4j_config: Neo4jConfig) -> FalkorConfig:
    # Helper to mirror the neo4j config style when porting demo code.
    # Uses Neo4jConfig as a placeholder only; not used by FalkorDB itself.
    return FalkorConfig(
        host="localhost",
        port=6379,
        graph="VMEM",
    )
