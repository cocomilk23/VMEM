from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
import logging

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from vmem.config import Neo4jConfig

if TYPE_CHECKING:
    from vmem.memory.models import MemoryRecord


@dataclass
class GraphStore:
    config: Neo4jConfig

    def __post_init__(self) -> None:
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
        )

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        logger = logging.getLogger(__name__)
        cypher = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX entity_high_value IF NOT EXISTS FOR (e:Entity) ON (e.high_value)",
            "CREATE INDEX entity_value_score IF NOT EXISTS FOR (e:Entity) ON (e.value_score)",
            "CREATE FULLTEXT INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
            "CREATE CONSTRAINT rel_predicate_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.predicate IS NOT NULL",
            "CREATE CONSTRAINT rel_value_score_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.value_score IS NOT NULL",
            "CREATE CONSTRAINT rel_memory_text_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.memory_text IS NOT NULL",
            "CREATE INDEX rel_value_score IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.value_score)",
            "CREATE INDEX rel_created_at IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.created_at)",
            "CREATE INDEX rel_predicate IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.predicate)",
            "CREATE INDEX rel_high_value IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.high_value)",
            "CREATE INDEX vecmem_value_score IF NOT EXISTS FOR (m:VectorMemory) ON (m.value_score)",
            "CREATE INDEX vecmem_high_value IF NOT EXISTS FOR (m:VectorMemory) ON (m.high_value)",
            "CREATE INDEX profile_value_score IF NOT EXISTS FOR (p:UserProfile) ON (p.value_score)",
            "CREATE INDEX profile_high_value IF NOT EXISTS FOR (p:UserProfile) ON (p.high_value)",
        ]
        with self._driver.session(database=self.config.database) as session:
            for stmt in cypher:
                try:
                    session.run(stmt)
                except Neo4jError as exc:
                    code = getattr(exc, "code", "")
                    message = str(exc)
                    if (
                        code == "Neo.DatabaseError.Schema.ConstraintCreationFailed"
                        and "Enterprise Edition" in message
                    ):
                        logger.debug(
                            "Skipping enterprise-only constraint in Community Edition: %s",
                            stmt,
                        )
                        continue
                    raise

    def write_memory(
        self,
        record: MemoryRecord,
        subject_score: float | None = None,
        object_score: float | None = None,
        subject_high_value: bool | None = None,
        object_high_value: bool | None = None,
    ) -> None:
        created_at = record.created_at.astimezone(timezone.utc).isoformat()
        cypher = (
            "MERGE (s:Entity {name: $subject}) "
            "MERGE (o:Entity {name: $object}) "
            "MERGE (s)-[r:MEMORY {predicate: $predicate, memory_text: $memory_text}]->(o) "
            "SET s.high_value = coalesce(s.high_value, false) OR $subject_high_value, "
            "o.high_value = coalesce(o.high_value, false) OR $object_high_value, "
            "s.value_score = CASE "
            "WHEN s.value_score IS NULL THEN $subject_score "
            "WHEN $subject_score IS NULL THEN s.value_score "
            "WHEN s.value_score >= $subject_score THEN s.value_score ELSE $subject_score END, "
            "o.value_score = CASE "
            "WHEN o.value_score IS NULL THEN $object_score "
            "WHEN $object_score IS NULL THEN o.value_score "
            "WHEN o.value_score >= $object_score THEN o.value_score ELSE $object_score END, "
            "r.value_score = $value_score, "
            "r.high_value = $high_value, "
            "r.embedding = $embedding, "
            "r.occurred_at = $occurred_at, "
            "r.created_at = $created_at, "
            "r.access_count = coalesce(r.access_count, 0), "
            "r.source = $source "
        )
        params = {
            "subject": record.subject,
            "predicate": record.predicate,
            "object": record.object,
            "memory_text": record.memory_text,
            "value_score": float(record.value_score),
            "high_value": bool(record.high_value),
            "subject_score": float(subject_score) if subject_score is not None else None,
            "object_score": float(object_score) if object_score is not None else None,
            "subject_high_value": bool(subject_high_value) if subject_high_value is not None else False,
            "object_high_value": bool(object_high_value) if object_high_value is not None else False,
            "embedding": record.embedding,
            "occurred_at": record.occurred_at,
            "created_at": created_at,
            "source": record.source,
        }
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, params)

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
            "memory_text: $memory_text, "
            "value_score: $value_score, "
            "high_value: $high_value, "
            "embedding: $embedding, "
            "occurred_at: $occurred_at, "
            "created_at: $created_at, "
            "source: $source, "
            "access_count: 0"
            "})"
        )
        params = {
            "memory_text": memory_text,
            "value_score": float(value_score),
            "high_value": bool(high_value),
            "embedding": embedding,
            "occurred_at": occurred_at,
            "created_at": created_at.astimezone(timezone.utc).isoformat(),
            "source": source,
        }
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, params)

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
            "memory_text: $memory_text, "
            "value_score: $value_score, "
            "high_value: $high_value, "
            "embedding: $embedding, "
            "created_at: $created_at, "
            "source: $source, "
            "access_count: 0"
            "})"
        )
        params = {
            "memory_text": memory_text,
            "value_score": float(value_score),
            "high_value": bool(high_value),
            "embedding": embedding,
            "created_at": created_at.astimezone(timezone.utc).isoformat(),
            "source": source,
        }
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, params)

    def ensure_vector_index(self, index_name: str, dimensions: int) -> None:
        cypher = (
            "CREATE VECTOR INDEX " + index_name + " IF NOT EXISTS "
            "FOR (m:VectorMemory) ON (m.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
            "`vector.similarity_function`: 'cosine'}}"
        )
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, {"dim": int(dimensions)})

    def ensure_profile_vector_index(self, index_name: str, dimensions: int) -> None:
        cypher = (
            "CREATE VECTOR INDEX " + index_name + " IF NOT EXISTS "
            "FOR (p:UserProfile) ON (p.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
            "`vector.similarity_function`: 'cosine'}}"
        )
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, {"dim": int(dimensions)})

    def query_memories(self, value_threshold: float | None, limit: int) -> list[dict]:
        if value_threshold is None:
            cypher = (
                "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.high_value AS high_value, "
                "r.embedding AS embedding "
                "ORDER BY r.value_score DESC LIMIT $limit"
            )
            params = {"limit": limit}
        else:
            cypher = (
                "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
                "WHERE r.value_score >= $threshold "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.high_value AS high_value, "
                "r.embedding AS embedding "
                "ORDER BY r.value_score DESC LIMIT $limit"
            )
            params = {"threshold": float(value_threshold), "limit": limit}

        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, params)
            return [record.data() for record in records]

    def query_by_entities(
        self, entities: list[str], value_threshold: float | None, limit: int
    ) -> list[dict]:
        if not entities:
            return []
        params = {"entities": entities, "limit": limit}
        where_clause = "WHERE s.name IN $entities OR o.name IN $entities"
        if value_threshold is not None:
            where_clause += " AND r.value_score >= $threshold"
            params["threshold"] = float(value_threshold)
        cypher = (
            "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
            f"{where_clause} "
            "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
            "r.memory_text AS memory_text, r.value_score AS value_score, r.high_value AS high_value, "
            "r.embedding AS embedding "
            "ORDER BY r.value_score DESC LIMIT $limit"
        )
        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, params)
            return [record.data() for record in records]

    def query_entity_names_fuzzy(
        self, entities: list[str], index_name: str, limit: int
    ) -> list[str]:
        if not entities:
            return []
        # Build a simple OR query for fulltext search.
        terms = [t.strip() for t in entities if t and t.strip()]
        if not terms:
            return []
        query = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        cypher = (
            "CALL db.index.fulltext.queryNodes($index, $query) "
            "YIELD node, score "
            "RETURN node.name AS name "
            "ORDER BY score DESC LIMIT $limit"
        )
        params = {"index": index_name, "query": query, "limit": int(limit)}
        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, params)
            return [record["name"] for record in records]

    def query_vector(
        self,
        embedding: list[float],
        limit: int,
        index_name: str,
        value_threshold: float | None = None,
        only_low_value: bool = False,
    ) -> list[dict]:
        if value_threshold is None:
            where_clause = ""
            if only_low_value:
                where_clause = "WHERE m.high_value = false"
            cypher = (
                "CALL db.index.vector.queryNodes($index, $limit, $embedding) "
                "YIELD node AS m, score "
                f"{where_clause} "
                "RETURN elementId(m) AS node_id, m.memory_text AS memory_text, "
                "m.value_score AS value_score, m.high_value AS high_value, "
                "m.embedding AS embedding, m.source AS source, score AS similarity"
            )
            params = {"index": index_name, "limit": limit, "embedding": embedding}
        else:
            where_clause = "WHERE m.value_score >= $threshold"
            if only_low_value:
                where_clause = "WHERE m.value_score < $threshold"
            cypher = (
                "CALL db.index.vector.queryNodes($index, $limit, $embedding) "
                "YIELD node AS m, score "
                f"{where_clause} "
                "RETURN elementId(m) AS node_id, m.memory_text AS memory_text, "
                "m.value_score AS value_score, m.high_value AS high_value, "
                "m.embedding AS embedding, m.source AS source, score AS similarity"
            )
            params = {
                "index": index_name,
                "limit": limit,
                "embedding": embedding,
                "threshold": float(value_threshold),
            }

        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, params)
            return [record.data() for record in records]

    def query_vector_memories(
        self, value_threshold: float | None, limit: int, only_low_value: bool = False
    ) -> list[dict]:
        if value_threshold is None:
            where_clause = ""
            if only_low_value:
                where_clause = "WHERE m.high_value = false"
            cypher = (
                "MATCH (m:VectorMemory) "
                f"{where_clause} "
                "RETURN elementId(m) AS node_id, m.memory_text AS memory_text, "
                "m.value_score AS value_score, m.high_value AS high_value, "
                "m.embedding AS embedding, m.source AS source "
                "ORDER BY m.value_score DESC LIMIT $limit"
            )
            params = {"limit": limit}
        else:
            where_clause = "WHERE m.value_score >= $threshold"
            if only_low_value:
                where_clause = "WHERE m.value_score < $threshold"
            cypher = (
                "MATCH (m:VectorMemory) "
                f"{where_clause} "
                "RETURN elementId(m) AS node_id, m.memory_text AS memory_text, "
                "m.value_score AS value_score, m.high_value AS high_value, "
                "m.embedding AS embedding, m.source AS source "
                "ORDER BY m.value_score DESC LIMIT $limit"
            )
            params = {"threshold": float(value_threshold), "limit": limit}
        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, params)
            return [record.data() for record in records]

    def expand_neighbors(self, node_names: list[str], hops: int, limit: int) -> list[dict]:
        if not node_names:
            return []
        if hops <= 1:
            cypher = (
                "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
                "WHERE s.name IN $nodes OR o.name IN $nodes "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding "
                "LIMIT $limit"
            )
            params = {"nodes": node_names, "limit": limit}
        else:
            cypher = (
                "MATCH p=(a:Entity)-[:MEMORY*1..$hops]-(b:Entity) "
                "WHERE a.name IN $nodes OR b.name IN $nodes "
                "UNWIND relationships(p) AS r "
                "WITH DISTINCT r "
                "MATCH (s:Entity)-[r]->(o:Entity) "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding "
                "LIMIT $limit"
            )
            params = {"nodes": node_names, "limit": limit, "hops": hops}
        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, params)
            return [record.data() for record in records]

    def bump_access(self, subject: str, predicate: str, object_: str) -> None:
        cypher = (
            "MATCH (s:Entity {name: $subject})-[r:MEMORY {predicate: $predicate}]->(o:Entity {name: $object}) "
            "SET r.access_count = coalesce(r.access_count, 0) + 1, "
            "r.last_accessed = $ts"
        )
        params = {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, params)

    def bump_vector_access(self, node_id: str) -> None:
        cypher = (
            "MATCH (m:VectorMemory) WHERE elementId(m) = $node_id "
            "SET m.access_count = coalesce(m.access_count, 0) + 1, "
            "m.last_accessed = $ts"
        )
        params = {
            "node_id": str(node_id),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, params)

    def get_entity_scores(self, names: list[str]) -> dict[str, float | None]:
        if not names:
            return {}
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.name IN $names "
            "RETURN e.name AS name, e.value_score AS value_score"
        )
        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, {"names": names})
            out: dict[str, float | None] = {}
            for record in records:
                name = record.get("name")
                value = record.get("value_score")
                try:
                    out[str(name)] = float(value) if value is not None else None
                except (TypeError, ValueError):
                    out[str(name)] = None
            return out
