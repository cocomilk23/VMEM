from dataclasses import dataclass
from datetime import datetime, timezone

from neo4j import GraphDatabase

from vmem.config import Neo4jConfig
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
        cypher = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT rel_predicate_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.predicate IS NOT NULL",
            "CREATE CONSTRAINT rel_value_score_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.value_score IS NOT NULL",
            "CREATE CONSTRAINT rel_memory_text_exists IF NOT EXISTS FOR ()-[r:MEMORY]-() REQUIRE r.memory_text IS NOT NULL",
            "CREATE INDEX rel_value_score IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.value_score)",
            "CREATE INDEX rel_created_at IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.created_at)",
            "CREATE INDEX rel_predicate IF NOT EXISTS FOR ()-[r:MEMORY]-() ON (r.predicate)",
        ]
        with self._driver.session(database=self.config.database) as session:
            for stmt in cypher:
                session.run(stmt)

    def write_memory(self, record: MemoryRecord) -> None:
        created_at = record.created_at.astimezone(timezone.utc).isoformat()
        cypher = (
            "MERGE (s:Entity {name: $subject}) "
            "MERGE (o:Entity {name: $object}) "
            "MERGE (s)-[r:MEMORY {predicate: $predicate, memory_text: $memory_text}]->(o) "
            "SET r.value_score = $value_score, "
            "r.embedding = $embedding, "
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
            "embedding": record.embedding,
            "created_at": created_at,
            "source": record.source,
        }
        with self._driver.session(database=self.config.database) as session:
            session.run(cypher, params)

    def query_memories(self, value_threshold: float | None, limit: int) -> list[dict]:
        if value_threshold is None:
            cypher = (
                "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding "
                "ORDER BY r.value_score DESC LIMIT $limit"
            )
            params = {"limit": limit}
        else:
            cypher = (
                "MATCH (s:Entity)-[r:MEMORY]->(o:Entity) "
                "WHERE r.value_score >= $threshold "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding "
                "ORDER BY r.value_score DESC LIMIT $limit"
            )
            params = {"threshold": float(value_threshold), "limit": limit}

        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, params)
            return [record.data() for record in records]

    def query_vector(
        self,
        embedding: list[float],
        limit: int,
        index_name: str,
        value_threshold: float | None = None,
    ) -> list[dict]:
        if value_threshold is None:
            cypher = (
                "CALL db.index.vector.queryRelationships($index, $limit, $embedding) "
                "YIELD relationship AS r, score "
                "MATCH (s:Entity)-[r]->(o:Entity) "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding, "
                "score AS similarity"
            )
            params = {"index": index_name, "limit": limit, "embedding": embedding}
        else:
            cypher = (
                "CALL db.index.vector.queryRelationships($index, $limit, $embedding) "
                "YIELD relationship AS r, score "
                "WHERE r.value_score >= $threshold "
                "MATCH (s:Entity)-[r]->(o:Entity) "
                "RETURN s.name AS subject, r.predicate AS predicate, o.name AS object, "
                "r.memory_text AS memory_text, r.value_score AS value_score, r.embedding AS embedding, "
                "score AS similarity"
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
