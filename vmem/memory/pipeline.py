from dataclasses import dataclass
from datetime import datetime, timezone

from vmem.cache import ValueCache
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.models import MemoryRecord
from vmem.utils import normalize_text


@dataclass
class MemoryPipeline:
    llm: LLMClient
    embedder: OpenAIEmbedder
    graph: GraphStore
    cache: ValueCache

    def ingest_text(self, text: str, source: str = "user") -> list[MemoryRecord]:
        facts = self.llm.extract_facts(text)
        if not facts:
            return []

        records: list[MemoryRecord] = []
        to_embed: list[str] = []
        payloads: list[dict] = []

        for fact in facts:
            normalized = normalize_text(fact)
            cached = self.cache.get(normalized)
            if cached is None:
                cached = self.cache.find_similar(normalized)
            if cached is None:
                payload = self.llm.score_and_triple(fact)
                payload = {
                    "value_score": float(payload.get("value_score", 0.0)),
                    "subject": str(payload.get("subject", "")).strip() or "unknown",
                    "predicate": str(payload.get("predicate", "")).strip() or "related_to",
                    "object": str(payload.get("object", "")).strip() or "unknown",
                }
                self.cache.set(normalized, payload)
            else:
                payload = {
                    "value_score": float(cached["value_score"]),
                    "subject": cached["subject"],
                    "predicate": cached["predicate"],
                    "object": cached["object"],
                }
            to_embed.append(fact)
            payloads.append(payload)

        embeddings = self.embedder.embed_texts(to_embed)
        now = datetime.now(timezone.utc)
        for fact, embedding, payload in zip(to_embed, embeddings, payloads):
            record = MemoryRecord(
                subject=payload["subject"],
                predicate=payload["predicate"],
                object=payload["object"],
                memory_text=fact,
                value_score=payload["value_score"],
                embedding=embedding,
                created_at=now,
                source=source,
            )
            self.graph.write_memory(record)
            records.append(record)

        return records
