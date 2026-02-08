from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Deque

from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.models import IngestResult, MemoryRecord, VectorMemoryRecord, ProfileMemoryRecord


@dataclass
class MemoryPipeline:
    llm: LLMClient
    embedder: OpenAIEmbedder
    graph: GraphStore
    value_threshold: float = 0.8
    entity_value_threshold: float = 0.7
    buffer_max_turns: int = 10
    buffer_max_tokens: int = 2048
    user_max_tokens: int = 512
    vector_index_name: str = "memory_embedding_index"
    profile_index_name: str = "profile_embedding_index"
    _big_buffer: Deque[dict[str, str]] = field(default_factory=deque, init=False, repr=False)
    _small_buffer_texts: list[str] = field(default_factory=list, init=False, repr=False)
    _small_token_count: int = field(default=0, init=False, repr=False)
    _small_turn_count: int = field(default=0, init=False, repr=False)

    def ingest_text(
        self,
        text: str | list[dict[str, Any]],
        source: str = "user",
        add_vector: bool = True,
        add_graph: bool = False,
        immediate: bool = False,
    ) -> IngestResult:
        return self.ingest_messages(
            text,
            source=source,
            add_vector=add_vector,
            add_graph=add_graph,
            immediate=immediate,
        )

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str | None = None,
        source: str = "user",
        add_vector: bool = True,
        add_graph: bool = False,
        immediate: bool = False,
    ) -> IngestResult:
        payload = [{"role": "user", "context": user_text}]
        if assistant_text:
            payload.append({"role": "assistant", "context": assistant_text})
        return self.ingest_messages(
            payload,
            source=source,
            add_vector=add_vector,
            add_graph=add_graph,
            immediate=immediate,
        )

    def ingest_messages(
        self,
        payload: Any,
        source: str = "user",
        add_vector: bool = True,
        add_graph: bool = False,
        immediate: bool = False,
    ) -> IngestResult:
        if not add_vector and not add_graph:
            return IngestResult([], [], [])
        messages = self._parse_messages(payload)
        if not messages:
            return IngestResult([], [], [])

        self._big_buffer.extend(messages)
        return self._drain_big_buffer(
            source=source, add_vector=add_vector, add_graph=add_graph, force_flush=immediate
        )

    def flush_buffer(
        self,
        source: str = "user",
        add_vector: bool = True,
        add_graph: bool = False,
    ) -> IngestResult:
        if not add_vector and not add_graph:
            return IngestResult([], [], [])
        return self._drain_big_buffer(
            source=source, add_vector=add_vector, add_graph=add_graph, force_flush=True
        )

    def _drain_big_buffer(
        self,
        source: str,
        add_vector: bool,
        add_graph: bool,
        force_flush: bool,
    ) -> IngestResult:
        result = IngestResult([], [], [])
        while self._big_buffer:
            item = self._big_buffer.popleft()
            role = item.get("role", "")
            if role != "user":
                continue
            text = self._truncate_tokens(item.get("context", ""), self.user_max_tokens)
            if not text:
                continue
            token_count = self._estimate_tokens(text)
            if self._small_buffer_texts and (
                self._small_token_count + token_count >= self.buffer_max_tokens
            ):
                result.extend(self._flush_small_buffer(source, add_vector, add_graph))
                self._big_buffer.appendleft(item)
                continue
            if token_count >= self.buffer_max_tokens:
                continue
            self._small_buffer_texts.append(text)
            self._small_turn_count += 1
            self._small_token_count += token_count
            if self._should_flush():
                result.extend(self._flush_small_buffer(source, add_vector, add_graph))

        if force_flush:
            result.extend(self._flush_small_buffer(source, add_vector, add_graph))
        return result

    def ingest_document(
        self,
        payload: Any,
        source: str = "document",
        chunk_tokens: int = 2000,
        add_vector: bool = True,
    ) -> IngestResult:
        if not add_vector:
            return IngestResult([], [], [])
        text = self._load_document_text(payload)
        if not text:
            return IngestResult([], [], [])
        chunks = self._chunk_text(text, max_tokens=chunk_tokens)
        result = IngestResult([], [], [])
        for chunk in chunks:
            result.extend(self._process_batch(chunk, source, add_vector=True, add_graph=False))
        return result

    def _flush_small_buffer(
        self, source: str, add_vector: bool, add_graph: bool
    ) -> IngestResult:
        if not self._small_buffer_texts:
            return IngestResult([], [], [])
        combined = "\n".join(self._small_buffer_texts)
        self._small_buffer_texts.clear()
        self._small_token_count = 0
        self._small_turn_count = 0
        return self._process_batch(combined, source, add_vector, add_graph)

    def _process_batch(
        self, text: str, source: str, add_vector: bool, add_graph: bool
    ) -> IngestResult:
        vector_records: list[VectorMemoryRecord] = []
        graph_records: list[MemoryRecord] = []
        profile_records: list[ProfileMemoryRecord] = []
        now = datetime.now(timezone.utc)

        score_map: dict[str, float] = {}
        profile_score_map: dict[str, float] = {}
        facts: list[str] = self.llm.extract_facts(text) if (add_vector or add_graph) else []
        profile_facts: list[str] = []
        if facts:
            scored = self.llm.score_facts(facts)
            for item in scored:
                fact_text = str(item.get("fact", "")).strip()
                if not fact_text:
                    continue
                try:
                    score_value = float(item.get("value_score", 0.0))
                except (TypeError, ValueError):
                    score_value = 0.0
                score_map[fact_text] = score_value
        if facts and add_vector:
            profile_facts = self.llm.classify_profiles(facts)
        if profile_facts:
            scored = self.llm.score_facts(profile_facts)
            for item in scored:
                fact_text = str(item.get("fact", "")).strip()
                if not fact_text:
                    continue
                try:
                    score_value = float(item.get("value_score", 0.0))
                except (TypeError, ValueError):
                    score_value = 0.0
                profile_score_map[fact_text] = score_value
        time_map: dict[str, str | None] = {}
        if facts:
            time_map = self.llm.extract_fact_times(facts)

        if add_vector and facts:
            embeddings = self.embedder.embed_texts(facts)
            if embeddings:
                self.graph.ensure_vector_index(self.vector_index_name, len(embeddings[0]))
            for fact, embedding in zip(facts, embeddings):
                score = float(score_map.get(fact, 0.0))
                high_value = score >= self.value_threshold
                occurred_at = time_map.get(fact)
                self.graph.write_vector_memory(
                    memory_text=fact,
                    value_score=score,
                    embedding=embedding,
                    created_at=now,
                    source=source,
                    high_value=high_value,
                    occurred_at=occurred_at,
                )
                vector_records.append(
                    VectorMemoryRecord(
                        memory_text=fact,
                        value_score=score,
                        high_value=high_value,
                        embedding=embedding,
                        created_at=now,
                        source=source,
                        occurred_at=occurred_at,
                    )
                )
        if add_vector and profile_facts:
            profile_embeddings = self.embedder.embed_texts(profile_facts)
            if profile_embeddings:
                self.graph.ensure_profile_vector_index(
                    self.profile_index_name, len(profile_embeddings[0])
                )
            for profile, embedding in zip(profile_facts, profile_embeddings):
                score = float(profile_score_map.get(profile, 0.0))
                high_value = score >= self.value_threshold
                self.graph.write_profile_memory(
                    memory_text=profile,
                    value_score=score,
                    embedding=embedding,
                    created_at=now,
                    source=source,
                    high_value=high_value,
                )
                profile_records.append(
                    ProfileMemoryRecord(
                        memory_text=profile,
                        value_score=score,
                        high_value=high_value,
                        embedding=embedding,
                        created_at=now,
                        source=source,
                    )
                )
        if add_graph:
            if not facts:
                return IngestResult(vector_records, graph_records, profile_records)
            for fact in facts:
                entity_types = [
                    "Person",
                    "Organization",
                    "Location",
                    "Event",
                    "Product",
                    "Work",
                    "Other",
                ]
                nodes = self.llm.extract_graph_entities(fact, entity_types=entity_types)
                entity_list = [item["entity"] for item in nodes if item.get("entity")]
                if not entity_list:
                    continue
                relations = self.llm.extract_graph_relations(fact, nodes)
                relation_rows: list[tuple[str, str, str]] = []
                banned_predicates = {
                    "temporal",
                    "time",
                    "date",
                    "participant",
                    "subject",
                    "type",
                    "preference",
                    "event",
                    "attribute",
                    "relation",
                }
                for rel in relations:
                    subject = str(rel.get("source", "")).strip()
                    object_ = str(rel.get("target", "")).strip()
                    keywords = str(rel.get("keywords", "")).strip()
                    if not subject or not object_ or not keywords:
                        continue
                    predicate = keywords.split(",")[0].strip()
                    if not predicate:
                        continue
                    if predicate.lower() in banned_predicates:
                        continue
                    # Skip time/date as entities even if the extractor emits them.
                    if re.search(r"\b(yesterday|today|tomorrow|last\s+\w+)\b", object_.lower()):
                        continue
                    if re.search(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", object_):
                        continue
                    relation_rows.append((subject, predicate, object_))
                if not relation_rows:
                    continue

                entity_scores = self.llm.score_entities(fact, entity_list)
                fact_score = float(score_map.get(fact, 0.0))
                rel_high_value = fact_score >= self.value_threshold
                fact_embedding = self.embedder.embed_texts([fact])[0]

                for subject, predicate, object_ in relation_rows:
                    subj_score = entity_scores.get(subject)
                    obj_score = entity_scores.get(object_)
                    subj_high = (
                        subj_score is not None and subj_score >= self.entity_value_threshold
                    )
                    obj_high = (
                        obj_score is not None and obj_score >= self.entity_value_threshold
                    )
                    occurred_at = time_map.get(fact)
                    for rel in relations:
                        if (
                            str(rel.get("source", "")).strip() == subject
                            and str(rel.get("target", "")).strip() == object_
                            and str(rel.get("keywords", "")).split(",")[0].strip() == predicate
                        ):
                            occurred_at = rel.get("occurred_at") or occurred_at
                            break
                    record = MemoryRecord(
                        subject=subject,
                        predicate=predicate,
                        object=object_,
                        memory_text=fact,
                        value_score=fact_score,
                        high_value=rel_high_value,
                        embedding=fact_embedding,
                        created_at=now,
                        source=source,
                        occurred_at=occurred_at,
                    )
                    self.graph.write_memory(
                        record,
                        subject_score=subj_score,
                        object_score=obj_score,
                        subject_high_value=subj_high,
                        object_high_value=obj_high,
                    )
                    graph_records.append(record)

        return IngestResult(vector_records, graph_records, profile_records)

    def _should_flush(self) -> bool:
        return (
            self._small_turn_count >= self.buffer_max_turns
            or self._small_token_count >= self.buffer_max_tokens
        )

    def _parse_messages(self, payload: Any) -> list[dict[str, str]]:
        if payload is None:
            return []
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return []
            if os.path.exists(stripped) and stripped.lower().endswith(".json"):
                with open(stripped, "r", encoding="utf-8") as handle:
                    return self._parse_messages(json.loads(handle.read()))
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return [{"role": "user", "context": stripped}]
                return self._parse_messages(parsed)
            return [{"role": "user", "context": stripped}]

        if isinstance(payload, dict):
            if "messages" in payload:
                return self._parse_messages(payload.get("messages"))
            message = self._coerce_message(payload)
            return [message] if message else []

        if isinstance(payload, list):
            messages: list[dict[str, str]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                message = self._coerce_message(item)
                if message:
                    messages.append(message)
            return messages
        return []

    def _load_document_text(self, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            raw = payload.strip()
            if not raw:
                return ""
            if os.path.exists(raw):
                if raw.lower().endswith(".json"):
                    with open(raw, "r", encoding="utf-8") as handle:
                        return self._load_document_text(json.loads(handle.read()))
                with open(raw, "r", encoding="utf-8") as handle:
                    return handle.read()
            if raw.startswith("{") or raw.startswith("["):
                try:
                    return self._load_document_text(json.loads(raw))
                except json.JSONDecodeError:
                    return raw
            return raw
        if isinstance(payload, dict):
            if "text" in payload:
                return str(payload.get("text") or "").strip()
            if "content" in payload:
                return str(payload.get("content") or "").strip()
            if "documents" in payload:
                return self._load_document_text(payload.get("documents"))
            if "chunks" in payload:
                return self._load_document_text(payload.get("chunks"))
            return ""
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text).strip())
            return "\n".join(parts)
        return ""

    def _chunk_text(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens <= 0:
            return []
        sentences = self._split_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if sentence_tokens >= max_tokens:
                if current:
                    chunks.append("".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.append(self._truncate_tokens(sentence, max_tokens))
                continue
            if current_tokens + sentence_tokens > max_tokens and current:
                chunks.append("".join(current).strip())
                current = [sentence]
                current_tokens = sentence_tokens
                continue
            current.append(sentence)
            current_tokens += sentence_tokens
        if current:
            chunks.append("".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _split_sentences(self, text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r"(?<=[。！？!?；;])", text)
        return [part for part in (p.strip() for p in parts) if part]

    def _coerce_message(self, item: dict[str, Any]) -> dict[str, str] | None:
        role = item.get("role")
        context = item.get("context") or item.get("content")
        if not role:
            for key in ("user", "assistant", "system"):
                if key in item:
                    role = key
                    context = item.get(key)
                    break
        if not role or context is None:
            return None
        role = str(role).strip().lower()
        context = str(context).strip()
        if not role or not context:
            return None
        if role != "user":
            return None
        return {"role": role, "context": context}

    def _estimate_tokens(self, text: str) -> int:
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        latin = re.findall(r"[A-Za-z0-9_]+", text)
        return cjk + len(latin)

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = 0
        out = []
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                tokens += 1
            elif re.match(r"[A-Za-z0-9_]", ch):
                if out and re.match(r"[A-Za-z0-9_]", out[-1]):
                    pass
                else:
                    tokens += 1
            if tokens > max_tokens:
                break
            out.append(ch)
        return "".join(out).strip()
