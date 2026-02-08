from dataclasses import dataclass

from vmem.config import RetrievalConfig
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient


@dataclass
class MemoryRetriever:
    graph: GraphStore
    embedder: OpenAIEmbedder
    llm: LLMClient
    config: RetrievalConfig

    def retrieve(self, query: str) -> list[dict]:
        mode = (self.config.retrieval_mode or "entity").lower()
        if mode == "vector":
            results = self._retrieve_vector(query)
        elif mode == "hybrid":
            results = self._retrieve_hybrid(query)
        else:
            results = self._retrieve_entity(query)

        for item in results:
            record_type = item.get("record_type", "graph")
            if record_type == "graph":
                self.graph.bump_access(item["subject"], item["predicate"], item["object"])
            elif record_type == "vector":
                node_id = item.get("node_id")
                if node_id is not None:
                    self.graph.bump_vector_access(node_id)

        return results

    def _retrieve_vector(self, query: str) -> list[dict]:
        query_vec = self.embedder.embed_texts([query])[0]
        high_value = self._query_vector(
            query_vec, value_threshold=self.config.value_threshold
        )
        high_value = self._filter_by_similarity(high_value)
        if high_value:
            return self._rank_by_similarity(high_value)[: self.config.top_k]
        fallback = self._query_vector(
            query_vec,
            value_threshold=self.config.value_threshold,
            only_low_value=True,
        )
        fallback = self._filter_by_similarity(fallback)
        return self._rank_by_similarity(fallback)[: self.config.top_k]

    def _retrieve_entity(self, query: str) -> list[dict]:
        entities = self.llm.extract_entities(query)
        if not entities:
            return self._retrieve_vector(query)
        query_vec = self.embedder.embed_texts([query])[0]
        high_value = self.graph.query_by_entities(
            entities=entities,
            value_threshold=self.config.value_threshold,
            limit=self.config.candidate_k,
        )
        high_value = self._mark_type(self._attach_similarity(high_value, query_vec), "graph")
        high_value = self._filter_by_similarity(high_value)
        if high_value:
            return self._rank_by_similarity(high_value)[: self.config.top_k]
        fallback = self.graph.query_by_entities(
            entities=entities,
            value_threshold=None,
            limit=self.config.candidate_k,
        )
        fallback = self._mark_type(self._attach_similarity(fallback, query_vec), "graph")
        fallback = self._filter_by_similarity(fallback)
        if fallback:
            return self._rank_by_similarity(fallback)[: self.config.top_k]
        # Fuzzy entity name fallback (fulltext index).
        fuzzy_names = self.graph.query_entity_names_fuzzy(
            entities=entities,
            index_name=self.config.entity_fulltext_index,
            limit=self.config.candidate_k,
        )
        if fuzzy_names:
            fuzzy_high = self.graph.query_by_entities(
                entities=fuzzy_names,
                value_threshold=self.config.value_threshold,
                limit=self.config.candidate_k,
            )
            fuzzy_high = self._mark_type(self._attach_similarity(fuzzy_high, query_vec), "graph")
            fuzzy_high = self._filter_by_similarity(fuzzy_high)
            if fuzzy_high:
                return self._rank_by_similarity(fuzzy_high)[: self.config.top_k]
            fuzzy_all = self.graph.query_by_entities(
                entities=fuzzy_names,
                value_threshold=None,
                limit=self.config.candidate_k,
            )
            fuzzy_all = self._mark_type(self._attach_similarity(fuzzy_all, query_vec), "graph")
            fuzzy_all = self._filter_by_similarity(fuzzy_all)
            if fuzzy_all:
                return self._rank_by_similarity(fuzzy_all)[: self.config.top_k]
        return self._retrieve_vector(query)

    def _retrieve_hybrid(self, query: str) -> list[dict]:
        entity_results = self._retrieve_entity(query)
        vector_results = self._retrieve_vector(query)
        merged = self._merge_unique(entity_results, vector_results)
        return self._rank_by_similarity(merged)[: self.config.top_k]

    def _merge_unique(self, base: list[dict], extra: list[dict]) -> list[dict]:
        seen = set()
        merged = []
        for item in base + extra:
            record_type = item.get("record_type")
            node_id = item.get("node_id")
            if record_type == "vector":
                key = (record_type, node_id or item.get("memory_text"))
            else:
                key = (record_type, item.get("subject"), item.get("predicate"), item.get("object"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _query_vector(
        self,
        query_vec: list[float],
        value_threshold: float | None,
        only_low_value: bool = False,
    ) -> list[dict]:
        try:
            results = self.graph.query_vector(
                embedding=query_vec,
                limit=self.config.candidate_k,
                index_name=self.config.vector_index_name,
                value_threshold=value_threshold,
                only_low_value=only_low_value,
            )
            return self._mark_type(results, "vector")
        except Exception:
            candidates = self.graph.query_vector_memories(
                value_threshold=value_threshold,
                limit=self.config.candidate_k,
                only_low_value=only_low_value,
            )
            return self._mark_type(self._attach_similarity(candidates, query_vec), "vector")

    def _attach_similarity(self, candidates: list[dict], query_vec: list[float]) -> list[dict]:
        from vmem.utils import cosine_similarity

        enriched = []
        for candidate in candidates:
            embedding = candidate.get("embedding")
            similarity = cosine_similarity(query_vec, embedding) if embedding else 0.0
            candidate = dict(candidate)
            candidate["similarity"] = similarity
            enriched.append(candidate)
        return enriched

    def _filter_by_similarity(self, candidates: list[dict]) -> list[dict]:
        filtered = []
        for candidate in candidates:
            similarity = float(candidate.get("similarity") or 0.0)
            if similarity >= self.config.answer_similarity_threshold:
                filtered.append(candidate)
        return filtered

    def _rank_by_similarity(self, candidates: list[dict]) -> list[dict]:
        ranked = [dict(item) for item in candidates]
        ranked.sort(key=lambda item: float(item.get("similarity") or 0.0), reverse=True)
        return ranked

    def _mark_type(self, candidates: list[dict], record_type: str) -> list[dict]:
        return [dict(item, record_type=record_type) for item in candidates]
