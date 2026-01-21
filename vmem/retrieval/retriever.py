from dataclasses import dataclass

from vmem.config import RetrievalConfig
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore


@dataclass
class MemoryRetriever:
    graph: GraphStore
    embedder: OpenAIEmbedder
    config: RetrievalConfig

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.embedder.embed_texts([query])[0]

        high_value = self._query_vector(
            query_vec, value_threshold=self.config.value_threshold
        )
        scored = self._score_candidates(high_value)
        results = scored[: self.config.top_k]

        if results and self.config.expand_hops > 0:
            nodes = self._collect_nodes(results)
            neighbors = self.graph.expand_neighbors(
                node_names=nodes,
                hops=self.config.expand_hops,
                limit=self.config.expand_limit,
            )
            neighbors = self._attach_similarity(neighbors, query_vec)
            neighbors_scored = self._score_candidates(neighbors)
            results = self._merge_unique(results, neighbors_scored)[: self.config.top_k]

        if not self._is_answer_ready(results):
            fallback = self._query_vector(query_vec, value_threshold=None)
            fallback_scored = self._score_candidates(fallback)
            results = self._merge_unique(results, fallback_scored)[: self.config.top_k]

        for item in results:
            self.graph.bump_access(item["subject"], item["predicate"], item["object"])

        return results

    def _query_vector(self, query_vec: list[float], value_threshold: float | None) -> list[dict]:
        try:
            return self.graph.query_vector(
                embedding=query_vec,
                limit=self.config.candidate_k,
                index_name=self.config.vector_index_name,
                value_threshold=value_threshold,
            )
        except Exception:
            candidates = self.graph.query_memories(
                value_threshold=value_threshold,
                limit=self.config.candidate_k,
            )
            return self._attach_similarity(candidates, query_vec)

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

    def _score_candidates(self, candidates: list[dict]) -> list[dict]:
        scored = []
        for candidate in candidates:
            similarity = float(candidate.get("similarity") or 0.0)
            value_score = float(candidate.get("value_score") or 0.0)
            combined = (
                self.config.score_weight_value * value_score
                + self.config.score_weight_similarity * similarity
            )
            candidate = dict(candidate)
            candidate["combined_score"] = combined
            scored.append(candidate)
        scored.sort(key=lambda item: item["combined_score"], reverse=True)
        return scored

    def _merge_unique(self, base: list[dict], extra: list[dict]) -> list[dict]:
        seen = set()
        merged = []
        for item in base + extra:
            key = (item["subject"], item["predicate"], item["object"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        merged.sort(key=lambda item: item.get("combined_score", 0.0), reverse=True)
        return merged

    def _collect_nodes(self, items: list[dict]) -> list[str]:
        nodes = set()
        for item in items:
            nodes.add(item["subject"])
            nodes.add(item["object"])
        return list(nodes)

    def _is_answer_ready(self, items: list[dict]) -> bool:
        if len(items) < self.config.min_high_value_hits:
            return False
        best = items[0]
        return float(best.get("similarity") or 0.0) >= self.config.answer_similarity_threshold
