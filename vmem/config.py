import os
from dataclasses import dataclass


def _get_env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key, default)
    if value is None or value == "":
        return default
    return value


def _get_env_str(key: str, default: str) -> str:
    return _get_env(key, default) or default


def _get_env_int(key: str, default: int) -> int:
    raw = _get_env(key, str(default))
    return int(raw) if raw is not None else default


def _get_env_float(key: str, default: float) -> float:
    raw = _get_env(key, str(default))
    return float(raw) if raw is not None else default


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-5-mini"
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: int = 30
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=_get_env_str("VMEM_LLM_MODEL", "gpt-5-mini"),
            api_key=_get_env("OPENAI_API_KEY"),
            base_url=_get_env("OPENAI_BASE_URL"),
            timeout_s=_get_env_int("VMEM_LLM_TIMEOUT_S", 30),
            max_retries=_get_env_int("VMEM_LLM_MAX_RETRIES", 2),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            model=_get_env_str("VMEM_EMBED_MODEL", "text-embedding-3-small"),
            api_key=_get_env("OPENAI_API_KEY"),
            base_url=_get_env("OPENAI_BASE_URL"),
        )


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=_get_env_str("NEO4J_URI", "bolt://localhost:7687"),
            user=_get_env_str("NEO4J_USER", "neo4j"),
            password=_get_env_str("NEO4J_PASSWORD", "neo4j"),
            database=_get_env_str("NEO4J_DATABASE", "neo4j"),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    value_threshold: float = 0.8
    top_k: int = 8
    candidate_k: int = 50
    min_high_value_hits: int = 3
    expand_hops: int = 1
    expand_limit: int = 20
    score_weight_value: float = 0.6
    score_weight_similarity: float = 0.4
    vector_index_name: str = "memory_embedding_index"
    entity_fulltext_index: str = "entity_name_idx"
    answer_similarity_threshold: float = 0.78
    retrieval_mode: str = "entity"

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            value_threshold=_get_env_float("VMEM_VALUE_THRESHOLD", 0.8),
            top_k=_get_env_int("VMEM_TOP_K", 8),
            candidate_k=_get_env_int("VMEM_CANDIDATE_K", 50),
            min_high_value_hits=_get_env_int("VMEM_MIN_HIGH_VALUE_HITS", 3),
            expand_hops=_get_env_int("VMEM_EXPAND_HOPS", 1),
            expand_limit=_get_env_int("VMEM_EXPAND_LIMIT", 20),
            score_weight_value=_get_env_float("VMEM_WEIGHT_VALUE", 0.6),
            score_weight_similarity=_get_env_float("VMEM_WEIGHT_SIM", 0.4),
            vector_index_name=_get_env_str("VMEM_VECTOR_INDEX_NAME", "memory_embedding_index"),
            entity_fulltext_index=_get_env_str("VMEM_ENTITY_FT_INDEX", "entity_name_idx"),
            answer_similarity_threshold=_get_env_float("VMEM_ANSWER_SIM_THRESHOLD", 0.78),
            retrieval_mode=_get_env_str("VMEM_RETRIEVAL_MODE", "entity"),
        )


@dataclass(frozen=True)
class CacheConfig:
    path: str = "vmem_cache.sqlite"
    similarity_threshold: float = 0.92

    @classmethod
    def from_env(cls) -> "CacheConfig":
        return cls(
            path=_get_env_str("VMEM_CACHE_PATH", "vmem_cache.sqlite"),
            similarity_threshold=_get_env_float("VMEM_CACHE_SIM_THRESHOLD", 0.92),
        )


@dataclass(frozen=True)
class ValueMemConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    neo4j: Neo4jConfig
    retrieval: RetrievalConfig
    cache: CacheConfig

    @classmethod
    def from_env(cls) -> "ValueMemConfig":
        return cls(
            llm=LLMConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            cache=CacheConfig.from_env(),
        )
