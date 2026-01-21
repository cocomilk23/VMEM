import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from vmem.config import (
    CacheConfig,
    EmbeddingConfig,
    LLMConfig,
    Neo4jConfig,
    RetrievalConfig,
    ValueMemConfig,
)


def _load_yaml(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping")
    return data


def _overlay(base, updates: dict[str, Any], keys: list[str]):
    values = {}
    for key in keys:
        if key in updates:
            values[key] = _resolve_value(updates[key])
    return replace(base, **values) if values else base


def _resolve_value(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            env_key = value[2:-1]
            return os.getenv(env_key, "")
        if value.startswith("$"):
            env_key = value[1:]
            return os.getenv(env_key, "")
    return value


def build_config(config_path: str | None = None) -> ValueMemConfig:
    if config_path is None:
        config_path = None
    if not config_path:
        return ValueMemConfig.from_env()

    raw = _load_yaml(config_path)
    llm = _overlay(LLMConfig(), raw.get("llm", {}), ["model", "api_key", "base_url", "timeout_s", "max_retries"])
    embedding = _overlay(
        EmbeddingConfig(), raw.get("embedding", {}), ["model", "api_key", "base_url"]
    )
    neo4j = _overlay(Neo4jConfig(), raw.get("neo4j", {}), ["uri", "user", "password", "database"])
    retrieval = _overlay(
        RetrievalConfig(),
        raw.get("retrieval", {}),
        [
            "value_threshold",
            "top_k",
            "candidate_k",
            "min_high_value_hits",
            "expand_hops",
            "expand_limit",
            "score_weight_value",
            "score_weight_similarity",
            "vector_index_name",
            "answer_similarity_threshold",
        ],
    )
    cache = _overlay(CacheConfig(), raw.get("cache", {}), ["path", "similarity_threshold"])
    return ValueMemConfig(llm=llm, embedding=embedding, neo4j=neo4j, retrieval=retrieval, cache=cache)
