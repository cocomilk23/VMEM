from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryRecord:
    subject: str
    predicate: str
    object: str
    memory_text: str
    value_score: float
    high_value: bool
    embedding: list[float]
    created_at: datetime
    source: str = "user"
    access_count: int = 0
    # Optional: when the memory happened (human text or ISO 8601 UTC). If not mentioned, keep None.
    occurred_at: str | None = None


@dataclass
class VectorMemoryRecord:
    memory_text: str
    value_score: float
    high_value: bool
    embedding: list[float]
    created_at: datetime
    source: str = "user"
    access_count: int = 0
    occurred_at: str | None = None


@dataclass
class ProfileMemoryRecord:
    memory_text: str
    value_score: float
    high_value: bool
    embedding: list[float]
    created_at: datetime
    source: str = "user"
    access_count: int = 0


@dataclass
class IngestResult:
    vector_records: list[VectorMemoryRecord]
    graph_records: list[MemoryRecord]
    profile_records: list[ProfileMemoryRecord]

    def extend(self, other: "IngestResult") -> None:
        self.vector_records.extend(other.vector_records)
        self.graph_records.extend(other.graph_records)
        self.profile_records.extend(other.profile_records)
