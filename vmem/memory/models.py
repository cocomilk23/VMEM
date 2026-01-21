from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryRecord:
    subject: str
    predicate: str
    object: str
    memory_text: str
    value_score: float
    embedding: list[float]
    created_at: datetime
    source: str = "user"
    access_count: int = 0
