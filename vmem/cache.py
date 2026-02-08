import sqlite3
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from vmem.config import CacheConfig


@dataclass
class ValueCache:
    config: CacheConfig

    def __post_init__(self) -> None:
        self._conn = sqlite3.connect(self.config.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS value_cache ("
            "fact TEXT PRIMARY KEY, "
            "value_score REAL NOT NULL, "
            "subject TEXT NOT NULL, "
            "predicate TEXT NOT NULL, "
            "object TEXT NOT NULL, "
            "relations_json TEXT"
            ")"
        )
        try:
            self._conn.execute(
                "ALTER TABLE value_cache ADD COLUMN relations_json TEXT"
            )
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, fact: str) -> dict[str, Any] | None:
        cursor = self._conn.execute(
            "SELECT fact, value_score, subject, predicate, object, relations_json "
            "FROM value_cache WHERE fact = ?",
            (fact,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        relations = []
        if row[5]:
            try:
                relations = json.loads(row[5])
            except json.JSONDecodeError:
                relations = []
        return {
            "fact": row[0],
            "value_score": row[1],
            "subject": row[2],
            "predicate": row[3],
            "object": row[4],
            "relations": relations,
        }

    def find_similar(self, fact: str) -> dict[str, Any] | None:
        cursor = self._conn.execute(
            "SELECT fact, value_score, subject, predicate, object, relations_json FROM value_cache"
        )
        best = None
        best_score = 0.0
        for row in cursor.fetchall():
            score = SequenceMatcher(None, fact, row[0]).ratio()
            if score > best_score:
                best_score = score
                best = row
        if best is None or best_score < self.config.similarity_threshold:
            return None
        relations = []
        if best[5]:
            try:
                relations = json.loads(best[5])
            except json.JSONDecodeError:
                relations = []
        return {
            "fact": best[0],
            "value_score": best[1],
            "subject": best[2],
            "predicate": best[3],
            "object": best[4],
            "relations": relations,
            "similarity": best_score,
        }

    def set(self, fact: str, payload: dict[str, Any]) -> None:
        relations = payload.get("relations") or []
        if relations:
            head = relations[0]
            subject = head.get("source") or head.get("subject") or "unknown"
            predicate = head.get("relationship") or head.get("predicate") or "related_to"
            obj = head.get("destination") or head.get("object") or "unknown"
            relations_json = json.dumps(relations, ensure_ascii=False)
        else:
            subject = payload.get("subject") or "unknown"
            predicate = payload.get("predicate") or "related_to"
            obj = payload.get("object") or "unknown"
            relations_json = None
        self._conn.execute(
            "INSERT OR REPLACE INTO value_cache "
            "(fact, value_score, subject, predicate, object, relations_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                fact,
                float(payload["value_score"]),
                subject,
                predicate,
                obj,
                relations_json,
            ),
        )
        self._conn.commit()
