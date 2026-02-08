import math
import re
from typing import Iterable, Any


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_query_and_history(
    text: Any, max_turns: int = 10
) -> tuple[str, list[str]]:
    messages: list[dict[str, str]] = []
    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                payload = __import__("json").loads(stripped)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                payload = payload.get("messages")
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    context = item.get("context") or item.get("content")
                    if not role:
                        for key in ("user", "assistant", "system"):
                            if key in item:
                                role = key
                                context = item.get(key)
                                break
                    if not role or context is None:
                        continue
                    role = str(role).strip()
                    context = str(context).strip()
                    if role and context:
                        messages.append({"role": role, "context": context})
    elif isinstance(text, list):
        for item in text:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            context = item.get("context") or item.get("content")
            if not role:
                for key in ("user", "assistant", "system"):
                    if key in item:
                        role = key
                        context = item.get(key)
                        break
            if not role or context is None:
                continue
            role = str(role).strip()
            context = str(context).strip()
            if role and context:
                messages.append({"role": role, "context": context})

    if not messages:
        return str(text).strip(), []

    # build turns and find last user message
    turns: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "user":
            if current:
                turns.append(current)
            current = [msg]
        else:
            if not current:
                current = [msg]
            else:
                current.append(msg)
    if current:
        turns.append(current)

    # last user message is the query
    query = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            query = msg["context"]
            break
    if not query:
        query = messages[-1]["context"]

    history_turns = turns[:-1][-max_turns:]
    history_lines = [
        f'{msg["role"]}: {msg["context"]}'
        for turn in history_turns
        for msg in turn
    ]
    return query, history_lines
