import json
import re
from dataclasses import dataclass
from typing import Any
import logging

from openai import OpenAI
from tenacity import Retrying, stop_after_attempt, wait_exponential

from vmem.config import LLMConfig
from vmem.llm.prompts import (
    answer_system,
    answer_user,
    entity_extraction_system,
    entity_extraction_user,
    entity_value_system,
    entity_value_user,
    fact_time_system,
    fact_time_user,
    fact_extraction_prompt,
    fact_extraction_system,
    profile_classify_prompt,
    profile_classify_system,
    graph_edge_extraction_system,
    graph_edge_extraction_user,
    memory_entity_system,
    memory_entity_user,
    memory_relation_system,
    memory_relation_user,
    memory_value_system,
    memory_value_user,
    score_and_triple_system,
    score_and_triple_user,
)

EXTRACT_ENTITIES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "The name or identifier of the entity."},
                            "entity_type": {"type": "string", "description": "The type or category of the entity."},
                        },
                        "required": ["entity", "entity_type"],
                        "additionalProperties": False,
                    },
                    "description": "An array of entities with their types.",
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

RELATIONS_STRUCT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "establish_relations",
        "description": "Establish relationships among the entities based on the provided text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The source entity of the relationship.",
                            },
                            "relationship": {
                                "type": "string",
                                "description": "The relationship between the source and destination entities.",
                            },
                            "destination": {
                                "type": "string",
                                "description": "The destination entity of the relationship.",
                            },
                        },
                        "required": ["source", "relationship", "destination"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}


def _extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"facts": parsed}

    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.DOTALL)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"facts": parsed}

    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if match:
        parsed = json.loads(match.group(1))
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"facts": parsed}

    raise ValueError(f"Failed to parse JSON from LLM response: {raw[:200]}")


def _is_html(text: str) -> bool:
    return text.lstrip().startswith("<!DOCTYPE html")


def _fallback_facts(text: str) -> list[dict[str, Any]]:
    parts = [p.strip() for p in re.split(r"[。！？!?;；]\s*", text) if p.strip()]
    if not parts:
        parts = [text.strip()] if text.strip() else []
    return [{"fact": part, "value_score": 0.5} for part in parts]


@dataclass
class LLMClient:
    config: LLMConfig

    def __post_init__(self) -> None:
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
        )
        self._fallback_client: OpenAI | None = None

    def _response_content(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {}) or {}
                return message.get("content") or ""
            return response.get("output_text") or ""
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                return message.content or ""
        output_text = getattr(response, "output_text", None)
        return output_text or ""

    def _create_chat(
        self,
        client: OpenAI,
        messages: list[dict[str, str]],
        json_only: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
    ) -> Any:
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if json_only:
            params["response_format"] = {"type": "json_object"}
        try:
            return client.chat.completions.create(**params)
        except Exception as exc:
            if json_only and "response_format" in str(exc):
                params.pop("response_format", None)
                return client.chat.completions.create(**params)
            raise

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        json_only: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
    ) -> Any:
        response = self._create_chat(
            self._client, messages, json_only, tools=tools, tool_choice=tool_choice
        )
        content = self._response_content(response).lstrip()
        if content.startswith("<!DOCTYPE html") and self.config.base_url:
            if self._fallback_client is None:
                self._fallback_client = OpenAI(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout_s,
                )
            response = self._create_chat(
                self._fallback_client,
                messages,
                json_only,
                tools=tools,
                tool_choice=tool_choice,
            )
        return response

    def _response_tool_calls(self, response: Any) -> list[Any]:
        # Supports OpenAI SDK objects and dict responses.
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if not choices:
                return []
            message = choices[0].get("message", {}) or {}
            return message.get("tool_calls") or []
        choices = getattr(response, "choices", None)
        if not choices:
            return []
        message = getattr(choices[0], "message", None)
        if message is None:
            return []
        tool_calls = getattr(message, "tool_calls", None)
        return tool_calls or []

    def extract_facts(self, input_text: str) -> list[str]:
        def _call():
            logger = logging.getLogger(__name__)
            from datetime import datetime
            messages = [
                {"role": "system", "content": fact_extraction_system()},
                {
                    "role": "user",
                    "content": fact_extraction_prompt(
                        today=datetime.now().strftime("%Y-%m-%d"),
                        input_text=input_text,
                    ),
                },
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using fallback fact extraction. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                    return _fallback_facts(input_text)
                logger.warning("LLM JSON parse failed; using fallback fact extraction: %s", exc)
                return [item["fact"] for item in _fallback_facts(input_text)]
            facts = payload.get("facts", [])
            results: list[str] = []
            if isinstance(facts, list):
                for item in facts:
                    if isinstance(item, str):
                        value = item.strip()
                        if value:
                            results.append(value)
                    elif isinstance(item, dict):
                        value = str(item.get("fact", "")).strip()
                        if value:
                            results.append(value)
            return results

        return self._with_retry(_call)

    def classify_profiles(self, facts: list[str]) -> list[str]:
        if not facts:
            return []
        def _call():
            logger = logging.getLogger(__name__)
            from datetime import datetime
            facts_json = json.dumps({"facts": facts}, ensure_ascii=False)
            messages = [
                {"role": "system", "content": profile_classify_system()},
                {
                    "role": "user",
                    "content": profile_classify_prompt(
                        today=datetime.now().strftime("%Y-%m-%d"),
                        facts_json=facts_json,
                    ),
                },
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using empty profile classification. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                    return []
                logger.warning("LLM JSON parse failed; using empty profile classification: %s", exc)
                return []
            profiles = payload.get("profiles", [])
            results: list[str] = []
            if isinstance(profiles, list):
                for item in profiles:
                    if isinstance(item, str):
                        value = item.strip()
                        if value:
                            results.append(value)
            return results

        return self._with_retry(_call)

    def score_and_triple(self, fact: str) -> dict[str, Any]:
        def _call():
            logger = logging.getLogger(__name__)
            messages = [
                {"role": "system", "content": score_and_triple_system()},
                {"role": "user", "content": score_and_triple_user(fact)},
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using fallback triple parsing. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                else:
                    logger.warning("LLM JSON parse failed; using fallback triple parsing: %s", exc)
                payload = {"subject": "unknown", "predicate": "related_to", "object": "unknown"}
            return payload

        return self._with_retry(_call)

    def score_facts(self, facts: list[str]) -> list[dict[str, Any]]:
        if not facts:
            return []

        def _call():
            logger = logging.getLogger(__name__)
            memory_json = json.dumps({"facts": facts}, ensure_ascii=False)
            messages = [
                {"role": "system", "content": memory_value_system()},
                {"role": "user", "content": memory_value_user(memory_json)},
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using fallback memory scoring. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                else:
                    logger.warning("LLM JSON parse failed; using fallback memory scoring: %s", exc)
                return [{"fact": fact, "value_score": 0.0} for fact in facts]
            items = payload.get("facts", [])
            results: list[dict[str, Any]] = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        fact_text = str(item.get("fact", "")).strip()
                        if not fact_text:
                            continue
                        try:
                            score_value = float(item.get("value_score", 0.0))
                        except (TypeError, ValueError):
                            score_value = 0.0
                        results.append({"fact": fact_text, "value_score": score_value})
            if not results:
                return [{"fact": fact, "value_score": 0.0} for fact in facts]
            return results

        return self._with_retry(_call)

    def extract_fact_times(self, facts: list[str]) -> dict[str, str | None]:
        if not facts:
            return {}

        def _call():
            logger = logging.getLogger(__name__)
            from datetime import datetime, timezone

            reference_time = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            facts_json = json.dumps({"facts": facts}, ensure_ascii=False)
            messages = [
                {"role": "system", "content": fact_time_system()},
                {"role": "user", "content": fact_time_user(facts_json, reference_time)},
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using empty time extraction. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                else:
                    logger.warning("LLM JSON parse failed; using empty time extraction: %s", exc)
                return {}

            items = payload.get("facts", [])
            out: dict[str, str | None] = {}
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    fact_text = str(item.get("fact", "")).strip()
                    if not fact_text:
                        continue
                    occurred_at = item.get("occurred_at")
                    occurred_at_str = str(occurred_at).strip() if occurred_at is not None else ""
                    out[fact_text] = occurred_at_str or None
            return out

        return self._with_retry(_call)

    def extract_memory_entities(self, fact: str) -> list[dict[str, str]]:
        def _call():
            logger = logging.getLogger(__name__)
            messages = [
                {"role": "system", "content": memory_entity_system()},
                {"role": "user", "content": memory_entity_user(fact)},
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using fallback entity extraction. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                else:
                    logger.warning("LLM JSON parse failed; using fallback entity extraction: %s", exc)
                return [{"entity": "USER", "entity_type": "person"}]
            entities = payload.get("entities", [])
            results: list[dict[str, str]] = []
            if isinstance(entities, list):
                for item in entities:
                    if not isinstance(item, dict):
                        continue
                    entity = str(item.get("entity", "")).strip()
                    entity_type = str(item.get("entity_type", "other")).strip().lower()
                    if entity:
                        results.append({"entity": entity, "entity_type": entity_type or "other"})
            return results

        return self._with_retry(_call)

    def extract_memory_relations(self, fact: str, entities: list[str]) -> list[dict[str, str]]:
        def _call():
            logger = logging.getLogger(__name__)
            messages = [
                {"role": "system", "content": memory_relation_system()},
                {
                    "role": "user",
                    "content": memory_relation_user(fact, ", ".join(entities)),
                },
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using fallback relation extraction. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                else:
                    logger.warning("LLM JSON parse failed; using fallback relation extraction: %s", exc)
                return []
            relations = payload.get("relations", [])
            results: list[dict[str, str]] = []
            if isinstance(relations, list):
                for item in relations:
                    if not isinstance(item, dict):
                        continue
                    source = str(item.get("source", "")).strip()
                    relationship = str(item.get("relationship", "")).strip()
                    destination = str(item.get("destination", "")).strip()
                    if source and relationship and destination:
                        results.append(
                            {
                                "source": source,
                                "relationship": relationship,
                                "destination": destination,
                            }
                        )
            return results

        return self._with_retry(_call)

    def score_entities(self, fact: str, entities: list[str]) -> dict[str, float]:
        if not entities:
            return {}

        def _call():
            logger = logging.getLogger(__name__)
            messages = [
                {"role": "system", "content": entity_value_system()},
                {"role": "user", "content": entity_value_user(fact, ", ".join(entities))},
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using fallback entity scoring. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                else:
                    logger.warning("LLM JSON parse failed; using fallback entity scoring: %s", exc)
                return {}
            items = payload.get("entities", [])
            scores: dict[str, float] = {}
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("entity", "")).strip()
                    if not name:
                        continue
                    try:
                        score = float(item.get("score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    scores[name] = score
            return scores

        return self._with_retry(_call)

    def extract_entities(self, query: str) -> list[str]:
        def _call():
            logger = logging.getLogger(__name__)
            messages = [
                {"role": "system", "content": entity_extraction_system()},
                {"role": "user", "content": entity_extraction_user(query)},
            ]
            response = self._chat_completion(messages, json_only=True)
            content = self._response_content(response) or "{}"
            try:
                payload = _extract_json(content)
            except ValueError as exc:
                if _is_html(content):
                    logger.warning(
                        "LLM returned HTML instead of JSON; using fallback entity extraction. "
                        "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                    )
                else:
                    logger.warning("LLM JSON parse failed; using fallback entity extraction: %s", exc)
                return [query.strip()] if query.strip() else []
            entities = payload.get("entities", [])
            results: list[str] = []
            if isinstance(entities, list):
                for item in entities:
                    if isinstance(item, str):
                        value = item.strip()
                        if value:
                            results.append(value)
            if not results and query.strip():
                results = [query.strip()]
            return results

        return self._with_retry(_call)

    def extract_graph_entities(self, fact: str, entity_types: list[str]) -> list[dict[str, str]]:
        def _call():
            logger = logging.getLogger(__name__)
            system_content = (
                "You are a smart assistant who understands entities and their types in a given text. "
                "If user message contains self reference such as 'I', 'me', 'my' etc. then use USER as the source entity. "
                "Extract all the entities from the text. "
                "***DO NOT*** answer the question itself if the given text is a question. "
                "\n\nVMEM constraints:\n"
                "- Do NOT extract dates, times, temporal expressions, or frequencies as entities (e.g., 2022, yesterday, every evening).\n"
                "- Do NOT extract verbs, verb phrases, or action phrases as entities (e.g., learning the piano, attempted to cross). "
                "Actions should be represented as relationships/predicates.\n"
                "- If a relative/role is mentioned without a name (e.g., 'my younger sister'), use a stable entity name like "
                "\"USER's younger sister\".\n"
                "Output MUST be provided via the tool call `extract_entities`."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": fact},
            ]

            try:
                response = self._chat_completion(
                    messages,
                    json_only=False,
                    tools=[EXTRACT_ENTITIES_TOOL],
                    tool_choice={"type": "function", "function": {"name": "extract_entities"}},
                )
                tool_calls = self._response_tool_calls(response)
                results: list[dict[str, str]] = []
                for call in tool_calls:
                    if isinstance(call, dict):
                        fn = call.get("function") or {}
                        name = fn.get("name")
                        args = fn.get("arguments")
                    else:
                        fn = getattr(call, "function", None)
                        name = getattr(fn, "name", None) if fn is not None else None
                        args = getattr(fn, "arguments", None) if fn is not None else None
                    if name != "extract_entities":
                        continue
                    if isinstance(args, str):
                        try:
                            payload = json.loads(args)
                        except json.JSONDecodeError:
                            payload = {}
                    elif isinstance(args, dict):
                        payload = args
                    else:
                        payload = {}
                    items = payload.get("entities", [])
                    if isinstance(items, list):
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            ent = str(item.get("entity", "")).strip()
                            if not ent:
                                continue
                            etype = str(item.get("entity_type", "Other")).strip() or "Other"
                            results.append({"entity": ent, "entity_type": etype})
                if results:
                    return results
            except Exception as exc:
                logger.warning("Tool-based entity extraction failed; returning empty entities: %s", exc)
                return []

            return []

        return self._with_retry(_call)

    def extract_graph_relations(self, fact: str, nodes: list[dict[str, str]]) -> list[dict[str, str]]:
        if not nodes:
            return []

        def _call():
            logger = logging.getLogger(__name__)
            nodes_json = json.dumps(nodes, ensure_ascii=False)
            from datetime import datetime, timezone
            reference_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            messages = [
                {"role": "system", "content": graph_edge_extraction_system()},
                {"role": "user", "content": graph_edge_extraction_user(fact, nodes_json, reference_time)},
            ]

            # Tool-based relation extraction (mem0-style). Do not fall back to JSON parsing.
            try:
                response = self._chat_completion(
                    messages,
                    json_only=False,
                    tools=[RELATIONS_STRUCT_TOOL],
                    tool_choice={"type": "function", "function": {"name": "establish_relations"}},
                )
                tool_calls = self._response_tool_calls(response)
                results: list[dict[str, str]] = []
                for call in tool_calls:
                    if isinstance(call, dict):
                        fn = call.get("function") or {}
                        name = fn.get("name")
                        args = fn.get("arguments")
                    else:
                        fn = getattr(call, "function", None)
                        name = getattr(fn, "name", None) if fn is not None else None
                        args = getattr(fn, "arguments", None) if fn is not None else None
                    if name != "establish_relations":
                        continue
                    if isinstance(args, str):
                        try:
                            payload = json.loads(args)
                        except json.JSONDecodeError:
                            payload = {}
                    elif isinstance(args, dict):
                        payload = args
                    else:
                        payload = {}
                    items = payload.get("entities", [])
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        source = str(item.get("source", "")).strip()
                        relationship = str(item.get("relationship", "")).strip()
                        destination = str(item.get("destination", "")).strip()
                        if not source or not destination or not relationship:
                            continue
                        # Keep the downstream schema stable: pipeline expects (source, target, keywords).
                        results.append(
                            {
                                "source": source,
                                "target": destination,
                                "keywords": relationship,
                                "occurred_at": None,
                            }
                        )
                return results
            except Exception as exc:
                logger.warning("Tool-based edge extraction failed; returning empty edges: %s", exc)
                return []

        return self._with_retry(_call)

    def _with_retry(self, fn):
        retries = max(int(self.config.max_retries), 1)
        for attempt in Retrying(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(min=1, max=4),
            reraise=True,
        ):
            with attempt:
                return fn()
        return fn()

    def answer_question(self, question: str, memories: list[str]) -> str:
        def _call():
            logger = logging.getLogger(__name__)
            memories_block = "\n".join(f"- {memory}" for memory in memories)
            messages = [
                {"role": "system", "content": answer_system()},
                {
                    "role": "user",
                    "content": answer_user(question, memories_block),
                },
            ]
            response = self._chat_completion(messages, json_only=False)
            content = self._response_content(response) or ""
            if _is_html(content):
                logger.warning(
                    "LLM returned HTML instead of text; returning 'insufficient'. "
                    "Check OPENAI_BASE_URL and OPENAI_API_KEY."
                )
                return "insufficient"
            return content.strip()

        return self._with_retry(_call)
