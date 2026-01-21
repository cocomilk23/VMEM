import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tenacity import Retrying, stop_after_attempt, wait_exponential

from vmem.config import LLMConfig
from vmem.llm.prompts import (
    FACT_EXTRACTION_SYSTEM,
    FACT_EXTRACTION_USER,
    SCORE_AND_TRIPLE_SYSTEM,
    SCORE_AND_TRIPLE_USER,
    ANSWER_SYSTEM,
    ANSWER_USER,
)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("Failed to parse JSON from LLM response")


@dataclass
class LLMClient:
    config: LLMConfig

    def __post_init__(self) -> None:
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
        )

    def extract_facts(self, input_text: str) -> list[str]:
        def _call():
            messages = [
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM},
                {"role": "user", "content": FACT_EXTRACTION_USER.format(input_text=input_text)},
            ]
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            payload = _extract_json(content)
            facts = payload.get("facts", [])
            return [fact.strip() for fact in facts if isinstance(fact, str) and fact.strip()]

        return self._with_retry(_call)

    def score_and_triple(self, fact: str) -> dict[str, Any]:
        def _call():
            messages = [
                {"role": "system", "content": SCORE_AND_TRIPLE_SYSTEM},
                {"role": "user", "content": SCORE_AND_TRIPLE_USER.format(fact=fact)},
            ]
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            payload = _extract_json(content)
            return payload

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
            memories_block = "\n".join(f"- {memory}" for memory in memories)
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": ANSWER_USER.format(question=question, memories=memories_block),
                },
            ]
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.2,
            )
            content = response.choices[0].message.content or ""
            return content.strip()

        return self._with_retry(_call)
