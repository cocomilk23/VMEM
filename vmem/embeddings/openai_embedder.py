from dataclasses import dataclass

from openai import OpenAI

from vmem.config import EmbeddingConfig


@dataclass
class OpenAIEmbedder:
    config: EmbeddingConfig

    def __post_init__(self) -> None:
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self.config.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
