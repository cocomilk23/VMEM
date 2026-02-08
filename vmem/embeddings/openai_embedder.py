from dataclasses import dataclass
import logging

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
        self._fallback_client: OpenAI | None = None

    def _embedding_dim(self) -> int:
        model = (self.config.model or "").lower()
        if "3-large" in model:
            return 3072
        if "3-small" in model or "ada-002" in model:
            return 1536
        return 1536

    def _extract_embeddings(self, response) -> list[list[float]] | None:
        if isinstance(response, str):
            return None
        if isinstance(response, dict):
            data = response.get("data") or []
            return [item.get("embedding", []) for item in data if isinstance(item, dict)]
        data = getattr(response, "data", None)
        if data is None:
            return None
        return [item.embedding for item in data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger = logging.getLogger(__name__)
        response = None
        try:
            response = self._client.embeddings.create(
                model=self.config.model,
                input=texts,
            )
        except Exception as exc:
            logger.warning("Embedding request failed; trying fallback client: %s", exc)
        embeddings = self._extract_embeddings(response) if response is not None else None
        if embeddings is None and self.config.base_url:
            if self._fallback_client is None:
                self._fallback_client = OpenAI(
                    api_key=self.config.api_key,
                )
            response = None
            try:
                response = self._fallback_client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                )
            except Exception as exc:
                logger.warning("Fallback embedding request failed: %s", exc)
            embeddings = self._extract_embeddings(response) if response is not None else None
        if embeddings is None or not embeddings:
            logger.warning(
                "Embedding response invalid; using zero vectors. Check OPENAI_BASE_URL and OPENAI_API_KEY."
            )
            dim = self._embedding_dim()
            return [[0.0] * dim for _ in texts]
        return embeddings
