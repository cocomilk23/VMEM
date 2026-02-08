from dataclasses import dataclass
import os

from vmem.config import EmbeddingConfig


@dataclass
class LocalEmbedder:
    config: EmbeddingConfig

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        if not os.getenv("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        device = os.getenv("VMEM_EMBED_DEVICE", "cuda")
        self._model = SentenceTransformer(self.config.model, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]
