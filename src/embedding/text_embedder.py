import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM


class TextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)
