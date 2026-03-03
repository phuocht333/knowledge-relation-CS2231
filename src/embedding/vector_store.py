import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized vectors)
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> None:
        assert len(embeddings) == len(metadata_list)
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = {**self.metadata[idx], "score": float(score)}
            results.append(result)
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta_path = directory / "metadata.json"
        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Vector store saved: {self.index.ntotal} vectors")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        index = faiss.read_index(str(directory / "index.faiss"))
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        store = cls(dim=index.d)
        store.index = index
        store.metadata = metadata
        print(f"Vector store loaded: {store.index.ntotal} vectors")
        return store
