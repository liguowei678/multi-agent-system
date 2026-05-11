import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from config.settings import QDRANT_URL, QDRANT_COLLECTION, BGE_MODEL_PATH


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.model: Optional[SentenceTransformer] = None
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(BGE_MODEL_PATH)
        return self.model

    def _embed(self, text: str) -> list[float]:
        return self._get_model().encode(text).tolist()

    def add(self, memory_id: str, text: str, metadata: dict) -> None:
        vector = self._embed(text)
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=memory_id, vector=vector, payload=metadata)],
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        vector = self._embed(query)
        results = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=limit,
        )
        return [{"id": r.id, "score": r.score, "content": r.payload.get("summary", ""), **r.payload} for r in results]

    def delete(self, memory_id: str) -> None:
        self.client.delete(collection_name=QDRANT_COLLECTION, points_selector=[memory_id])
