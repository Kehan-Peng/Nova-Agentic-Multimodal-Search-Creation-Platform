from __future__ import annotations

from typing import Any

from camcat.retrieval.milvus_store import MilvusHit
from camcat.retrieval.service import RetrievalService


class FakeEmbedding:
    def embed_text(self, _text: str) -> list[float]:
        return [0.1] * 2048


class FakeStore:
    def dense_search(self, _vector: list[float], **_kwargs: Any) -> list[MilvusHit]:
        return [
            MilvusHit(
                segment_id=f"segment-{index}",
                score=1.0 - index / 10,
                entity={
                    "description_text": f"clip {index}",
                    "storage_key": f"segments/clip-{index}.mp4",
                    "duration": 2.0,
                    "risk_score": 0.0,
                    "created_at_epoch": 0,
                    "tags": ["travel"],
                    "event_type": "travel",
                    "semantic_metadata": {"scene": "shore"},
                    "license_name": "Pixabay",
                    "source_url": "https://pixabay.example/source",
                },
            )
            for index in range(5)
        ]

    def bm25_search(self, _text: str, **_kwargs: Any) -> list[MilvusHit]:
        return []

    def scalar_search(self, _filters: dict[str, Any], **_kwargs: Any) -> list[MilvusHit]:
        return []


class RecordingReranker:
    def __init__(self) -> None:
        self.document_batches: list[list[dict[str, Any]]] = []

    def rerank(self, _query: dict[str, Any], documents: list[dict[str, Any]]) -> list[float]:
        self.document_batches.append(documents)
        return [0.9] * len(documents)


class FakeSigner:
    def signed_url(self, key: str, expires_seconds: int = 3600) -> str:
        assert expires_seconds <= 900
        return f"https://media.example/{key}?signed=true"


def test_retrieval_reranks_real_video_documents_in_provider_sized_batches() -> None:
    reranker = RecordingReranker()
    service = RetrievalService(  # type: ignore[arg-type]
        store=FakeStore(),
        embedding=FakeEmbedding(),
        reranker=reranker,
        media_signer=FakeSigner(),
    )

    result = service.search(
        query_text="sunset travel",
        query_image_base64=None,
        filters={},
        top_k=5,
    )

    assert len(result) == 5
    assert [len(batch) for batch in reranker.document_batches] == [4, 1]
    first = reranker.document_batches[0][0]
    assert first["text"] == "clip 0"
    assert first["video_url"].startswith("https://media.example/segments/clip-0.mp4")
    assert first["metadata"]["license_name"] == "Pixabay"
    assert first["metadata"]["semantic_metadata"] == {"scene": "shore"}
