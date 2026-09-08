from __future__ import annotations

import pytest
from camcat.gateway.bailian import (
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_RERANKER_MODEL,
    build_embedding_payload,
    build_rerank_payload,
    extract_embedding_response,
    extract_rerank_response,
)


def test_bailian_embedding_maps_joint_input_to_one_2048_fusion_request() -> None:
    payload = build_embedding_payload(
        canonical_model=CANONICAL_EMBEDDING_MODEL,
        dimensions=2048,
        text="sunset",
        image_data_uri="data:image/jpeg;base64,YQ==",
        video_url="https://media.example/temporary/source.mp4?signature=short-lived",
        instruction="Represent the clip for retrieval.",
        fps=0.5,
    )

    assert payload == {
        "model": "qwen3-vl-embedding",
        "input": {
            "contents": [
                {"text": "sunset"},
                {"image": "data:image/jpeg;base64,YQ=="},
                {"video": "https://media.example/temporary/source.mp4?signature=short-lived"},
            ]
        },
        "parameters": {
            "dimension": 2048,
            "enable_fusion": True,
            "fps": 0.5,
            "instruct": "Represent the clip for retrieval.",
        },
    }


def test_bailian_embedding_rejects_wrong_model_or_dimension() -> None:
    with pytest.raises(ValueError, match="canonical embedding model"):
        build_embedding_payload(
            canonical_model="qwen3-vl-embedding",
            dimensions=2048,
            text="query",
        )
    with pytest.raises(ValueError, match="2048"):
        build_embedding_payload(
            canonical_model=CANONICAL_EMBEDDING_MODEL,
            dimensions=1024,
            text="query",
        )


def test_bailian_embedding_response_is_not_averaged() -> None:
    vector = extract_embedding_response(
        {"output": {"embeddings": [{"index": 0, "type": "fusion", "embedding": [0.5] * 2048}]}},
        dimensions=2048,
    )
    assert vector == [0.5] * 2048
    with pytest.raises(ValueError, match="exactly one"):
        extract_embedding_response(
            {
                "output": {
                    "embeddings": [
                        {"embedding": [0.5] * 2048},
                        {"embedding": [0.25] * 2048},
                    ]
                }
            },
            dimensions=2048,
        )


def test_bailian_rerank_maps_one_official_query_modality_and_keeps_metadata() -> None:
    payload, metadata = build_rerank_payload(
        canonical_model=CANONICAL_RERANKER_MODEL,
        query={"image_base64": "data:image/jpeg;base64,YQ=="},
        documents=[
            {
                "video_url": "https://media.example/segment.mp4?signature=short-lived",
                "metadata": {"segment_id": "seg-1", "license_name": "Pixabay"},
            }
        ],
        instruction="Rank clips for editing.",
    )

    assert payload["model"] == "qwen3-vl-rerank"
    assert payload["input"]["query"] == {"image": "data:image/jpeg;base64,YQ=="}
    assert payload["input"]["documents"] == [
        {"video": "https://media.example/segment.mp4?signature=short-lived"}
    ]
    assert payload["parameters"]["instruct"] == "Rank clips for editing."
    assert metadata == [{"segment_id": "seg-1", "license_name": "Pixabay"}]


def test_bailian_rerank_rejects_unsupported_mixed_upstream_objects() -> None:
    with pytest.raises(ValueError, match="exactly one query modality"):
        build_rerank_payload(
            canonical_model=CANONICAL_RERANKER_MODEL,
            query={"text": "sunset", "image_base64": "data:image/jpeg;base64,YQ=="},
            documents=[{"text": "beach"}],
        )


def test_bailian_rerank_response_uses_original_document_indices() -> None:
    assert extract_rerank_response(
        {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        },
        document_count=2,
    ) == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]
