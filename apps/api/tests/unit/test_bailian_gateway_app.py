from __future__ import annotations

from io import BytesIO
from typing import Any

from camcat.gateway.app import GatewayConfig, create_gateway_app
from camcat.gateway.upstream import BailianUpstreamError
from fastapi.testclient import TestClient


class FakeStore:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def upload_stream(
        self,
        stream: BytesIO,
        key: str,
        content_type: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        assert content_type == "video/mp4"
        assert metadata and metadata["kind"] == "provider-staging"
        self.uploads[key] = stream.read()

    def signed_url(self, key: str, expires_seconds: int = 3600) -> str:
        assert expires_seconds <= 900
        return f"https://media.example/{key}?signed=true"

    def delete_key(self, key: str) -> None:
        self.deleted.append(key)


class FakeUpstream:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((path, payload))
        if "embeddings" in path:
            return {
                "output": {
                    "embeddings": [{"index": 0, "type": "fusion", "embedding": [0.25] * 2048}]
                }
            }
        if "chat/completions" in path:
            model = payload["model"]
            if model == "qwen3-asr-flash":
                return {"choices": [{"message": {"content": "hello transcript"}}]}
            if any(
                item.get("type") == "video_url"
                for message in payload.get("messages", [])
                for item in message.get("content", [])
                if isinstance(item, dict)
            ):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"description":"shore","scene":"beach",'
                                    '"actions":[],"people":[],"composition":"wide",'
                                    '"tags":["sea"],"event_type":"travel",'
                                    '"risk_score":0,"risk_labels":[]}'
                                )
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": '{"summary":"ok"}'}}]}
        query = payload.get("input", {}).get("query", {})
        score = 0.6 if "text" in query else 1.0
        return {"output": {"results": [{"index": 0, "relevance_score": score}]}}


def test_gateway_accepts_original_video_and_removes_staging_object() -> None:
    store = FakeStore()
    upstream = FakeUpstream()
    app = create_gateway_app(
        GatewayConfig(incoming_api_key="gateway-secret"),
        object_store=store,
        upstream=upstream,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer gateway-secret"},
            data={
                "model": "Qwen/Qwen3-VL-Embedding-8B",
                "dimensions": "2048",
                "text": "sunset",
                "fps": "0.5",
            },
            files={"video": ("source.mp4", b"original-video", "video/mp4")},
        )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "data": [{"embedding": [0.25] * 2048}],
        "model": "Qwen/Qwen3-VL-Embedding-8B",
    }
    assert list(store.uploads.values()) == [b"original-video"]
    assert store.deleted == list(store.uploads)
    path, payload = upstream.calls[0]
    assert path.endswith("/services/embeddings/multimodal-embedding/multimodal-embedding")
    assert payload["parameters"]["dimension"] == 2048
    assert payload["parameters"]["enable_fusion"] is True
    assert payload["input"]["contents"][1]["video"].startswith("https://media.example/")


def test_gateway_requires_bearer_auth_and_adapts_rerank_response() -> None:
    store = FakeStore()
    upstream = FakeUpstream()
    app = create_gateway_app(
        GatewayConfig(incoming_api_key="gateway-secret"),
        object_store=store,
        upstream=upstream,
    )
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health").status_code == 401
        response = client.post(
            "/v1/rerank",
            headers={"Authorization": "Bearer gateway-secret"},
            json={
                "model": "Qwen/Qwen3-VL-Reranker-8B",
                "query": {"text": "sunset", "image_base64": "data:image/jpeg;base64,YQ=="},
                "documents": [
                    {
                        "text": "beach",
                        "video_url": "https://media.example/segment.mp4?signed=true",
                        "metadata": {"segment_id": "seg-1", "license_name": "Pixabay"},
                    }
                ],
                "top_n": 1,
                "instruction": "Rank source clips.",
            },
        )

    assert response.status_code == 200, response.text
    assert response.json() == {"results": [{"index": 0, "relevance_score": 0.8}]}
    rerank_payloads = [payload for path, payload in upstream.calls if "rerank" in path]
    assert [payload["input"]["query"] for payload in rerank_payloads] == [
        {"text": "sunset"},
        {"text": "sunset"},
        {"image": "data:image/jpeg;base64,YQ=="},
        {"image": "data:image/jpeg;base64,YQ=="},
    ]
    assert [payload["input"]["documents"] for payload in rerank_payloads] == [
        [{"text": "beach"}],
        [{"video": "https://media.example/segment.mp4?signed=true"}],
        [{"text": "beach"}],
        [{"video": "https://media.example/segment.mp4?signed=true"}],
    ]


def test_gateway_adapts_chat_visual_analysis_and_asr() -> None:
    store = FakeStore()
    upstream = FakeUpstream()
    app = create_gateway_app(
        GatewayConfig(incoming_api_key="gateway-secret"),
        object_store=store,
        upstream=upstream,
    )
    headers = {"Authorization": "Bearer gateway-secret"}
    with TestClient(app) as client:
        chat = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "qwen3-vl-plus",
                "messages": [{"role": "user", "content": "return json"}],
            },
        )
        analysis = client.post(
            "/v1/analyze",
            headers=headers,
            data={
                "model": "qwen3-vl-plus",
                "prompt": "describe",
                "transcript": "waves",
                "response_schema": '{"type":"object"}',
            },
            files={"video": ("source.mp4", b"original-video", "video/mp4")},
        )
        asr = client.post(
            "/v1/audio/transcriptions",
            headers=headers,
            data={"model": "qwen3-asr-flash"},
            files={"file": ("audio.mp3", b"original-audio", "audio/mpeg")},
        )

    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == '{"summary":"ok"}'
    assert analysis.status_code == 200, analysis.text
    assert analysis.json()["analysis"]["scene"] == "beach"
    assert asr.status_code == 200, asr.text
    assert asr.json() == {"text": "hello transcript"}
    asr_payload = upstream.calls[-1][1]
    audio_data = asr_payload["messages"][0]["content"][0]["input_audio"]["data"]
    assert audio_data.startswith("data:audio/mpeg;base64,")
    assert store.deleted == list(store.uploads)


def test_gateway_sanitizes_upstream_failures_as_bad_gateway() -> None:
    class FailingUpstream:
        def post_json(self, _path: str, _payload: dict[str, Any]) -> dict[str, Any]:
            raise BailianUpstreamError("Bailian transient HTTP response 503")

    app = create_gateway_app(
        GatewayConfig(incoming_api_key="gateway-secret"),
        object_store=FakeStore(),
        upstream=FailingUpstream(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer gateway-secret"},
            json={"model": "qwen3-vl-plus", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "Bailian transient HTTP response 503"}


def test_gateway_rejects_video_staging_url_that_bailian_cannot_reach() -> None:
    class PrivateStore(FakeStore):
        def signed_url(self, key: str, expires_seconds: int = 3600) -> str:
            return f"http://minio:9000/camcat/{key}"

    store = PrivateStore()
    app = create_gateway_app(
        GatewayConfig(incoming_api_key="gateway-secret"),
        object_store=store,
        upstream=FakeUpstream(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer gateway-secret"},
            data={"model": "Qwen/Qwen3-VL-Embedding-8B", "dimensions": "2048"},
            files={"video": ("source.mp4", b"original-video", "video/mp4")},
        )

    assert response.status_code == 503
    assert "publicly reachable" in response.json()["detail"]
    assert store.deleted == list(store.uploads)


def test_visual_analysis_cleans_partial_staging_upload_failure() -> None:
    class PartialFailureStore(FakeStore):
        def upload_stream(
            self,
            stream: BytesIO,
            key: str,
            content_type: str,
            *,
            metadata: dict[str, str] | None = None,
        ) -> None:
            super().upload_stream(stream, key, content_type, metadata=metadata)
            raise RuntimeError("simulated object store interruption")

    store = PartialFailureStore()
    app = create_gateway_app(
        GatewayConfig(incoming_api_key="gateway-secret"),
        object_store=store,
        upstream=FakeUpstream(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/analyze",
            headers={"Authorization": "Bearer gateway-secret"},
            data={
                "model": "qwen3-vl-plus",
                "prompt": "describe",
                "response_schema": '{"type":"object"}',
            },
            files={"video": ("source.mp4", b"original-video", "video/mp4")},
        )

    assert response.status_code == 500
    assert store.deleted == list(store.uploads)
