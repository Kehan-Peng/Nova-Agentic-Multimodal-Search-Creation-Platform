from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import mimetypes
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from camcat.gateway.bailian import (
    CANONICAL_EMBEDDING_MODEL,
    build_embedding_payload,
    build_rerank_payload,
    extract_embedding_response,
    extract_rerank_response,
)
from camcat.gateway.upstream import BailianUpstreamError


class StagingObjectStore(Protocol):
    def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        content_type: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None: ...

    def signed_url(self, key: str, expires_seconds: int = 3600) -> str: ...

    def delete_key(self, key: str) -> None: ...


class BailianUpstream(Protocol):
    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    incoming_api_key: str
    embedding_path: str = "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    reranker_path: str = "/api/v1/services/rerank/text-rerank/text-rerank"
    chat_path: str = "/compatible-mode/v1/chat/completions"
    staging_url_seconds: int = 600
    maximum_image_bytes: int = 10 * 1024 * 1024
    maximum_video_bytes: int = 50 * 1024 * 1024
    maximum_audio_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.incoming_api_key:
            raise ValueError("provider gateway incoming API key is required")


class RerankBody(BaseModel):
    model: str
    query: dict[str, Any]
    documents: list[dict[str, Any]] = Field(min_length=1, max_length=100)
    top_n: int = Field(ge=1, le=100)
    instruction: str | None = None


def create_gateway_app(
    config: GatewayConfig,
    *,
    object_store: StagingObjectStore,
    upstream: BailianUpstream,
) -> FastAPI:
    app = FastAPI(title="CamCat Bailian provider gateway", version="0.1.0")

    @app.exception_handler(BailianUpstreamError)
    async def bailian_upstream_error(_request: Request, exc: BailianUpstreamError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    def authorize(authorization: str | None = Header(default=None)) -> None:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token, config.incoming_api_key):
            raise HTTPException(401, "invalid provider gateway bearer token")

    Authorized = Depends(authorize)

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health", dependencies=[Authorized])
    def health() -> dict[str, str]:
        return {"status": "ok", "provider": "aliyun-bailian"}

    @app.post("/v1/embeddings", dependencies=[Authorized])
    async def embeddings(
        model: str = Form(...),
        dimensions: int = Form(...),
        instruction: str | None = Form(default=None),
        text: str | None = Form(default=None),
        image: UploadFile | None = File(default=None),
        video: UploadFile | None = File(default=None),
        fps: float | None = Form(default=None),
    ) -> dict[str, Any]:
        image_data_uri: str | None = None
        if image is not None:
            content = await _read_upload(image, maximum_bytes=config.maximum_image_bytes)
            media_type = image.content_type or mimetypes.guess_type(image.filename or "")[0]
            if not media_type or not media_type.startswith("image/"):
                raise HTTPException(415, "embedding image must use an image MIME type")
            image_data_uri = f"data:{media_type};base64,{base64.b64encode(content).decode()}"

        staged_key: str | None = None
        try:
            video_url: str | None = None
            if video is not None:
                content = await _read_upload(video, maximum_bytes=config.maximum_video_bytes)
                media_type = video.content_type or mimetypes.guess_type(video.filename or "")[0]
                if media_type not in {"video/mp4", "video/quicktime", "video/x-msvideo"}:
                    raise HTTPException(415, "embedding video must be MP4, MOV or AVI")
                filename = Path(video.filename or "source.mp4").name
                staged_key = f"temporary/provider-staging/{uuid4()}/{filename}"
                expires_at = datetime.now(UTC) + timedelta(seconds=config.staging_url_seconds)
                object_store.upload_stream(
                    BytesIO(content),
                    staged_key,
                    media_type,
                    metadata={
                        "kind": "provider-staging",
                        "expires-at": expires_at.isoformat(),
                    },
                )
                video_url = object_store.signed_url(
                    staged_key, expires_seconds=config.staging_url_seconds
                )
                _require_public_video_url(video_url)
            try:
                payload = build_embedding_payload(
                    canonical_model=model,
                    dimensions=dimensions,
                    text=text,
                    image_data_uri=image_data_uri,
                    video_url=video_url,
                    instruction=instruction,
                    fps=fps,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            upstream_payload = upstream.post_json(config.embedding_path, payload)
            try:
                vector = extract_embedding_response(upstream_payload, dimensions=dimensions)
            except ValueError as exc:
                raise HTTPException(502, str(exc)) from exc
            return {"data": [{"embedding": vector}], "model": CANONICAL_EMBEDDING_MODEL}
        finally:
            if staged_key is not None:
                object_store.delete_key(staged_key)

    @app.post("/v1/rerank", dependencies=[Authorized])
    def rerank(body: RerankBody) -> dict[str, Any]:
        if body.top_n != len(body.documents):
            raise HTTPException(422, "CamCat gateway requires one score per document")
        query_variants = _rerank_query_variants(body.query)
        document_variants = _rerank_document_variants(body.documents)
        scores_by_index = {index: 0.0 for index in range(len(body.documents))}
        for query in query_variants:
            for documents in document_variants:
                try:
                    payload, _metadata = build_rerank_payload(
                        canonical_model=body.model,
                        query=query,
                        documents=documents,
                        instruction=body.instruction,
                    )
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                upstream_payload = upstream.post_json(config.reranker_path, payload)
                try:
                    results = extract_rerank_response(
                        upstream_payload, document_count=len(body.documents)
                    )
                except ValueError as exc:
                    raise HTTPException(502, str(exc)) from exc
                for item in results:
                    scores_by_index[int(item["index"])] += float(item["relevance_score"])
        divisor = float(len(query_variants) * len(document_variants))
        return {
            "results": [
                {"index": index, "relevance_score": score / divisor}
                for index, score in scores_by_index.items()
            ]
        }

    @app.post("/v1/chat/completions", dependencies=[Authorized])
    def chat_completions(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            raise HTTPException(422, "chat model is required")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise HTTPException(422, "chat messages are required")
        return upstream.post_json(config.chat_path, body)

    @app.post("/v1/analyze", dependencies=[Authorized])
    async def analyze_video(
        model: str = Form(...),
        prompt: str = Form(...),
        transcript: str = Form(default=""),
        response_schema: str = Form(...),
        video: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            schema = json.loads(response_schema)
        except json.JSONDecodeError as exc:
            raise HTTPException(422, "response_schema must be valid JSON") from exc
        if not isinstance(schema, dict):
            raise HTTPException(422, "response_schema must be a JSON object")

        content = await _read_upload(video, maximum_bytes=config.maximum_video_bytes)
        media_type = video.content_type or mimetypes.guess_type(video.filename or "")[0]
        if media_type not in {"video/mp4", "video/quicktime", "video/x-msvideo"}:
            raise HTTPException(415, "analysis video must be MP4, MOV or AVI")
        filename = Path(video.filename or "source.mp4").name
        staged_key = f"temporary/provider-staging/{uuid4()}/{filename}"
        expires_at = datetime.now(UTC) + timedelta(seconds=config.staging_url_seconds)
        try:
            object_store.upload_stream(
                BytesIO(content),
                staged_key,
                media_type,
                metadata={
                    "kind": "provider-staging",
                    "expires-at": expires_at.isoformat(),
                },
            )
            video_url = object_store.signed_url(
                staged_key, expires_seconds=config.staging_url_seconds
            )
            _require_public_video_url(video_url)
            transcript_context = transcript.strip() or "(no transcript available)"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only a JSON object conforming to the supplied JSON "
                            "Schema. Do not wrap it in Markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_url}},
                            {
                                "type": "text",
                                "text": (
                                    f"{prompt}\nTranscript context: {transcript_context}\n"
                                    f"JSON Schema: {json.dumps(schema, ensure_ascii=False)}"
                                ),
                            },
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }
            response = upstream.post_json(config.chat_path, payload)
            try:
                analysis = _parse_json_content(_extract_chat_content(response))
            except ValueError as exc:
                raise HTTPException(502, str(exc)) from exc
            return {"analysis": analysis}
        finally:
            object_store.delete_key(staged_key)

    @app.post("/v1/audio/transcriptions", dependencies=[Authorized])
    async def transcribe_audio(
        model: str = Form(...), file: UploadFile = File(...)
    ) -> dict[str, str]:
        content = await _read_upload(file, maximum_bytes=config.maximum_audio_bytes)
        media_type = file.content_type or mimetypes.guess_type(file.filename or "")[0]
        if not media_type or not media_type.startswith("audio/"):
            raise HTTPException(415, "transcription input must use an audio MIME type")
        data_uri = f"data:{media_type};base64,{base64.b64encode(content).decode()}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": data_uri}}],
                }
            ],
            "stream": False,
            "asr_options": {"enable_itn": True},
        }
        response = upstream.post_json(config.chat_path, payload)
        try:
            text = _extract_chat_content(response).strip()
        except ValueError as exc:
            raise HTTPException(502, str(exc)) from exc
        if not text:
            raise HTTPException(502, "ASR upstream returned an empty transcript")
        return {"text": text}

    return app


async def _read_upload(upload: UploadFile, *, maximum_bytes: int) -> bytes:
    content = await upload.read(maximum_bytes + 1)
    if not content:
        raise HTTPException(422, "uploaded provider media is empty")
    if len(content) > maximum_bytes:
        raise HTTPException(413, "uploaded provider media exceeds the upstream limit")
    return content


def _extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat upstream response is missing choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise ValueError("chat upstream response is missing a message")
    content = first["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("chat upstream message content must be text")
    return content


def _parse_json_content(content: str) -> dict[str, Any]:
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError("visual analysis upstream response is not JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("visual analysis upstream response must be a JSON object")
    return parsed


def _rerank_query_variants(query: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    text = query.get("text")
    if isinstance(text, str) and text.strip():
        variants.append({"text": text.strip()})
    image = query.get("image_base64", query.get("image"))
    if isinstance(image, str) and image:
        variants.append({"image_base64": image})
    if not variants:
        raise HTTPException(422, "reranker query requires text or image")
    return variants


def _rerank_document_variants(
    documents: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    variants: list[list[dict[str, Any]]] = []
    modality_fields = (
        ("text", "text"),
        ("image_base64", "image_base64"),
        ("video_url", "video_url"),
    )
    for input_field, output_field in modality_fields:
        values = [document.get(input_field) for document in documents]
        if all(isinstance(value, str) and value.strip() for value in values):
            if input_field == "video_url":
                for value in values:
                    _require_public_video_url(str(value))
            variants.append(
                [
                    {
                        output_field: str(value).strip(),
                        "metadata": dict(document.get("metadata") or {}),
                    }
                    for document, value in zip(documents, values, strict=True)
                ]
            )
    if not variants:
        raise HTTPException(
            422,
            "reranker documents require one common text, image or video modality",
        )
    return variants


def _require_public_video_url(value: str) -> None:
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    private = parsed.scheme != "https" or not hostname or "." not in hostname
    if not private:
        try:
            private = ipaddress.ip_address(hostname).is_private
        except ValueError:
            private = hostname == "localhost" or hostname.endswith(".local")
    if private:
        raise HTTPException(
            503,
            "provider video staging requires a publicly reachable HTTPS object-store endpoint",
        )
