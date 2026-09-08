from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from camcat.agent.graph import CamCatGraph, CamCatState
from camcat.agent.scope import editing_retrieval_filters
from camcat.config import Settings, get_settings
from camcat.database import get_db
from camcat.domain.state_patch import PatchConflict
from camcat.editing.policies import choose_aspect_ratio, expiry_for_upload, resolution_for_ratio
from camcat.media.ffmpeg import MediaCommandError, probe
from camcat.models import (
    Asset,
    AssetStatus,
    AuditEvent,
    EditingSession,
    GraphRun,
    Job,
    JobKind,
    JobStatus,
    Project,
    Segment,
    StatePatch,
    StateVersion,
    utcnow,
)
from camcat.repositories import JobRepository, StateRepository, sanitize_job_error
from camcat.retrieval.milvus_store import MilvusSegmentStore
from camcat.retrieval.service import RetrievalService
from camcat.schemas import (
    AgentEditRequest,
    AgenticSearchResponse,
    CreateEditingSessionRequest,
    CreateProjectRequest,
    EditingSessionResponse,
    ErrorEnvelope,
    GraphRunResponse,
    ImportOpenMediaRequest,
    JobResponse,
    PageResponse,
    PatchEditingSessionRequest,
    PatchResponse,
    ProjectResponse,
    RankedSegmentResponse,
    RenderRequest,
    RollbackRequest,
    SearchRequest,
    SourceMediaReference,
    SourceUploadResponse,
    VersionResponse,
    VideoResponse,
)
from camcat.security import AuthenticationError, authorize_library_import, resolve_owner
from camcat.services.object_store import ObjectStore
from camcat.services.providers import (
    QwenAsrClient,
    QwenChatClient,
    QwenEmbeddingClient,
    QwenRerankerClient,
)
from camcat.services.remote_media import (
    UnsafeRemoteMediaUrl,
    download_remote_media,
)


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.object_store = ObjectStore(settings)
        self.milvus = MilvusSegmentStore(settings)
        self.embedding = QwenEmbeddingClient(settings)
        self.reranker = QwenRerankerClient(settings)
        self.llm = QwenChatClient(settings)
        self.asr = QwenAsrClient(settings)
        self.retrieval = RetrievalService(
            store=self.milvus,
            embedding=self.embedding,
            reranker=self.reranker,
            media_signer=self.object_store,
        )
        self.graph = CamCatGraph(llm=self.llm, retrieval=self.retrieval)


settings = get_settings()
services = Services(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    Path(settings.runtime_dir).mkdir(parents=True, exist_ok=True)
    services.object_store.ensure_bucket()
    services.milvus.ensure_collection()
    yield


app = FastAPI(
    title="CamCat API",
    version="0.1.0",
    description="Multimodal intelligent video editing harness",
    lifespan=lifespan,
    responses={
        status: {"model": ErrorEnvelope, "description": "CamCat error envelope"}
        for status in (400, 401, 403, 404, 409, 410, 413, 415, 422, 500, 502)
    },
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Any:
    request_id = request.headers.get("X-Request-Id") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.exception_handler(PatchConflict)
async def patch_conflict_handler(request: Request, exc: PatchConflict) -> JSONResponse:
    return _error(
        request,
        status=409,
        code="state_version_conflict",
        message="编辑状态已被其他操作更新，请刷新后重试或重新应用补丁。",
        details={
            "expected_version": exc.expected_version,
            "current_version": exc.current_version,
            "current_patch": exc.current_patch,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(
        request,
        status=422,
        code="validation_error",
        message="请求参数不符合 API 合同。",
        details={"errors": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return _error(
        request,
        status=exc.status_code,
        code="http_error",
        message=str(exc.detail),
        details={},
    )


@app.exception_handler(LookupError)
async def lookup_error_handler(request: Request, exc: LookupError) -> JSONResponse:
    return _error(
        request,
        status=404,
        code="not_found",
        message=str(exc),
        details={},
    )


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, _exc: Exception) -> JSONResponse:
    return _error(
        request,
        status=500,
        code="internal_error",
        message="服务器无法完成请求。",
        details={},
    )


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    services.object_store.healthcheck()
    services.milvus.healthcheck()
    services.embedding.healthcheck()
    services.reranker.healthcheck()
    services.llm.healthcheck()
    services.asr.healthcheck()
    return {"status": "ready"}


Db = Annotated[Session, Depends(get_db)]


def authenticated_owner(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    x_authenticated_user: Annotated[str | None, Header(alias="X-Authenticated-User")] = None,
    x_proxy_secret: Annotated[str | None, Header(alias="X-CamCat-Proxy-Secret")] = None,
) -> str:
    try:
        return resolve_owner(
            security_mode=settings.security_mode,
            local_user_id=settings.local_user_id,
            claimed_user_id=x_user_id,
            authenticated_user=x_authenticated_user,
            proxy_secret=x_proxy_secret,
            expected_proxy_secret=settings.trusted_proxy_secret.get_secret_value(),
        )
    except AuthenticationError as exc:
        raise HTTPException(401, str(exc)) from exc


Owner = Annotated[str, Depends(authenticated_owner)]


def _require_library_admin(supplied_key: str | None) -> None:
    if not authorize_library_import(
        settings.security_mode,
        supplied_key,
        settings.library_admin_key.get_secret_value(),
    ):
        raise HTTPException(403, "长期素材导入需要素材库管理员权限")


@app.post("/api/v1/projects", response_model=ProjectResponse, status_code=201)
def create_project(request: CreateProjectRequest, db: Db, owner_id: Owner) -> ProjectResponse:
    project = Project(owner_id=owner_id, name=request.name.strip())
    db.add(project)
    db.commit()
    return ProjectResponse(
        project_id=project.id,
        name=project.name,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@app.get("/api/v1/projects", response_model=PageResponse)
def list_projects(
    db: Db, owner_id: Owner, limit: int = 50, cursor: UUID | None = None
) -> PageResponse:
    query = select(Project).where(Project.owner_id == owner_id)
    if cursor:
        query = query.where(Project.id > cursor)
    projects = db.scalars(query.order_by(Project.id).limit(min(100, max(1, limit)) + 1)).all()
    page, next_cursor = _slice_page(projects, limit)
    return PageResponse(
        items=[
            ProjectResponse(
                project_id=item.id,
                name=item.name,
                created_at=item.created_at,
                updated_at=item.updated_at,
            ).model_dump(mode="json")
            for item in page
        ],
        next_cursor=next_cursor,
    )


@app.post("/api/v1/videos", response_model=VideoResponse, status_code=202)
def upload_video(
    db: Db,
    owner_id: Owner,
    file: UploadFile = File(...),
    license_name: str = Form(..., min_length=1, max_length=255),
    source_url: str = Form(..., min_length=1, max_length=2048),
    analysis_mode: str = Form("keyframes"),
    admin_key: Annotated[str | None, Header(alias="X-CamCat-Admin-Key")] = None,
) -> VideoResponse:
    _require_library_admin(admin_key)
    if not source_url.startswith(("http://", "https://")) or license_name == "user-provided":
        raise HTTPException(422, "长期素材必须提供可验证的 HTTP(S) 来源与非用户上传许可证")
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(415, "仅支持视频文件")
    if analysis_mode not in {"keyframes", "per-second"}:
        raise HTTPException(422, "analysis_mode 仅支持 keyframes 或 per-second")
    upload_dir = Path(settings.runtime_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary = upload_dir / f"{uuid4()}.upload"
    size = _copy_upload(file, temporary, settings.upload_max_bytes)
    try:
        _validate_video_file(temporary)
        _enforce_owner_quota(db, owner_id, size)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    deterministic_asset_id = uuid5(NAMESPACE_URL, f"camcat-asset:{source_url}:{license_name}")
    existing = db.get(Asset, deterministic_asset_id)
    if existing is not None:
        temporary.unlink(missing_ok=True)
        return _video_response(existing, db)
    asset = Asset(
        id=deterministic_asset_id,
        owner_id=owner_id,
        filename=Path(file.filename or "video.mp4").name,
        content_type=file.content_type,
        size_bytes=size,
        storage_key="pending",
        status=AssetStatus.UPLOADED,
        license_name=license_name,
        source_url=source_url,
    )
    db.add(asset)
    db.flush()
    asset.storage_key = f"originals/{asset.id}/{asset.filename}"
    try:
        services.object_store.upload_file(temporary, asset.storage_key, asset.content_type)
    finally:
        temporary.unlink(missing_ok=True)
    job = JobRepository(db).enqueue(
        owner_id=owner_id,
        kind=JobKind.INGEST_MEDIA,
        payload={"asset_id": str(asset.id), "analysis_mode": analysis_mode},
    )
    return _video_response(asset, db, job=job)


@app.post("/api/v1/source-media", response_model=SourceUploadResponse, status_code=202)
def upload_source_media(
    db: Db,
    owner_id: Owner,
    files: list[UploadFile] = File(...),
    analysis_mode: str = Form("keyframes"),
) -> SourceUploadResponse:
    """Store user originals transiently; never create Asset/Segment/Milvus records."""
    if not files:
        raise HTTPException(422, "至少需要一个原始视频")
    if len(files) > 20:
        raise HTTPException(422, "单次最多上传 20 个原始视频")
    if analysis_mode not in {"keyframes", "per-second"}:
        raise HTTPException(422, "analysis_mode 仅支持 keyframes 或 per-second")
    created_at = datetime.now(UTC)
    expires_at = expiry_for_upload(created_at)
    batch_id = str(uuid4())
    upload_dir = Path(settings.runtime_dir) / "uploads" / batch_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    source_media: list[dict[str, Any]] = []
    total_size = 0
    for upload in files:
        if not upload.content_type or not upload.content_type.startswith("video/"):
            raise HTTPException(415, "仅支持视频文件")
        media_id = str(uuid4())
        filename = Path(upload.filename or "source.mp4").name
        temporary = upload_dir / f"{media_id}.upload"
        size = _copy_upload(upload, temporary, settings.upload_max_bytes)
        try:
            _validate_video_file(temporary)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        total_size += size
        if total_size > settings.upload_max_bytes:
            temporary.unlink(missing_ok=True)
            raise HTTPException(413, "本次视频总大小超过上传限制")
        try:
            _enforce_owner_quota(db, owner_id, total_size)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        storage_key = f"temporary/{owner_id}/{batch_id}/{media_id}/{filename}"
        try:
            services.object_store.upload_file(
                temporary,
                storage_key,
                upload.content_type,
                metadata={"expires-at": expires_at.isoformat(), "kind": "user-source"},
            )
        finally:
            temporary.unlink(missing_ok=True)
        source_media.append(
            {
                "media_id": media_id,
                "filename": filename,
                "content_type": upload.content_type,
                "storage_key": storage_key,
                "expires_at": expires_at.isoformat(),
                "size_bytes": size,
            }
        )
    job = JobRepository(db).enqueue(
        owner_id=owner_id,
        kind=JobKind.ANALYZE_SOURCE,
        payload={
            "batch_id": batch_id,
            "analysis_mode": analysis_mode,
            "expires_at": expires_at.isoformat(),
            "source_media": source_media,
        },
        idempotency_key=f"source-analysis:{batch_id}",
        expires_at=expires_at,
    )
    return SourceUploadResponse(
        batch_id=batch_id,
        status=job.status.value,
        job_id=job.id,
        expires_at=expires_at,
        media=[
            SourceMediaReference(
                **item,
                playback_url=services.object_store.signed_url(
                    str(item["storage_key"]), expires_seconds=4 * 60 * 60
                ),
            )
            for item in source_media
        ],
    )


@app.post("/api/v1/videos/import", response_model=VideoResponse, status_code=202)
def import_open_media(
    request: ImportOpenMediaRequest,
    db: Db,
    owner_id: Owner,
    admin_key: Annotated[str | None, Header(alias="X-CamCat-Admin-Key")] = None,
) -> VideoResponse:
    _require_library_admin(admin_key)
    existing = db.scalar(
        select(Asset).where(
            Asset.source_url == str(request.source_url),
            Asset.license_name == request.license_name,
        )
    )
    if existing is not None:
        return _video_response(existing, db)
    upload_dir = Path(settings.runtime_dir) / "imports"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary = upload_dir / f"{uuid4()}.mp4"
    try:
        try:
            downloaded = download_remote_media(
                str(request.download_url),
                temporary,
                maximum_bytes=settings.upload_max_bytes,
            )
        except UnsafeRemoteMediaUrl as exc:
            raise HTTPException(422, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "远程素材服务暂时不可用") from exc
        _validate_video_file(temporary)
        _enforce_owner_quota(db, owner_id, downloaded.size_bytes)
        asset = Asset(
            id=uuid5(
                NAMESPACE_URL,
                f"camcat-asset:{request.source_url}:{request.license_name}",
            ),
            owner_id=owner_id,
            filename=Path(request.filename).name,
            content_type=downloaded.content_type,
            size_bytes=downloaded.size_bytes,
            storage_key="pending",
            status=AssetStatus.UPLOADED,
            license_name=request.license_name,
            source_url=str(request.source_url),
        )
        db.add(asset)
        db.flush()
        asset.storage_key = f"originals/{asset.id}/{asset.filename}"
        services.object_store.upload_file(temporary, asset.storage_key, downloaded.content_type)
        job = JobRepository(db).enqueue(
            owner_id=owner_id,
            kind=JobKind.INGEST_MEDIA,
            payload={"asset_id": str(asset.id)},
            idempotency_key=f"ingest:{request.source_url}:{request.license_name}",
        )
        return _video_response(asset, db, job=job)
    finally:
        temporary.unlink(missing_ok=True)


@app.get("/api/v1/videos", response_model=PageResponse)
def list_videos(
    db: Db, owner_id: Owner, limit: int = 50, cursor: UUID | None = None
) -> PageResponse:
    query = select(Asset).where(Asset.owner_id == owner_id)
    if cursor:
        query = query.where(Asset.id > cursor)
    assets = db.scalars(query.order_by(Asset.id).limit(min(100, max(1, limit)) + 1)).all()
    page, next_cursor = _slice_page(assets, limit)
    return PageResponse(
        items=[_video_response(asset, db).model_dump(mode="json") for asset in page],
        next_cursor=next_cursor,
    )


@app.get("/api/v1/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: UUID, db: Db, owner_id: Owner) -> VideoResponse:
    asset = db.scalar(select(Asset).where(Asset.id == video_id, Asset.owner_id == owner_id))
    if asset is None:
        raise HTTPException(404, "素材不存在")
    return _video_response(asset, db)


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: UUID, db: Db, owner_id: Owner) -> JobResponse:
    job = db.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
    if job is None:
        raise HTTPException(404, "任务不存在")
    return _job_response(job)


@app.get("/api/v1/jobs", response_model=PageResponse)
def list_jobs(db: Db, owner_id: Owner, limit: int = 50, cursor: UUID | None = None) -> PageResponse:
    query = select(Job).where(Job.owner_id == owner_id)
    if cursor:
        query = query.where(Job.id > cursor)
    jobs = db.scalars(query.order_by(Job.id).limit(min(100, max(1, limit)) + 1)).all()
    page, next_cursor = _slice_page(jobs, limit)
    return PageResponse(
        items=[_job_response(item).model_dump(mode="json") for item in page],
        next_cursor=next_cursor,
    )


@app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: UUID, db: Db, owner_id: Owner) -> JobResponse:
    job = db.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
    if job is None:
        raise HTTPException(404, "任务不存在")
    return _job_response(JobRepository(db).cancel(job))


@app.post("/api/v1/jobs/{job_id}/retry", response_model=JobResponse)
def retry_job(job_id: UUID, db: Db, owner_id: Owner) -> JobResponse:
    job = db.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
    if job is None:
        raise HTTPException(404, "任务不存在")
    try:
        return _job_response(JobRepository(db).retry(job))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _job_response(job: Job) -> JobResponse:
    result = dict(job.result) if job.result else None
    if result and result.get("output_key"):
        result["output_url"] = services.object_store.signed_url(str(result["output_key"]))
        result["download_url"] = services.object_store.signed_download_url(
            str(result["output_key"]), f"camcat-{job.id}.mp4"
        )
    if result and result.get("subtitle_key"):
        result["subtitle_url"] = services.object_store.signed_url(str(result["subtitle_key"]))
    if result and result.get("source_media"):
        result["source_media"] = [
            {
                **item,
                "playback_url": services.object_store.signed_url(
                    str(item["storage_key"]), expires_seconds=4 * 60 * 60
                ),
            }
            for item in result["source_media"]
        ]
    return JobResponse(
        job_id=job.id,
        kind=job.kind.value,
        status=job.status.value,
        progress=job.progress,
        result=result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        checkpoint=job.checkpoint or {},
    )


def _editing_session_response(
    session: EditingSession, *, version: int, document: dict[str, Any]
) -> EditingSessionResponse:
    """Add short-lived playback URLs without persisting them in version history."""
    state = deepcopy(document)
    for media in state.get("source_media") or []:
        storage_key = media.get("storage_key")
        if storage_key:
            media["playback_url"] = services.object_store.signed_url(
                str(storage_key), expires_seconds=60 * 60
            )
    for segment in state.get("source_segments") or []:
        storage_key = segment.get("storage_key")
        thumbnail_key = segment.get("thumbnail_key")
        if storage_key:
            segment["source_video_url"] = services.object_store.signed_url(
                str(storage_key), expires_seconds=60 * 60
            )
        if thumbnail_key:
            segment["thumbnail_url"] = services.object_store.signed_url(
                str(thumbnail_key), expires_seconds=60 * 60
            )
    return EditingSessionResponse(
        editing_session_id=session.id,
        state_version=version,
        state=state,
        updated_at=session.updated_at,
    )


@app.post("/api/v1/search/agentic", response_model=AgenticSearchResponse)
def agentic_search(request: SearchRequest, db: Db, owner_id: Owner) -> AgenticSearchResponse:
    run = GraphRun(
        owner_id=owner_id,
        thread_id=request.thread_id,
        state=_redact_graph_state(request.model_dump(exclude_none=True)),
        node_trace=[],
        status=JobStatus.RUNNING,
    )
    db.add(run)
    db.commit()
    try:
        result = services.graph.invoke(
            CamCatState(
                mode="search",
                query_text=request.query_text or "",
                query_image_base64=request.query_image_base64 or "",
                explicit_filters=request.filters,
                top_k=request.top_k,
            )
        )
        run.state = _redact_graph_state(dict(result))
        run.node_trace = result.get("node_trace", [])
        run.status = JobStatus.SUCCEEDED
        run.finished_at = utcnow()
        db.commit()
        return _search_response(run, result, db)
    except Exception as exc:
        run.status = JobStatus.FAILED
        run.error = sanitize_job_error(exc)
        run.finished_at = utcnow()
        db.commit()
        raise


@app.get("/api/v1/graph-runs", response_model=PageResponse)
def list_graph_runs(
    db: Db, owner_id: Owner, limit: int = 50, cursor: UUID | None = None
) -> PageResponse:
    query = select(GraphRun).where(GraphRun.owner_id == owner_id)
    if cursor:
        query = query.where(GraphRun.id > cursor)
    runs = db.scalars(query.order_by(GraphRun.id).limit(min(100, max(1, limit)) + 1)).all()
    page, next_cursor = _slice_page(runs, limit)
    return PageResponse(
        items=[_graph_run_response(item).model_dump(mode="json") for item in page],
        next_cursor=next_cursor,
    )


@app.get("/api/v1/graph-runs/{run_id}", response_model=GraphRunResponse)
def get_graph_run(run_id: UUID, db: Db, owner_id: Owner) -> GraphRunResponse:
    run = db.scalar(select(GraphRun).where(GraphRun.id == run_id, GraphRun.owner_id == owner_id))
    if run is None:
        raise HTTPException(404, "Graph Run 不存在")
    return _graph_run_response(run)


@app.get("/api/v1/graph-runs/{run_id}/search-result", response_model=AgenticSearchResponse)
def get_graph_run_search_result(run_id: UUID, db: Db, owner_id: Owner) -> AgenticSearchResponse:
    run = db.scalar(select(GraphRun).where(GraphRun.id == run_id, GraphRun.owner_id == owner_id))
    if run is None:
        raise HTTPException(404, "Graph Run 不存在")
    if run.status != JobStatus.SUCCEEDED:
        raise HTTPException(409, "Graph Run 尚未完成")
    return _search_response(run, cast(CamCatState, run.state), db)


@app.get("/api/v1/graph-runs/{run_id}/events")
def replay_graph_run_events(
    run_id: UUID, db: Db, owner_id: Owner, after: int = -1
) -> StreamingResponse:
    run = db.scalar(select(GraphRun).where(GraphRun.id == run_id, GraphRun.owner_id == owner_id))
    if run is None:
        raise HTTPException(404, "Graph Run 不存在")

    def replay() -> Any:
        for index, item in enumerate(run.node_trace):
            if index > after:
                yield _sse(
                    "node_completed",
                    {
                        "graph_run_id": str(run.id),
                        "node": item.get("node_name"),
                        "status": item.get("status"),
                        "duration_ms": item.get("duration_ms"),
                        "message": f"{item.get('node_name')} 已完成",
                    },
                    event_id=index,
                )
        yield _sse(
            "run_status",
            {
                "graph_run_id": str(run.id),
                "status": run.status.value,
                "message": f"Graph Run {run.status.value}",
            },
            event_id=len(run.node_trace),
        )

    return StreamingResponse(
        replay(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.post("/api/v1/editing/sessions", response_model=EditingSessionResponse, status_code=201)
def create_editing_session(
    request: CreateEditingSessionRequest, db: Db, owner_id: Owner
) -> EditingSessionResponse:
    if (
        request.project_id is not None
        and db.scalar(
            select(Project.id).where(Project.id == request.project_id, Project.owner_id == owner_id)
        )
        is None
    ):
        raise HTTPException(404, "项目不存在")
    if (
        request.video_id is not None
        and db.scalar(
            select(Asset.id).where(Asset.id == request.video_id, Asset.owner_id == owner_id)
        )
        is None
    ):
        raise HTTPException(404, "素材不存在")
    source_media: list[dict[str, Any]] = []
    source_segments: list[dict[str, Any]] = []
    transient_expires_at: datetime | None = None
    if request.source_job_id is not None:
        source_job = db.scalar(
            select(Job).where(Job.id == request.source_job_id, Job.owner_id == owner_id)
        )
        if source_job is None or source_job.kind != JobKind.ANALYZE_SOURCE:
            raise HTTPException(404, "临时原片分析任务不存在")
        if source_job.status != JobStatus.SUCCEEDED or not source_job.result:
            raise HTTPException(409, "临时原片仍在分析中")
        expires_at = datetime.fromisoformat(
            str(source_job.result["expires_at"]).replace("Z", "+00:00")
        )
        if expires_at <= datetime.now(UTC):
            raise HTTPException(410, "临时原片已超过 4 小时保留期")
        source_media = list(source_job.result.get("source_media", []))
        source_segments = list(source_job.result.get("source_segments", []))
        transient_expires_at = expires_at
    audio_catalog = services.object_store.read_json("library/audio/catalog.json", default=[])
    if not isinstance(audio_catalog, list):
        raise RuntimeError("audio library catalog is invalid")
    session, state = StateRepository(db).create(
        owner_id=owner_id,
        goal=request.current_goal,
        asset_id=request.video_id,
        source_media=source_media,
        source_segments=source_segments,
        audio_library=cast(list[dict[str, Any]], audio_catalog),
        transient_expires_at=transient_expires_at,
        project_id=request.project_id,
    )
    return _editing_session_response(session, version=state.version, document=state.document)


@app.get("/api/v1/editing/sessions", response_model=PageResponse)
def list_editing_sessions(
    db: Db,
    owner_id: Owner,
    limit: int = 50,
    cursor: UUID | None = None,
    project_id: UUID | None = None,
) -> PageResponse:
    query = select(EditingSession).where(
        EditingSession.owner_id == owner_id, EditingSession.deleted_at.is_(None)
    )
    if project_id:
        query = query.where(EditingSession.project_id == project_id)
    if cursor:
        query = query.where(EditingSession.id > cursor)
    sessions = db.scalars(
        query.order_by(EditingSession.id).limit(min(100, max(1, limit)) + 1)
    ).all()
    page, next_cursor = _slice_page(sessions, limit)
    items: list[dict[str, Any]] = []
    repository = StateRepository(db)
    for session in page:
        _, state = repository.current(session.id, owner_id=owner_id)
        items.append(
            _editing_session_response(
                session, version=state.version, document=state.document
            ).model_dump(mode="json")
        )
    return PageResponse(items=items, next_cursor=next_cursor)


@app.get("/api/v1/editing/sessions/{session_id}", response_model=EditingSessionResponse)
def get_editing_session(session_id: UUID, db: Db, owner_id: Owner) -> EditingSessionResponse:
    session, state = StateRepository(db).current(session_id, owner_id=owner_id)
    return _editing_session_response(session, version=state.version, document=state.document)


@app.delete("/api/v1/editing/sessions/{session_id}", status_code=204)
def delete_editing_session(session_id: UUID, db: Db, owner_id: Owner) -> Response:
    session = db.scalar(
        select(EditingSession).where(
            EditingSession.id == session_id, EditingSession.owner_id == owner_id
        )
    )
    if session is None:
        raise HTTPException(404, "剪辑会话不存在")
    session.deleted_at = utcnow()
    db.add(
        AuditEvent(
            owner_id=owner_id,
            project_id=session.project_id,
            editing_session_id=session.id,
            event_type="editing_session_deleted",
            metadata_json={"soft_deleted": True},
        )
    )
    db.commit()
    return Response(status_code=204)


@app.get("/api/v1/editing/sessions/{session_id}/versions", response_model=PageResponse)
def list_state_versions(
    session_id: UUID, db: Db, owner_id: Owner, limit: int = 50, cursor: int | None = None
) -> PageResponse:
    StateRepository(db).current(session_id, owner_id=owner_id)
    query = select(StateVersion).where(StateVersion.session_id == session_id)
    if cursor is not None:
        query = query.where(StateVersion.version > cursor)
    versions = db.scalars(
        query.order_by(StateVersion.version).limit(min(100, max(1, limit)) + 1)
    ).all()
    bounded = min(100, max(1, limit))
    page = versions[:bounded]
    return PageResponse(
        items=[
            VersionResponse(
                version=item.version, document=item.document, created_at=item.created_at
            ).model_dump(mode="json")
            for item in page
        ],
        next_cursor=str(page[-1].version) if len(versions) > bounded and page else None,
    )


@app.get("/api/v1/editing/sessions/{session_id}/patches", response_model=PageResponse)
def list_state_patches(
    session_id: UUID, db: Db, owner_id: Owner, limit: int = 50, cursor: UUID | None = None
) -> PageResponse:
    StateRepository(db).current(session_id, owner_id=owner_id)
    query = select(StatePatch).where(StatePatch.session_id == session_id)
    if cursor:
        query = query.where(StatePatch.id > cursor)
    patches = db.scalars(query.order_by(StatePatch.id).limit(min(100, max(1, limit)) + 1)).all()
    page, next_cursor = _slice_page(patches, limit)
    return PageResponse(
        items=[
            PatchResponse(
                patch_id=item.id,
                base_version=item.base_version,
                result_version=item.result_version,
                operations=item.operations,
                actor=item.actor,
                reason=item.reason,
                created_at=item.created_at,
            ).model_dump(mode="json")
            for item in page
        ],
        next_cursor=next_cursor,
    )


@app.get("/api/v1/editing/sessions/{session_id}/audit", response_model=PageResponse)
def list_audit_events(
    session_id: UUID, db: Db, owner_id: Owner, limit: int = 50, cursor: UUID | None = None
) -> PageResponse:
    StateRepository(db).current(session_id, owner_id=owner_id)
    query = select(AuditEvent).where(AuditEvent.editing_session_id == session_id)
    if cursor:
        query = query.where(AuditEvent.id > cursor)
    events = db.scalars(query.order_by(AuditEvent.id).limit(min(100, max(1, limit)) + 1)).all()
    page, next_cursor = _slice_page(events, limit)
    return PageResponse(
        items=[
            {
                "audit_event_id": str(item.id),
                "event_type": item.event_type,
                "metadata": item.metadata_json,
                "created_at": item.created_at.isoformat(),
            }
            for item in page
        ],
        next_cursor=next_cursor,
    )


@app.patch("/api/v1/editing/sessions/{session_id}", response_model=EditingSessionResponse)
def patch_editing_session(
    session_id: UUID, request: PatchEditingSessionRequest, db: Db, owner_id: Owner
) -> EditingSessionResponse:
    state = StateRepository(db).apply(
        session_id=session_id,
        owner_id=owner_id,
        base_version=request.base_version,
        operations=[item.model_dump(exclude_none=False) for item in request.operations],
        actor=owner_id,
        reason=request.reason,
    )
    session = db.get(EditingSession, session_id)
    assert session is not None
    return _editing_session_response(session, version=state.version, document=state.document)


@app.post("/api/v1/editing/sessions/{session_id}/rollback", response_model=EditingSessionResponse)
def rollback_editing_session(
    session_id: UUID, request: RollbackRequest, db: Db, owner_id: Owner
) -> EditingSessionResponse:
    state = StateRepository(db).rollback(
        session_id=session_id,
        owner_id=owner_id,
        base_version=request.base_version,
        target_version=request.target_version,
    )
    session = db.get(EditingSession, session_id)
    assert session is not None
    return _editing_session_response(session, version=state.version, document=state.document)


@app.post("/api/v1/editing/sessions/{session_id}/agent", response_model=EditingSessionResponse)
def run_editing_agent(
    session_id: UUID, request: AgentEditRequest, db: Db, owner_id: Owner
) -> EditingSessionResponse:
    repository = StateRepository(db)
    session, current = repository.current(session_id, owner_id=owner_id)
    if request.base_version != current.version:
        raise PatchConflict(expected_version=request.base_version, current_version=current.version)
    run = GraphRun(
        owner_id=owner_id,
        thread_id=f"edit-{session_id}",
        editing_session_id=session_id,
        state={"instruction": request.instruction, "base_version": request.base_version},
        node_trace=[],
        status=JobStatus.RUNNING,
    )
    db.add(run)
    db.commit()
    try:
        result = services.graph.invoke(
            CamCatState(
                mode="edit",
                query_text=request.instruction,
                query_image_base64=request.query_image_base64 or "",
                explicit_filters=editing_retrieval_filters(
                    base_asset_id=str(session.asset_id) if session.asset_id else None
                ),
                top_k=request.top_k,
                current_document=current.document,
                base_version=current.version,
                session_id=str(session_id),
                owner_id=owner_id,
                persistence_reason=request.instruction,
            )
        )
        run.state = _redact_graph_state(dict(result))
        run.node_trace = result.get("node_trace", [])
        run.status = JobStatus.SUCCEEDED
        run.finished_at = utcnow()
        db.commit()
        db.refresh(session)
        return _editing_session_response(
            session,
            version=result["persisted_version"],
            document=result["persisted_document"],
        )
    except Exception as exc:
        db.rollback()
        failed_run = db.get(GraphRun, run.id)
        if failed_run is not None:
            failed_run.status = JobStatus.FAILED
            failed_run.error = sanitize_job_error(exc)
            failed_run.finished_at = utcnow()
            db.commit()
        raise


@app.post("/api/v1/editing/sessions/{session_id}/agent/stream")
def stream_editing_agent(
    session_id: UUID, request: AgentEditRequest, db: Db, owner_id: Owner
) -> StreamingResponse:
    repository = StateRepository(db)
    session, current = repository.current(session_id, owner_id=owner_id)
    if request.base_version != current.version:
        raise PatchConflict(expected_version=request.base_version, current_version=current.version)
    run = GraphRun(
        owner_id=owner_id,
        thread_id=f"edit-{session_id}",
        editing_session_id=session_id,
        state={"instruction": request.instruction, "base_version": request.base_version},
        node_trace=[],
        status=JobStatus.RUNNING,
    )
    db.add(run)
    db.commit()

    def events() -> Any:
        yield _sse(
            "run_started",
            {"graph_run_id": str(run.id), "message": "已建立剪辑上下文，开始理解需求"},
            event_id=0,
        )
        final_state: CamCatState | None = None
        emitted_nodes = 0
        try:
            for state in services.graph.stream(
                CamCatState(
                    mode="edit",
                    query_text=request.instruction,
                    query_image_base64=request.query_image_base64 or "",
                    explicit_filters=editing_retrieval_filters(
                        base_asset_id=str(session.asset_id) if session.asset_id else None
                    ),
                    top_k=request.top_k,
                    current_document=current.document,
                    base_version=current.version,
                    session_id=str(session_id),
                    owner_id=owner_id,
                    persistence_reason=request.instruction,
                )
            ):
                final_state = cast(CamCatState, state)
                trace = list(state.get("node_trace", []))
                for item in trace[emitted_nodes:]:
                    yield _sse(
                        "node_completed",
                        {
                            "graph_run_id": str(run.id),
                            "node": item.get("node_name"),
                            "status": item.get("status"),
                            "duration_ms": item.get("duration_ms"),
                            "message": _node_message(str(item.get("node_name", "")), state),
                        },
                        event_id=emitted_nodes + 1,
                    )
                emitted_nodes = len(trace)
            if final_state is None:
                raise RuntimeError("agent graph returned no state")
            run.state = _redact_graph_state(dict(final_state))
            run.node_trace = final_state.get("node_trace", [])
            run.status = JobStatus.SUCCEEDED
            run.finished_at = utcnow()
            db.commit()
            db.refresh(session)
            yield _sse(
                "completed",
                {
                    "graph_run_id": str(run.id),
                    "message": "剪辑计划已通过 State Patch 原子写入",
                    "session": _editing_session_response(
                        session,
                        version=final_state["persisted_version"],
                        document=final_state["persisted_document"],
                    ).model_dump(mode="json"),
                    "agent_run": _search_response(run, final_state, db).model_dump(mode="json"),
                },
            )
        except Exception as exc:
            db.rollback()
            failed = db.get(GraphRun, run.id)
            public_error = sanitize_job_error(exc)
            if failed is not None:
                failed.status = JobStatus.FAILED
                failed.error = public_error
                failed.finished_at = utcnow()
                db.commit()
            yield _sse("failed", {"graph_run_id": str(run.id), "message": public_error})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/api/v1/editing/sessions/{session_id}/render", response_model=JobResponse, status_code=202
)
def render_editing_session(
    session_id: UUID, request: RenderRequest, db: Db, owner_id: Owner
) -> JobResponse:
    _, current = StateRepository(db).current(session_id, owner_id=owner_id)
    if current.version != request.base_version:
        raise PatchConflict(expected_version=request.base_version, current_version=current.version)
    if not current.document.get("clips"):
        raise HTTPException(422, "剪辑计划没有片段，无法渲染")
    configured_ratio = str(current.document.get("settings", {}).get("aspect_ratio", "16:9"))
    if configured_ratio == "auto":
        source_media = current.document.get("source_media") or []
        first_source = source_media[0] if source_media else {}
        configured_ratio = choose_aspect_ratio(
            str(current.document.get("goal", "")),
            int(first_source.get("width") or 0),
            int(first_source.get("height") or 0),
        )
    job = JobRepository(db).enqueue(
        owner_id=owner_id,
        kind=JobKind.RENDER,
        payload={
            "session_id": str(session_id),
            "version": current.version,
            "resolution": request.resolution
            or "x".join(str(value) for value in resolution_for_ratio(configured_ratio)),
            "burn_subtitles": request.burn_subtitles,
        },
    )
    return JobResponse(
        job_id=job.id,
        kind=job.kind.value,
        status=job.status.value,
        progress=job.progress,
        created_at=job.created_at,
    )


def _sse(event: str, payload: dict[str, Any], event_id: int | None = None) -> str:
    import json

    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _node_message(node: str, state: dict[str, Any]) -> str:
    messages = {
        "understand_requirement": "已解析发布目标、节奏与画幅",
        "plan_query": "已规划素材库多路召回策略",
        "retrieve_material": f"已检索并重排 {len(state.get('ranked_materials', []))} 个补充素材",
        "generate_edit_plan": f"已完成 {len(state.get('edit_plan', []))} 个镜头的逻辑重排",
        "generate_subtitles": f"已生成 {len(state.get('subtitles', []))} 条字幕",
        "validate_patch": "已校验原片主体、25% 素材上限与乐观锁补丁",
    }
    return messages.get(node, f"{node} 已完成")


def _copy_upload(upload: UploadFile, target: Path, maximum: int) -> int:
    size = 0
    try:
        with target.open("wb") as destination:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise HTTPException(413, "视频超过上传大小限制")
                destination.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(422, "视频文件为空")
    return size


def _validate_video_file(path: Path) -> None:
    with path.open("rb") as stream:
        signature = stream.read(16)
    recognized = (
        (len(signature) >= 8 and signature[4:8] == b"ftyp")
        or signature.startswith(b"\x1aE\xdf\xa3")
        or signature.startswith(b"\x00\x00\x01\xba")
    )
    if not recognized:
        path.unlink(missing_ok=True)
        raise HTTPException(415, "文件内容不是受支持的视频容器")
    try:
        metadata = probe(path)
    except MediaCommandError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "ffprobe 无法验证视频流") from exc
    if metadata.width <= 0 or metadata.height <= 0:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "视频尺寸无效")


def _enforce_owner_quota(db: Session, owner_id: str, incoming_bytes: int) -> None:
    permanent = int(
        db.scalar(
            select(func.coalesce(func.sum(Asset.size_bytes), 0)).where(Asset.owner_id == owner_id)
        )
        or 0
    )
    active_jobs = db.scalars(
        select(Job).where(Job.owner_id == owner_id, Job.expires_at > datetime.now(UTC))
    ).all()
    transient = sum(
        int(item.get("size_bytes", 0))
        for job in active_jobs
        for item in job.payload.get("source_media", [])
    )
    if permanent + transient + incoming_bytes > settings.upload_owner_quota_bytes:
        raise HTTPException(413, "用户媒体配额不足")


def _video_response(asset: Asset, db: Session, job: Job | None = None) -> VideoResponse:
    segment_count = int(
        db.scalar(select(func.count(Segment.id)).where(Segment.asset_id == asset.id)) or 0
    )
    if job is None:
        job = db.scalar(
            select(Job)
            .where(Job.payload["asset_id"].as_string() == str(asset.id))
            .order_by(Job.created_at.desc())
            .limit(1)
        )
    return VideoResponse(
        video_id=asset.id,
        status=asset.status.value,
        filename=asset.filename,
        segment_count=segment_count,
        duration_seconds=asset.duration_seconds,
        job_id=job.id if job else None,
        playback_url=services.object_store.signed_url(asset.storage_key),
        error=asset.error,
    )


def _search_response(run: GraphRun, state: CamCatState, db: Session) -> AgenticSearchResponse:
    ranked: list[RankedSegmentResponse] = []
    for item in state.get("ranked_materials", []):
        segment = db.get(Segment, UUID(item["segment_id"]))
        if segment is None:
            continue
        asset = db.get(Asset, segment.asset_id)
        if asset is None:
            continue
        ranked.append(
            RankedSegmentResponse(
                segment_id=segment.id,
                video_id=segment.asset_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                score=item["score"],
                reranker_score=item["reranker_score"],
                caption=segment.description_text,
                tags=segment.tags,
                route_scores=item["route_scores"],
                route_ranks=item["route_ranks"],
                thumbnail_url=services.object_store.signed_url(segment.thumbnail_key)
                if segment.thumbnail_key
                else None,
                source_video_url=services.object_store.signed_url(segment.storage_key),
                event_type=segment.event_type,
                risk_score=segment.risk_score,
                semantic_metadata=segment.semantic_metadata,
                license_name=asset.license_name,
                source_url=asset.source_url,
            )
        )
    return AgenticSearchResponse(
        graph_run_id=run.id,
        thread_id=run.thread_id,
        final_answer=state.get("final_answer", "检索已完成。"),
        route_sequence=state.get("route_sequence", []),
        node_trace=state.get("node_trace", []),
        ranked_segments=ranked,
    )


def _error(
    request: Request,
    *,
    status: int,
    code: str,
    message: str,
    details: dict[str, Any],
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        },
    )


def _redact_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(state)
    image = redacted.pop("query_image_base64", None)
    if isinstance(image, str) and image:
        redacted["query_image_sha256"] = hashlib.sha256(image.encode()).hexdigest()
    for key in (
        "current_document",
        "persisted_document",
        "source_materials",
        "edit_plan",
        "subtitles",
        "patch_operations",
    ):
        sensitive = redacted.pop(key, None)
        if sensitive is not None:
            encoded = repr(sensitive).encode()
            redacted[f"{key}_sha256"] = hashlib.sha256(encoded).hexdigest()
    return redacted


def _slice_page(items: Sequence[Any], limit: int) -> tuple[list[Any], str | None]:
    bounded = min(100, max(1, limit))
    page = list(items[:bounded])
    next_cursor = str(page[-1].id) if len(items) > bounded and page else None
    return page, next_cursor


def _graph_run_response(run: GraphRun) -> GraphRunResponse:
    return GraphRunResponse(
        graph_run_id=run.id,
        thread_id=run.thread_id,
        editing_session_id=run.editing_session_id,
        status=run.status.value,
        state=run.state,
        node_trace=run.node_trace,
        error=run.error,
        created_at=run.created_at,
        finished_at=run.finished_at,
    )
