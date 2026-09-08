from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Select, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from camcat.domain.state_patch import (
    PatchConflict,
    VersionedState,
    apply_versioned_patch,
    build_rollback_patch,
)
from camcat.models import (
    AuditEvent,
    EditingSession,
    Job,
    JobKind,
    JobStatus,
    StatePatch,
    StateVersion,
    utcnow,
)


def redact_transient_document(document: dict[str, Any]) -> dict[str, Any]:
    """Remove all user-source media and derived analysis after its retention TTL."""

    redacted = deepcopy(document)
    redacted["source_media"] = []
    redacted["source_segments"] = []
    redacted["clips"] = [
        item for item in redacted.get("clips", []) if item.get("origin") != "source"
    ]
    redacted["transient_source_status"] = "expired"
    return redacted


def sanitize_job_error(error: Exception) -> str:
    """Return a stable public failure message without paths, frames or traceback text."""

    message = str(error).strip() or error.__class__.__name__
    return message.splitlines()[0][:1000]


class StateRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        owner_id: str,
        goal: str,
        asset_id: UUID | None = None,
        source_media: list[dict[str, Any]] | None = None,
        source_segments: list[dict[str, Any]] | None = None,
        audio_library: list[dict[str, Any]] | None = None,
        project_id: UUID | None = None,
        transient_expires_at: datetime | None = None,
    ) -> tuple[EditingSession, VersionedState]:
        session = EditingSession(
            owner_id=owner_id,
            project_id=project_id,
            asset_id=asset_id,
            current_version=1,
            transient_expires_at=transient_expires_at,
        )
        self.db.add(session)
        self.db.flush()
        document: dict[str, Any] = {
            "title": "未命名剪辑",
            "goal": goal,
            "target_duration": 30.0,
            "clips": [],
            "subtitles": [],
            "source_media": source_media or [],
            "source_segments": source_segments or [],
            "audio_library": audio_library or [],
            "audio_plan": {
                "normalize_loudness": True,
                "target_lufs": -14,
                "duck_music_under_dialogue": True,
                "bgm": [],
                "ambient": [],
                "sound_effects": [],
            },
            "settings": {
                "aspect_ratio": "auto",
                "burn_subtitles": True,
                "external_material_ratio_limit": 0.25,
                "transitions": True,
                "loudness_normalization": True,
                "basic_color_grade": True,
                "safe_area": True,
            },
        }
        self.db.add(StateVersion(session_id=session.id, version=1, document=document))
        self.db.add(
            AuditEvent(
                owner_id=owner_id,
                project_id=project_id,
                editing_session_id=session.id,
                event_type="editing_session_created",
                metadata_json={"version": 1},
            )
        )
        self.db.commit()
        return session, VersionedState(str(session.id), 1, document)

    def current(
        self,
        session_id: UUID,
        *,
        owner_id: str | None = None,
        include_deleted: bool = False,
    ) -> tuple[EditingSession, VersionedState]:
        query: Select[tuple[EditingSession]] = select(EditingSession).where(
            EditingSession.id == session_id
        )
        if owner_id is not None:
            query = query.where(EditingSession.owner_id == owner_id)
        if not include_deleted:
            query = query.where(EditingSession.deleted_at.is_(None))
        session = self.db.scalar(query)
        if session is None:
            raise LookupError("editing session not found")
        version = self.db.scalar(
            select(StateVersion).where(
                StateVersion.session_id == session.id,
                StateVersion.version == session.current_version,
            )
        )
        if version is None:
            raise RuntimeError("editing session current version is missing")
        return session, VersionedState(str(session.id), version.version, version.document)

    def apply(
        self,
        *,
        session_id: UUID,
        owner_id: str,
        base_version: int,
        operations: list[dict[str, Any]],
        actor: str,
        reason: str,
    ) -> VersionedState:
        _, before = self.current(session_id, owner_id=owner_id)
        if base_version != before.version:
            raise PatchConflict(
                expected_version=base_version,
                current_version=before.version,
                current_patch=self._latest_patch_metadata(session_id),
            )
        after, audit = apply_versioned_patch(
            before,
            base_version=base_version,
            operations=operations,
            actor=actor,
            reason=reason,
        )
        result = self.db.execute(
            update(EditingSession)
            .where(
                EditingSession.id == session_id,
                EditingSession.owner_id == owner_id,
                EditingSession.current_version == base_version,
            )
            .values(current_version=after.version, updated_at=utcnow())
        )
        if not isinstance(result, CursorResult) or result.rowcount != 1:
            self.db.rollback()
            current_version = self.db.scalar(
                select(EditingSession.current_version).where(EditingSession.id == session_id)
            )
            raise PatchConflict(
                expected_version=base_version,
                current_version=int(current_version or before.version),
                current_patch=self._latest_patch_metadata(session_id),
            )
        self.db.add(
            StateVersion(session_id=session_id, version=after.version, document=after.document)
        )
        self.db.add(
            AuditEvent(
                owner_id=owner_id,
                project_id=self.db.scalar(
                    select(EditingSession.project_id).where(EditingSession.id == session_id)
                ),
                editing_session_id=session_id,
                event_type="state_patch_applied",
                metadata_json={
                    "base_version": base_version,
                    "result_version": after.version,
                    "patch_id": audit.patch_id,
                    "actor": actor,
                    "reason": reason,
                },
            )
        )
        self.db.add(
            StatePatch(
                id=UUID(audit.patch_id),
                session_id=session_id,
                base_version=audit.base_version,
                result_version=audit.result_version,
                operations=[
                    {"op": operation.op, "path": operation.path, "value": operation.value}
                    for operation in audit.operations
                ],
                actor=actor,
                reason=reason,
            )
        )
        self.db.commit()
        return after

    def _latest_patch_metadata(self, session_id: UUID) -> dict[str, Any] | None:
        latest_patch = self.db.scalar(
            select(StatePatch)
            .where(StatePatch.session_id == session_id)
            .order_by(StatePatch.result_version.desc())
            .limit(1)
        )
        if latest_patch is None:
            return None
        return {
            "patch_id": str(latest_patch.id),
            "base_version": latest_patch.base_version,
            "result_version": latest_patch.result_version,
            "operations": latest_patch.operations,
            "actor": latest_patch.actor,
            "reason": latest_patch.reason,
        }

    def rollback(
        self,
        *,
        session_id: UUID,
        owner_id: str,
        base_version: int,
        target_version: int,
    ) -> VersionedState:
        _, current = self.current(session_id, owner_id=owner_id)
        target = self.db.scalar(
            select(StateVersion).where(
                StateVersion.session_id == session_id, StateVersion.version == target_version
            )
        )
        if target is None:
            raise LookupError("target state version not found")
        operations = build_rollback_patch(current.document, target.document)
        return self.apply(
            session_id=session_id,
            owner_id=owner_id,
            base_version=base_version,
            operations=operations,
            actor=owner_id,
            reason=f"rollback to version {target_version}",
        )


class JobRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def enqueue(
        self,
        *,
        owner_id: str,
        kind: Any,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        expires_at: datetime | None = None,
        max_attempts: int = 3,
    ) -> Job:
        if idempotency_key:
            existing = self.db.scalar(
                select(Job).where(
                    Job.owner_id == owner_id,
                    Job.kind == kind,
                    Job.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                return existing
        job = Job(
            owner_id=owner_id,
            kind=kind,
            status=JobStatus.QUEUED,
            payload=payload,
            idempotency_key=idempotency_key,
            expires_at=expires_at,
            max_attempts=max_attempts,
            available_at=utcnow(),
        )
        self.db.add(job)
        self.db.commit()
        return job

    def claim_next(self, *, worker_id: str, lease_seconds: int = 600) -> Job | None:
        now = utcnow()
        job = self.db.scalar(
            select(Job)
            .where(
                Job.cancel_requested_at.is_(None),
                Job.attempts < Job.max_attempts,
                or_(
                    (Job.status == JobStatus.QUEUED) & (Job.available_at <= now),
                    (Job.status == JobStatus.RUNNING) & (Job.lease_expires_at < now),
                ),
            )
            .order_by(Job.available_at, Job.created_at, Job.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            self.db.rollback()
            return None
        job.status = JobStatus.RUNNING
        job.worker_id = worker_id
        job.started_at = job.started_at or now
        job.heartbeat_at = now
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.attempts += 1
        job.progress = 0.01
        self.db.commit()
        return job

    def expire_exhausted_leases(self, *, now: datetime | None = None) -> list[Job]:
        cutoff = now or utcnow()
        jobs = self.db.scalars(
            select(Job).where(
                Job.status == JobStatus.RUNNING,
                Job.lease_expires_at < cutoff,
                Job.attempts >= Job.max_attempts,
            )
        ).all()
        for job in jobs:
            job.status = JobStatus.DEAD_LETTER
            job.worker_id = None
            job.lease_expires_at = None
            job.finished_at = cutoff
            checkpoint = dict(job.checkpoint or {})
            checkpoint.update({"stage": "dead_lettered", "updated_at": cutoff.isoformat()})
            job.checkpoint = checkpoint
        self.db.commit()
        return list(jobs)

    def pending_ingest_compensations(self) -> list[Job]:
        stage = Job.checkpoint["stage"].as_string()
        return list(
            self.db.scalars(
                select(Job)
                .where(
                    Job.kind == JobKind.INGEST_MEDIA,
                    Job.status == JobStatus.DEAD_LETTER,
                    stage.is_distinct_from("compensated"),
                )
                .order_by(Job.finished_at, Job.id)
            ).all()
        )

    def update_progress(self, job: Job, progress: float, *, lease_seconds: int = 600) -> None:
        self.db.refresh(job)
        if job.cancel_requested_at is not None:
            job.status = JobStatus.CANCELLED
            job.finished_at = utcnow()
            self.db.commit()
            raise RuntimeError("job cancellation requested")
        now = utcnow()
        job.progress = min(1.0, max(0.0, progress))
        job.heartbeat_at = now
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        self.db.commit()

    def checkpoint(self, job: Job, stage: str, data: dict[str, Any] | None = None) -> None:
        checkpoint = dict(job.checkpoint or {})
        checkpoint.update({"stage": stage, "updated_at": utcnow().isoformat()})
        if data:
            checkpoint["data"] = data
        job.checkpoint = checkpoint
        self.update_progress(job, job.progress)

    def succeed(self, job: Job, result: dict[str, Any]) -> None:
        job.status = JobStatus.SUCCEEDED
        job.progress = 1.0
        job.result = result
        job.finished_at = utcnow()
        job.lease_expires_at = None
        self.db.commit()

    def fail(self, job: Job, error: str) -> None:
        now = utcnow()
        job.error = error[:1000]
        job.worker_id = None
        job.lease_expires_at = None
        if job.cancel_requested_at is not None:
            job.status = JobStatus.CANCELLED
            job.finished_at = now
        elif job.attempts < job.max_attempts:
            job.status = JobStatus.QUEUED
            job.available_at = now + timedelta(seconds=min(300, 2**job.attempts))
        else:
            job.status = JobStatus.DEAD_LETTER
            job.finished_at = now
        self.db.commit()

    def cancel(self, job: Job) -> Job:
        now = utcnow()
        job.cancel_requested_at = now
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.finished_at = now
        self.db.commit()
        return job

    def retry(self, job: Job) -> Job:
        if job.status not in {JobStatus.FAILED, JobStatus.DEAD_LETTER, JobStatus.CANCELLED}:
            raise ValueError("only terminal jobs can be retried")
        job.status = JobStatus.QUEUED
        job.available_at = utcnow()
        job.attempts = 0
        job.finished_at = None
        job.error = None
        job.cancel_requested_at = None
        self.db.commit()
        return job

    def redact_expired(self, *, now: datetime | None = None) -> tuple[int, int]:
        cutoff = now or utcnow()
        sessions = self.db.scalars(
            select(EditingSession).where(
                EditingSession.transient_expires_at <= cutoff,
                EditingSession.expired_at.is_(None),
            )
        ).all()
        for session in sessions:
            versions = self.db.scalars(
                select(StateVersion).where(StateVersion.session_id == session.id)
            ).all()
            for version in versions:
                version.document = redact_transient_document(version.document)
            patches = self.db.scalars(
                select(StatePatch).where(StatePatch.session_id == session.id)
            ).all()
            for patch in patches:
                operations = deepcopy(patch.operations)
                for operation in operations:
                    if operation.get("path") == "/clips" and isinstance(
                        operation.get("value"), list
                    ):
                        operation["value"] = [
                            item for item in operation["value"] if item.get("origin") != "source"
                        ]
                patch.operations = operations
            session.expired_at = cutoff
        jobs = self.db.scalars(
            select(Job).where(Job.expires_at <= cutoff, Job.redacted_at.is_(None))
        ).all()
        for job in jobs:
            job.payload = {"redacted": True, "expired_at": cutoff.isoformat()}
            job.result = {"redacted": True, "expired_at": cutoff.isoformat()}
            job.redacted_at = cutoff
        self.db.commit()
        return len(sessions), len(jobs)
