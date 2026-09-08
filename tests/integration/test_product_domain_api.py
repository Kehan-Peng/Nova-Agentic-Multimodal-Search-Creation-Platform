from __future__ import annotations

from uuid import uuid4

import pytest
from camcat.api import app
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_project_session_patch_conflict_audit_and_soft_delete_journey() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        project = client.post("/api/v1/projects", json={"name": f"integration-{uuid4()}"})
        assert project.status_code == 201
        project_id = project.json()["project_id"]

        projects = client.get("/api/v1/projects", params={"limit": 1})
        assert projects.status_code == 200
        assert "items" in projects.json() and "next_cursor" in projects.json()

        created = client.post(
            "/api/v1/editing/sessions",
            json={"project_id": project_id, "current_goal": "integration edit"},
        )
        assert created.status_code == 201
        session_id = created.json()["editing_session_id"]

        updated = client.patch(
            f"/api/v1/editing/sessions/{session_id}",
            json={
                "base_version": 1,
                "operations": [{"op": "replace", "path": "/title", "value": "v2"}],
                "reason": "integration patch",
            },
        )
        assert updated.status_code == 200
        assert updated.json()["state_version"] == 2

        conflict = client.patch(
            f"/api/v1/editing/sessions/{session_id}",
            json={
                "base_version": 1,
                "operations": [{"op": "replace", "path": "/title", "value": "stale"}],
                "reason": "stale patch",
            },
        )
        assert conflict.status_code == 409
        details = conflict.json()["error"]["details"]
        assert details["expected_version"] == 1
        assert details["current_version"] == 2
        assert details["current_patch"]["result_version"] == 2

        versions = client.get(f"/api/v1/editing/sessions/{session_id}/versions")
        patches = client.get(f"/api/v1/editing/sessions/{session_id}/patches")
        audit = client.get(f"/api/v1/editing/sessions/{session_id}/audit")
        assert [item["version"] for item in versions.json()["items"]] == [1, 2]
        assert len(patches.json()["items"]) == 1
        assert {item["event_type"] for item in audit.json()["items"]} >= {
            "editing_session_created",
            "state_patch_applied",
        }

        deleted = client.delete(f"/api/v1/editing/sessions/{session_id}")
        assert deleted.status_code == 204
        missing = client.get(f"/api/v1/editing/sessions/{session_id}")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"
