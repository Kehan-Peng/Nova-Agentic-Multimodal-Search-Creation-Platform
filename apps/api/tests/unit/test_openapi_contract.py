from __future__ import annotations

from camcat.api import app


def _schema_ref(schema: dict[str, object]) -> str:
    return str(schema["$ref"])


def test_every_asset_list_uses_the_cursor_page_envelope() -> None:
    schema = app.openapi()
    response = schema["paths"]["/api/v1/videos"]["get"]["responses"]["200"]
    body_schema = response["content"]["application/json"]["schema"]

    assert _schema_ref(body_schema).endswith("/PageResponse")


def test_openapi_documents_the_shared_error_envelope() -> None:
    schema = app.openapi()
    response = schema["paths"]["/api/v1/editing/sessions/{session_id}"]["patch"]["responses"]["409"]
    body_schema = response["content"]["application/json"]["schema"]

    assert _schema_ref(body_schema).endswith("/ErrorEnvelope")
