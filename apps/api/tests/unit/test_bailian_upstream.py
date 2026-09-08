from __future__ import annotations

import json

import httpx
import pytest
from camcat.gateway.upstream import BailianHttpUpstream, BailianUpstreamError


def test_bailian_upstream_authenticates_and_retries_transient_responses() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.headers["Authorization"] == "Bearer rotated-secret"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {"model": "qwen3-vl-plus"}
        if attempts == 1:
            return httpx.Response(503, json={"message": "temporarily unavailable"})
        return httpx.Response(200, json={"choices": []})

    upstream = BailianHttpUpstream(
        base_url="https://workspace.example",
        api_key="rotated-secret",
        timeout_seconds=5,
        max_retries=1,
        transport=httpx.MockTransport(handler),
        sleeper=delays.append,
    )

    result = upstream.post_json("/compatible-mode/v1/chat/completions", {"model": "qwen3-vl-plus"})
    assert result == {"choices": []}
    assert attempts == 2
    assert delays == [0.25]


def test_bailian_upstream_rejects_non_object_json_without_retrying() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json=["not", "an", "object"])

    upstream = BailianHttpUpstream(
        base_url="https://workspace.example/",
        api_key="rotated-secret",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(BailianUpstreamError, match="non-object JSON"):
        upstream.post_json("/api/v1/test", {})
    assert attempts == 1


def test_bailian_upstream_fails_fast_for_missing_credentials() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        BailianHttpUpstream(base_url="https://workspace.example", api_key="")
