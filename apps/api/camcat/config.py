from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="CAMCAT_", case_sensitive=False, extra="ignore"
    )

    environment: str = "development"
    security_mode: str = "local-single-user"
    local_user_id: str = "camcat-local-user"
    trusted_proxy_secret: SecretStr = SecretStr("")
    library_admin_key: SecretStr = SecretStr("")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    public_api_url: str = "http://localhost:8000"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]

    database_url: str = "postgresql+psycopg://camcat:camcat@postgres:5432/camcat"
    object_store_endpoint: str = "http://minio:9000"
    object_store_public_endpoint: str = "http://localhost:9000"
    object_store_access_key: SecretStr = SecretStr("camcat")
    object_store_secret_key: SecretStr = SecretStr("change-me")
    object_store_bucket: str = "camcat"
    object_store_region: str = "us-east-1"

    milvus_uri: str = "http://milvus:19530"
    milvus_token: SecretStr = SecretStr("")
    # v7 adds source/license/semantic fields required for explainable retrieval.
    # Do not repurpose an older collection in place: it can contain incompatible rows.
    milvus_collection: str = "camcat_segments_v7"

    embedding_base_url: str
    embedding_api_key: SecretStr
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    embedding_dimension: int = 2048
    embedding_video_fps: float = 1.0
    reranker_base_url: str
    reranker_api_key: SecretStr
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-8B"
    llm_base_url: str
    llm_api_key: SecretStr
    llm_model: str = "qwen3-vl-plus"
    asr_base_url: str
    asr_api_key: SecretStr
    asr_model: str = "qwen3-asr-flash"

    provider_gateway_api_key: SecretStr = SecretStr("")
    bailian_api_host: str = ""
    bailian_api_key: SecretStr = SecretStr("")

    provider_timeout_seconds: float = 120.0
    provider_max_retries: int = 2
    upload_max_bytes: int = 250 * 1024 * 1024
    upload_owner_quota_bytes: int = 2 * 1024 * 1024 * 1024
    segment_window_seconds: float = 5.0
    segment_minimum_seconds: float = 3.0
    runtime_dir: str = "/var/lib/camcat"
    worker_poll_seconds: float = 1.0
    worker_id: str = "worker-1"
    job_lease_seconds: int = 600

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, value: int) -> int:
        if value != 2048:
            raise ValueError("CamCat's Milvus schema requires 2048 embedding dimensions")
        return value

    @field_validator("embedding_base_url", "reranker_base_url", "llm_base_url", "asr_base_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("provider URLs must be absolute HTTP(S) URLs")
        return value

    @model_validator(mode="after")
    def validate_security_mode(self) -> Settings:
        if self.security_mode not in {"local-single-user", "multi-user"}:
            raise ValueError("security_mode must be local-single-user or multi-user")
        if self.security_mode == "multi-user":
            if not self.trusted_proxy_secret.get_secret_value():
                raise ValueError("multi-user mode requires CAMCAT_TRUSTED_PROXY_SECRET")
            if not self.library_admin_key.get_secret_value():
                raise ValueError("multi-user mode requires CAMCAT_LIBRARY_ADMIN_KEY")
            if self.provider_gateway_api_key.get_secret_value() == "camcat-local-gateway":
                raise ValueError(
                    "multi-user mode requires a private CAMCAT_PROVIDER_GATEWAY_API_KEY"
                )
        provider_urls = (
            self.embedding_base_url,
            self.reranker_base_url,
            self.llm_base_url,
            self.asr_base_url,
        )
        if self.security_mode == "local-single-user" and any(
            "provider-gateway" in url for url in provider_urls
        ):
            gateway_key = self.provider_gateway_api_key.get_secret_value() or "camcat-local-gateway"
            self.provider_gateway_api_key = SecretStr(gateway_key)
            if not self.embedding_api_key.get_secret_value():
                self.embedding_api_key = SecretStr(gateway_key)
            if not self.reranker_api_key.get_secret_value():
                self.reranker_api_key = SecretStr(gateway_key)
            if not self.llm_api_key.get_secret_value():
                self.llm_api_key = SecretStr(gateway_key)
            if not self.asr_api_key.get_secret_value():
                self.asr_api_key = SecretStr(gateway_key)
        if self.environment != "test":
            provider_values = {
                "CAMCAT_EMBEDDING_BASE_URL": self.embedding_base_url,
                "CAMCAT_EMBEDDING_API_KEY": self.embedding_api_key.get_secret_value(),
                "CAMCAT_RERANKER_BASE_URL": self.reranker_base_url,
                "CAMCAT_RERANKER_API_KEY": self.reranker_api_key.get_secret_value(),
                "CAMCAT_LLM_BASE_URL": self.llm_base_url,
                "CAMCAT_LLM_API_KEY": self.llm_api_key.get_secret_value(),
                "CAMCAT_ASR_BASE_URL": self.asr_base_url,
                "CAMCAT_ASR_API_KEY": self.asr_api_key.get_secret_value(),
            }
            placeholders = [
                name
                for name, value in provider_values.items()
                if not value.strip()
                or "change-me" in value.lower()
                or "placeholder" in value.lower()
                or ".invalid" in value.lower()
            ]
            if placeholders:
                raise ValueError(
                    "placeholder provider configuration is forbidden outside test mode: "
                    + ", ".join(placeholders)
                )
            provider_url_set = set(provider_urls)
            if any("provider-gateway" in url for url in provider_url_set):
                gateway_key = self.provider_gateway_api_key.get_secret_value()
                bailian_key = self.bailian_api_key.get_secret_value()
                if not gateway_key or not self.bailian_api_host or not bailian_key:
                    raise ValueError(
                        "managed Bailian gateway requires CAMCAT_PROVIDER_GATEWAY_API_KEY, "
                        "CAMCAT_BAILIAN_API_HOST and CAMCAT_BAILIAN_API_KEY"
                    )
                bailian_values = {
                    "CAMCAT_PROVIDER_GATEWAY_API_KEY": gateway_key,
                    "CAMCAT_BAILIAN_API_HOST": self.bailian_api_host,
                    "CAMCAT_BAILIAN_API_KEY": bailian_key,
                }
                bailian_placeholders = [
                    name
                    for name, value in bailian_values.items()
                    if "change-me" in value.lower()
                    or "placeholder" in value.lower()
                    or ".example" in value.lower()
                    or ".invalid" in value.lower()
                ]
                if bailian_placeholders:
                    raise ValueError(
                        "placeholder Bailian gateway configuration is forbidden: "
                        + ", ".join(bailian_placeholders)
                    )
                if not self.bailian_api_host.startswith(("http://", "https://")):
                    raise ValueError("CAMCAT_BAILIAN_API_HOST must be an absolute HTTP(S) URL")
                provider_keys = {
                    self.embedding_api_key.get_secret_value(),
                    self.reranker_api_key.get_secret_value(),
                    self.llm_api_key.get_secret_value(),
                    self.asr_api_key.get_secret_value(),
                }
                if provider_keys != {gateway_key}:
                    raise ValueError(
                        "managed Bailian gateway provider API keys must equal "
                        "CAMCAT_PROVIDER_GATEWAY_API_KEY"
                    )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
