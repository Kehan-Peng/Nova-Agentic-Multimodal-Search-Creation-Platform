from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from camcat.config import Settings


class ObjectStore:
    def __init__(self, settings: Settings) -> None:
        credentials = {
            "aws_access_key_id": settings.object_store_access_key.get_secret_value(),
            "aws_secret_access_key": settings.object_store_secret_key.get_secret_value(),
            "region_name": settings.object_store_region,
            "config": Config(signature_version="s3v4", retries={"max_attempts": 3}),
        }
        self._client = boto3.client(
            "s3", endpoint_url=settings.object_store_endpoint, **credentials
        )
        self._public_client = boto3.client(
            "s3", endpoint_url=settings.object_store_public_endpoint, **credentials
        )
        self.bucket = settings.object_store_bucket

    def ensure_bucket(self) -> None:
        buckets = {item["Name"] for item in self._client.list_buckets().get("Buckets", [])}
        existing_rules: list[dict[str, object]] = []
        if self.bucket not in buckets:
            self._client.create_bucket(Bucket=self.bucket)
        else:
            try:
                lifecycle = self._client.get_bucket_lifecycle_configuration(Bucket=self.bucket)
                existing_rules = [
                    dict(rule)
                    for rule in lifecycle.get("Rules", [])
                    if rule.get("ID") != "expire-transient-user-media"
                ]
            except ClientError as exc:
                error_code = str(exc.response.get("Error", {}).get("Code", ""))
                if error_code not in {
                    "404",
                    "NoSuchLifecycle",
                    "NoSuchLifecycleConfiguration",
                }:
                    raise
        # S3 lifecycle is the object-deletion source of truth. S3/MinIO lifecycle
        # exposes whole-day granularity, while the application redacts database
        # references at the exact four-hour boundary.
        self._client.put_bucket_lifecycle_configuration(
            Bucket=self.bucket,
            LifecycleConfiguration={
                "Rules": [
                    *existing_rules,
                    {
                        "ID": "expire-transient-user-media",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "temporary/"},
                        "Expiration": {"Days": 1},
                        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                    },
                ]
            },
        )

    def upload_file(
        self, path: Path, key: str, content_type: str, *, metadata: dict[str, str] | None = None
    ) -> None:
        self._client.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type, **({"Metadata": metadata} if metadata else {})},
        )

    def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        content_type: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._client.upload_fileobj(
            stream,
            self.bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                **({"Metadata": metadata} if metadata else {}),
            },
        )

    def download_file(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(path))

    def signed_url(self, key: str, expires_seconds: int = 3600) -> str:
        return str(
            self._public_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )
        )

    def signed_download_url(self, key: str, filename: str, expires_seconds: int = 3600) -> str:
        return str(
            self._public_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": key,
                    "ResponseContentDisposition": f'attachment; filename="{filename}"',
                },
                ExpiresIn=expires_seconds,
            )
        )

    def healthcheck(self) -> None:
        self._client.head_bucket(Bucket=self.bucket)

    def write_json(self, key: str, value: object) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(value, ensure_ascii=False).encode(),
            ContentType="application/json",
        )

    def read_json(self, key: str, default: object) -> object:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return default
            raise
        return json.loads(response["Body"].read())

    def delete_key(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> int:
        deleted = 0
        token: str | None = None
        while True:
            params: dict[str, object] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                params["ContinuationToken"] = token
            page = self._client.list_objects_v2(**params)
            objects = [{"Key": str(item["Key"])} for item in page.get("Contents", [])]
            if objects:
                self._client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": objects, "Quiet": True}
                )
                deleted += len(objects)
            if not page.get("IsTruncated"):
                return deleted
            token = str(page["NextContinuationToken"])
