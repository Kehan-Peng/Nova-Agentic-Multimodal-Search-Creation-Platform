from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from camcat.config import Settings


@dataclass(frozen=True, slots=True)
class MilvusHit:
    segment_id: str
    score: float
    entity: dict[str, Any]


class MilvusSegmentStore:
    output_fields = [
        "segment_id",
        "asset_id",
        "storage_key",
        "start_time",
        "end_time",
        "duration",
        "trigger_type",
        "event_type",
        "tags",
        "description_text",
        "risk_score",
        "created_at_epoch",
        "embedding_model",
        "embedding_dimension",
        "semantic_metadata",
        "license_name",
        "source_url",
    ]

    def __init__(self, settings: Settings) -> None:
        self._uri = settings.milvus_uri
        self._token = settings.milvus_token.get_secret_value() or None
        self._client: MilvusClient | None = None
        self.collection = settings.milvus_collection
        self.dimension = settings.embedding_dimension
        self.embedding_model = settings.embedding_model

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri, token=self._token)
        return self._client

    def ensure_collection(self) -> None:
        if self.client.has_collection(collection_name=self.collection):
            self._validate_collection()
            self._wait_for_required_indexes()
            self.client.load_collection(collection_name=self.collection)
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="segment_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(field_name="asset_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="storage_key", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(
            field_name="multimodal_embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )
        schema.add_field(
            field_name="description_text",
            datatype=DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"tokenizer": "jieba"},
        )
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="start_time", datatype=DataType.FLOAT)
        schema.add_field(field_name="end_time", datatype=DataType.FLOAT)
        schema.add_field(field_name="duration", datatype=DataType.FLOAT)
        schema.add_field(field_name="risk_score", datatype=DataType.FLOAT)
        schema.add_field(field_name="created_at_epoch", datatype=DataType.INT64)
        schema.add_field(field_name="trigger_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="event_type", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="tags", datatype=DataType.JSON)
        schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="embedding_dimension", datatype=DataType.INT64)
        schema.add_field(field_name="semantic_metadata", datatype=DataType.JSON)
        schema.add_field(field_name="license_name", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="source_url", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_function(
            Function(
                name="description_bm25",
                input_field_names=["description_text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )

        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="multimodal_embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200},
        )
        indexes.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
        )
        self._wait_for_required_indexes()
        self.client.load_collection(collection_name=self.collection)

    def upsert(self, row: dict[str, Any]) -> None:
        vector = row.get("multimodal_embedding")
        if not isinstance(vector, list) or len(vector) != self.dimension:
            raise ValueError(f"expected a {self.dimension}-dimensional multimodal embedding")
        if row.get("embedding_model") != self.embedding_model:
            raise ValueError("embedding model does not match the collection model")
        if (
            not isinstance(row.get("semantic_metadata"), dict)
            or not isinstance(row.get("license_name"), str)
            or not row["license_name"].strip()
            or not isinstance(row.get("source_url"), str)
            or not row["source_url"].strip()
        ):
            raise ValueError("Milvus rows require semantic_metadata, license_name and source_url")
        if not isinstance(row.get("storage_key"), str) or not row["storage_key"].strip():
            raise ValueError("Milvus rows require a storage_key for visual reranking")
        self.client.upsert(collection_name=self.collection, data=[row])

    def dense_search(
        self, vector: list[float], *, limit: int, filter_expression: str = ""
    ) -> list[MilvusHit]:
        if len(vector) != self.dimension:
            raise ValueError("query embedding dimension does not match the collection")
        response = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="multimodal_embedding",
            limit=limit,
            filter=filter_expression,
            output_fields=self.output_fields,
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 4)}},
            consistency_level="Strong",
        )
        return self._hits(response)

    def bm25_search(self, text: str, *, limit: int, filter_expression: str = "") -> list[MilvusHit]:
        if not text.strip():
            return []
        response = self.client.search(
            collection_name=self.collection,
            data=[text],
            anns_field="sparse",
            limit=limit,
            filter=filter_expression,
            output_fields=self.output_fields,
            search_params={"metric_type": "BM25", "params": {}},
            consistency_level="Strong",
        )
        return self._hits(response)

    def scalar_search(self, filters: dict[str, Any], *, limit: int) -> list[MilvusHit]:
        expression = build_filter_expression(filters)
        if not expression:
            return []
        response = self.client.query(
            collection_name=self.collection,
            filter=expression,
            output_fields=self.output_fields,
            limit=limit,
            consistency_level="Strong",
        )
        return [
            MilvusHit(segment_id=str(item["segment_id"]), score=1.0, entity=dict(item))
            for item in response
        ]

    def delete_asset(self, asset_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            filter=f"asset_id == {json.dumps(asset_id)}",
        )

    def healthcheck(self) -> None:
        self.client.describe_collection(collection_name=self.collection)

    def _validate_collection(self) -> None:
        description = self.client.describe_collection(collection_name=self.collection)
        fields = {item["name"]: item for item in description.get("fields", [])}
        dense = fields.get("multimodal_embedding")
        if dense is None:
            raise RuntimeError("Milvus collection is missing multimodal_embedding")
        params = dense.get("params", {})
        dimension = int(params.get("dim", self.dimension))
        if dimension != self.dimension:
            raise RuntimeError(
                f"Milvus collection dimension {dimension} != configured {self.dimension}"
            )
        required = set(self.output_fields) | {"multimodal_embedding", "sparse"}
        missing = required - fields.keys()
        if missing:
            raise RuntimeError(f"Milvus collection is missing required fields: {sorted(missing)}")

    def _wait_for_required_indexes(self, *, attempts: int = 120) -> None:
        required = {"multimodal_embedding", "sparse"}
        for attempt in range(attempts):
            existing = set(self.client.list_indexes(collection_name=self.collection))
            if required <= existing:
                return
            if attempt + 1 < attempts:
                time.sleep(0.25)
        raise RuntimeError(
            f"Milvus collection is missing required indexes: {sorted(required - existing)}"
        )

    @staticmethod
    def _hits(response: list[list[dict[str, Any]]]) -> list[MilvusHit]:
        if not response:
            return []
        return [
            MilvusHit(
                segment_id=str(hit.get("id") or hit.get("entity", {}).get("segment_id")),
                score=float(hit.get("distance", hit.get("score", 0.0))),
                entity=dict(hit.get("entity", {})),
            )
            for hit in response[0]
        ]


def build_filter_expression(filters: dict[str, Any]) -> str:
    clauses: list[str] = []
    allowed_strings = {"asset_id", "trigger_type", "event_type", "embedding_model"}
    for key in sorted(filters):
        value = filters[key]
        if key in allowed_strings and isinstance(value, str) and value:
            clauses.append(f"{key} == {json.dumps(value)}")
        elif key == "tags" and isinstance(value, list):
            clauses.extend(f"json_contains(tags, {json.dumps(str(tag))})" for tag in value if tag)
        elif key == "minimum_risk_score" and isinstance(value, (int, float)):
            clauses.append(f"risk_score >= {min(1.0, max(0.0, float(value)))}")
    return " and ".join(clauses)
