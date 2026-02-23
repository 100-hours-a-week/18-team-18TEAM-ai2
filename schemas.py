"""요청/응답 Pydantic 모델"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── 임베딩 ──

class EmbedRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int


class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int


# ── 컬렉션 관리 ──

class CollectionCreateRequest(BaseModel):
    name: str
    dimension: int = 1024
    description: str = ""


class CollectionResponse(BaseModel):
    collection: str
    status: str


# ── 삽입 ──

class InsertItem(BaseModel):
    text: str
    category: str = ""
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None


class InsertRequest(BaseModel):
    items: List[InsertItem]
    auto_embed: bool = True


class InsertResponse(BaseModel):
    insert_count: int


# ── 검색 ──

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    output_fields: Optional[List[str]] = None


class SearchByVectorRequest(BaseModel):
    query_vector: List[float]
    limit: int = 5
    output_fields: Optional[List[str]] = None


class SearchHit(BaseModel):
    id: int
    distance: float
    text: str
    category: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchHit]
    count: int
