"""임베딩 서비스 - BGE-m3-ko + Milvus"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from embedding_model import EmbeddingModel
from milvus_manager import MilvusManager
from schemas import (
    CollectionCreateRequest,
    CollectionResponse,
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
    InsertRequest,
    InsertResponse,
    SearchByVectorRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 모델 로드 및 Milvus 연결을 관리한다."""
    load_dotenv()

    # 시작: 모델 로드 + Milvus 연결
    logger.info("임베딩 서비스 시작 중...")
    app.state.model = EmbeddingModel.get_instance()
    app.state.milvus = await MilvusManager.get_instance()
    logger.info("임베딩 서비스 준비 완료")

    yield

    # 종료
    logger.info("임베딩 서비스 종료")


app = FastAPI(title="Embedding Service", version="1.0", lifespan=lifespan)


# ── 헬스체크 ──

@app.get("/health")
async def health() -> Dict[str, Any]:
    """모델 로드 상태 및 Milvus 연결 상태를 확인한다."""
    model_ok = app.state.model.is_loaded
    milvus_ok = app.state.milvus.client is not None

    milvus_collections = []
    if milvus_ok:
        try:
            milvus_collections = await app.state.milvus.list_collections()
        except Exception:
            milvus_ok = False

    return {
        "status": "ok" if (model_ok and milvus_ok) else "degraded",
        "model": {
            "loaded": model_ok,
            "name": EmbeddingModel.MODEL_NAME,
            "dimension": EmbeddingModel.DIMENSION,
        },
        "milvus": {
            "connected": milvus_ok,
            "collections": milvus_collections,
        },
    }


# ── 임베딩 ──

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """단일 텍스트를 임베딩 벡터로 변환한다."""
    vector = app.state.model.encode(req.text)
    return EmbedResponse(embedding=vector, dimension=EmbeddingModel.DIMENSION)


@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(req: EmbedBatchRequest) -> EmbedBatchResponse:
    """여러 텍스트를 배치로 임베딩한다."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts 배열이 비어있습니다.")
    vectors = app.state.model.encode_batch(req.texts)
    return EmbedBatchResponse(
        embeddings=vectors,
        dimension=EmbeddingModel.DIMENSION,
        count=len(vectors),
    )


# ── 컬렉션 관리 ──

@app.post("/collection/create", response_model=CollectionResponse)
async def create_collection(req: CollectionCreateRequest) -> CollectionResponse:
    """Milvus 컬렉션을 생성한다."""
    result = await app.state.milvus.create_collection(
        name=req.name,
        dim=req.dimension,
        description=req.description,
    )
    return CollectionResponse(**result)


@app.get("/collection/list")
async def list_collections() -> Dict[str, List[str]]:
    """모든 컬렉션 목록을 반환한다."""
    collections = await app.state.milvus.list_collections()
    return {"collections": collections}


@app.delete("/collection/{name}", response_model=CollectionResponse)
async def drop_collection(name: str) -> CollectionResponse:
    """컬렉션을 삭제한다."""
    result = await app.state.milvus.drop_collection(name)
    return CollectionResponse(**result)


# ── 데이터 삽입 ──

@app.post("/collection/{name}/insert", response_model=InsertResponse)
async def insert(name: str, req: InsertRequest) -> InsertResponse:
    """벡터 + 메타데이터를 컬렉션에 삽입한다.

    auto_embed=True이면 text를 자동으로 임베딩하여 삽입한다.
    auto_embed=False이면 각 item에 embedding 필드가 필수다.
    """
    data = []
    for item in req.items:
        if req.auto_embed:
            embedding = app.state.model.encode(item.text)
        else:
            if item.embedding is None:
                raise HTTPException(
                    status_code=400,
                    detail="auto_embed=False일 때 각 item에 embedding이 필요합니다.",
                )
            embedding = item.embedding

        data.append({
            "text": item.text,
            "category": item.category,
            "metadata": item.metadata,
            "embedding": embedding,
        })

    result = await app.state.milvus.insert(collection_name=name, data=data)
    return InsertResponse(insert_count=result["insert_count"])


# ── 검색 ──

@app.post("/collection/{name}/search", response_model=SearchResponse)
async def search(name: str, req: SearchRequest) -> SearchResponse:
    """텍스트로 유사도 검색한다. 자동으로 임베딩 후 KNN 검색."""
    query_vector = app.state.model.encode(req.query)

    try:
        results = await app.state.milvus.search(
            collection_name=name,
            query_vectors=[query_vector],
            limit=req.limit,
            output_fields=req.output_fields,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    hits = []
    if results:
        for hit in results[0]:
            entity = hit["entity"]
            hits.append(SearchHit(
                id=hit["id"],
                distance=hit["distance"],
                text=entity.get("text", ""),
                category=entity.get("category", ""),
                metadata=entity.get("metadata", {}),
            ))

    return SearchResponse(results=hits, count=len(hits))


@app.post("/collection/{name}/search/vector", response_model=SearchResponse)
async def search_by_vector(name: str, req: SearchByVectorRequest) -> SearchResponse:
    """벡터로 직접 유사도 검색한다."""
    try:
        results = await app.state.milvus.search(
            collection_name=name,
            query_vectors=[req.query_vector],
            limit=req.limit,
            output_fields=req.output_fields,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    hits = []
    if results:
        for hit in results[0]:
            entity = hit["entity"]
            hits.append(SearchHit(
                id=hit["id"],
                distance=hit["distance"],
                text=entity.get("text", ""),
                category=entity.get("category", ""),
                metadata=entity.get("metadata", {}),
            ))

    return SearchResponse(results=hits, count=len(hits))
