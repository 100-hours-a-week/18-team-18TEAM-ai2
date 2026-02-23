# Embedding Service

**dragonkue/BGE-m3-ko** 모델로 텍스트를 1024차원 벡터로 임베딩하고, **Milvus** 벡터 DB에 저장·검색하는 FastAPI 서비스입니다.

## 기술 스택

- **임베딩 모델**: [dragonkue/BGE-m3-ko](https://huggingface.co/dragonkue/BGE-m3-ko) (1024차원, COSINE)
- **벡터 DB**: Milvus
- **API 프레임워크**: FastAPI + Uvicorn
- **Python**: 3.13

## 시작하기

### 환경변수 설정

`.env` 파일을 프로젝트 루트에 생성합니다.

```env
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
```

### 로컬 실행

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker 실행

```bash
docker build -t embedding-service .

docker run -p 8000:8000 \
  -e MILVUS_URI=http://host.docker.internal:19530 \
  -e MILVUS_TOKEN=root:Milvus \
  embedding-service
```

## API

API 문서는 서버 실행 후 `http://localhost:8000/docs` 에서 확인할 수 있습니다.

### 헬스체크

```
GET /health
```

### 임베딩

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /embed` | 단일 텍스트 임베딩 |
| `POST /embed/batch` | 여러 텍스트 배치 임베딩 |

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요"}'
```

### 컬렉션 관리

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /collection/create` | 컬렉션 생성 |
| `GET /collection/list` | 컬렉션 목록 조회 |
| `DELETE /collection/{name}` | 컬렉션 삭제 |

```bash
curl -X POST http://localhost:8000/collection/create \
  -H "Content-Type: application/json" \
  -d '{"name": "my_collection", "dimension": 1024}'
```

### 데이터 삽입

```
POST /collection/{name}/insert
```

```json
{
  "items": [
    {
      "text": "저장할 텍스트",
      "category": "카테고리",
      "metadata": {"source": "example"}
    }
  ],
  "auto_embed": true
}
```

`auto_embed: false`로 설정하면 각 item에 `embedding` 필드를 직접 전달해야 합니다.

### 검색

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /collection/{name}/search` | 텍스트로 유사도 검색 |
| `POST /collection/{name}/search/vector` | 벡터로 직접 검색 |

```bash
curl -X POST http://localhost:8000/collection/my_collection/search \
  -H "Content-Type: application/json" \
  -d '{"query": "검색할 텍스트", "limit": 5}'
```

## 프로젝트 구조

```
embedding-service/
├── main.py              # FastAPI 앱 및 라우터
├── embedding_model.py   # BGE-m3-ko 모델 싱글톤
├── milvus_manager.py    # Milvus 연결 및 컬렉션 관리
├── schemas.py           # Pydantic 요청/응답 모델
├── requirements.txt
├── Dockerfile
└── .env
```
