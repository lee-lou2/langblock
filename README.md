# langblock

> 에이전트 시스템 구축을 위한 모듈형 RAG 프레임워크

langblock은 LangChain, LangGraph와 같은 에이전트 시스템을 구축할 때 필요한 핵심 기능들을 독립적인 "블록" 단위로 제공하는 Python 패키지입니다. 각 블록은 서로 교체 가능하며, 다양한 모델 제공자를 지원합니다.

## 주요 특징

- **모듈형 설계**: 임베딩, 리랭킹, 챗 모델 등을 독립적으로 선택하고 조합
- **다양한 제공자 지원**: llama.cpp, Ollama, HuggingFace, OpenRouter
- **LanceDB 기반**: 빠르고 효율적인 벡터 검색
- **하이브리드 검색**: 벡터 검색과 BM25 전문 검색 결합
- **타입 안정성**: Generic을 활용한 타입 안전한 API

## 설치

```bash
# uv 패키지 매니저 사용 (Python 3.13+ 필요)
uv sync
```

## 핵심 블록 (Blocks)

### 1. 임베딩 모델 (Embedding Models)

텍스트를 벡터로 변환하는 블록입니다.

| 블록 | 제공자 | 사용 예시 |
|------|--------|-----------|
| `LCEmbedding` | llama.cpp | 로컬 서버에서 임베딩 생성 |
| `OLMEmbedding` | Ollama | Ollama를 통한 임베딩 생성 |

```python
from models.embedding_models import LCEmbedding, LCConfig

# llama.cpp 서버 사용
embedding = LCEmbedding(config=LCConfig())
```

### 2. 리랭킹 모델 (Reranking Models)

검색 결과의 관련성을 재평가하는 블록입니다.

| 블록 | 제공자 | 특징 |
|------|--------|------|
| `LCReranker` | llama.cpp | 서버 기반 리랭킹 |
| `HFReranker` | HuggingFace | Qwen3 Reranker 모델 지원 (0.6B/4B/8B) |

```python
from models.reranking_models import HFReranker, HFConfig, HFModel

# HuggingFace Reranker 사용
config = HFConfig(model=HFModel.QWEN3_RERANKER_0_6B)
reranker = HFReranker(config=config)
```

### 3. 챗 모델 (Chat Models)

대화형 AI를 위한 블록입니다. 모든 모델은 LangChain 호환됩니다.

| 블록 | 제공자 | 사용 예시 |
|------|--------|-----------|
| `ChatLlamaCpp` | llama.cpp | 로컬 추론 서버 |
| `ChatOllama` | Ollama | Ollama 모델 사용 |
| `ChatOpenRouter` | OpenRouter | 클라우드 API 사용 |

```python
from models.chat_models.ollama_chat import ChatOllama

# Ollama 모델 사용
chat = ChatOllama(model="qwen3:30b")
response = chat.invoke("안녕하세요!")
```

### 4. 벡터 데이터베이스 (Vector Database)

문서 저장 및 검색을 담당하는 블록입니다.

```python
from core.databases.lance import LanceDB, Record
from models.embedding_models import LCEmbedding, LCConfig

# LanceDB 초기화
embedding = LCEmbedding(LCConfig())
db = LanceDB(
    uri="./.lancedb",
    table_name="my_docs",
    embedding=embedding,
    create_fts_index=True  # BM25 검색 활성화
)

# 문서 추가
db.add([
    Record(id="1", text="Python은 프로그래밍 언어입니다.", metadata={"category": "programming"}),
    Record(id="2", text="LangChain은 LLM 프레임워크입니다.", metadata={"category": "ai"})
])

# 하이브리드 검색 (벡터 + BM25)
results = db.search(
    query="프로그래밍 언어",
    top_k=10,
    query_type=LanceDB.QueryType.HYBRID
)
```

## 검색 모드

LanceDB는 세 가지 검색 모드를 지원합니다:

- **VECTOR**: 순수 벡터 유사도 검색
- **FTS**: BM25 기반 전문 검색
- **HYBRID**: 벡터 + BM25 결합 (추천)

## 2단계 리랭킹

더 정확한 검색 결과를 위해 2단계 리랭킹을 지원합니다:

1. **1단계**: LanceDB 내장 리랭커 (RRF, LinearCombination, CrossEncoder)
2. **2단계**: 커스텀 리랭커 (BaseReranker 구현체)

```python
# 2단계 리랭킹 사용 예시
results = db.search(
    query="질문",
    top_k=10,
    query_type=LanceDB.QueryType.HYBRID,
    reranker_type=LanceDB.RerankType.RRF  # 1단계 리랭킹
)
```

## 예제: Q&A 검색 시스템

`examples/qna/` 디렉토리에 완전한 RAG 파이프라인 예제가 있습니다.

### 1. 데이터 인덱싱

```bash
python examples/qna/ingestion.py
```

72개의 Q&A 쌍을 벡터 데이터베이스에 저장합니다.

### 2. 인터랙티브 검색

```bash
python examples/qna/main.py
```

질문을 입력하면 가장 관련성 높은 답변을 검색합니다.

## 프로젝트 구조

```
langblock/
├── core/               # 핵심 컴포넌트
│   └── databases/      # 데이터베이스 구현 (LanceDB)
├── models/             # 모델 블록
│   ├── embedding_models/    # 임베딩 모델
│   ├── reranking_models/    # 리랭킹 모델
│   └── chat_models/         # 챗 모델
├── examples/           # 사용 예제
│   └── qna/           # Q&A 검색 시스템 예제
├── nodes/             # (예정) 파이프라인 노드
├── prompts/           # (예정) 프롬프트 템플릿
└── tools/             # (예정) 에이전트 도구
```

## 디자인 원칙

1. **모듈성**: 각 블록은 독립적으로 교체 가능
2. **타입 안전성**: Generic을 활용한 컴파일 타임 타입 체크
3. **생명주기 관리**: 컨텍스트 매니저를 통한 자원 관리
4. **불변성**: Frozen dataclass를 통한 설정 불변성
5. **에러 처리**: 명확한 에러 타입과 검증

## 코드 포매팅

```bash
uv run black .
```

## 라이선스

MIT

## 기여

이슈 리포트와 풀 리퀘스트를 환영합니다!