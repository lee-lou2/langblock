# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

langblock is a Python RAG (Retrieval-Augmented Generation) framework that provides modular components for embedding, reranking, vector search, and chat models. The project uses LanceDB as the vector database and supports multiple model providers (Ollama, llama.cpp, OpenRouter, HuggingFace).

## Development Setup

This project uses `uv` as the package manager. Python 3.13+ is required.

### Installation
```bash
uv sync
```

### Running Examples

The Q&A example demonstrates the full RAG pipeline:

1. **Data ingestion** (create embeddings and index):
```bash
python examples/qna/ingestion.py
```

2. **Interactive search**:
```bash
python examples/qna/main.py
```

### Code Formatting
```bash
uv run black .
```

## Architecture

### Core Components

The codebase is organized into modular, interchangeable components:

#### 1. Embedding Models (`models/embedding_models/`)
- **Base class**: `BaseEmbedding` - Abstract base with lifecycle hooks (`_setup`, `_cleanup`, `_pre_embed`, `_post_embed`)
- Implementations:
  - `LCEmbedding`: llama.cpp server (via HTTP API at `/v1/embeddings`)
  - `OLMEmbedding`: Ollama models (via `ollama.embed()`)
- All models use a `Config` dataclass pattern with frozen configurations
- Embedding lifecycle includes validation, timing, and error handling

#### 2. Reranking Models (`models/reranking_models/`)
- **Base class**: `BaseReranker` - Abstract base with lifecycle hooks
- **Types**: Document (input), RerankOutput (scored output)
- Implementations:
  - `LCReranker`: llama.cpp server reranking (via HTTP API at `/v1/rerank`)
  - `HFReranker`: HuggingFace cross-encoder models
- Rerankers take query + documents, return scored and ranked results

#### 3. Chat Models (`models/chat_models/`)
- All chat models inherit from LangChain's `ChatOpenAI` for OpenAI-compatible API endpoints
- Implementations:
  - `ChatLlamaCpp`: llama.cpp server (default: `http://localhost:8080/v1`)
  - `ChatOllama`: Ollama (default: `http://localhost:11434/v1/`)
  - `ChatOpenRouter`: OpenRouter API (requires `OPENROUTER_API_KEY` env var)

#### 4. Vector Database (`core/databases/lance.py`)
- **LanceDB**: Generic wrapper with type parameters for embedding and reranking models
- **Search modes**:
  - `QueryType.VECTOR`: Pure vector similarity
  - `QueryType.FTS`: BM25 full-text search
  - `QueryType.HYBRID`: Combined vector + BM25
- **Two-stage reranking**:
  1. First stage: Built-in LanceDB rerankers (RRF, LinearCombination, CrossEncoder)
  2. Second stage: Custom reranker (via `BaseReranker` implementation)
- Supports metadata filtering with SQL-like `where` clauses

### Key Design Patterns

1. **Generic typing**: Components use `TypeVar` to maintain type safety across the stack
   - `LanceDB[EmbeddingT, RerankerT]` ensures compatible model types
   - Config classes are generic over model enums

2. **Lifecycle management**: All model classes support context managers (`__enter__`/`__exit__`) for setup/cleanup

3. **Layered configuration**:
   - Base URL defaults in class variables
   - Override via config dataclasses
   - Environment variables for API keys

4. **Dual reranking architecture**: LanceDB supports both built-in rerankers (for hybrid search fusion) and custom rerankers (for semantic relevance)

### Data Flow

```
Query → LanceDB.search()
  ├─> Embedding model (query vector)
  ├─> Vector search / FTS / Hybrid
  ├─> [Optional] Built-in reranker (RRF/LinearCombination/CrossEncoder)
  ├─> [Optional] Custom reranker (BaseReranker implementation)
  └─> Ranked SearchResults
```

## Common Patterns

### Adding a New Embedding Model

1. Create new file in `models/embedding_models/`
2. Define model enum and config dataclass
3. Inherit from `BaseEmbedding[YourConfig]`
4. Implement `_embed_impl(validated_query) -> List[float]`
5. Export in `models/embedding_models/__init__.py`

### Adding a New Reranker

1. Create new file in `models/reranking_models/`
2. Define config dataclass inheriting from `Config`
3. Inherit from `BaseReranker[YourConfig]`
4. Implement `_rerank_impl(query, docs, top_k) -> list[RerankOutput]`
5. Export in `models/reranking_models/__init__.py`

### Using LanceDB

```python
from core.databases.lance import LanceDB, Record
from models.embedding_models import LCEmbedding, LCConfig

embedding = LCEmbedding(LCConfig())
db = LanceDB(
    uri="./.lancedb",
    table_name="my_table",
    embedding=embedding,
    create_fts_index=True  # Enable BM25 search
)

# Add documents
db.add([Record(id="1", text="...", metadata={})])

# Search with hybrid mode + custom reranking
results = db.search(
    query="...",
    top_k=10,
    query_type=LanceDB.QueryType.HYBRID,
    reranker_type=LanceDB.RerankType.RRF  # Built-in reranker
)
```

## Important Notes

- The project has empty placeholder directories: `nodes/`, `prompts/`, `tools/` (future expansion)
- LanceDB tables are lazily initialized on first `add()` call
- All model configs use frozen dataclasses for immutability
- llama.cpp embedding and reranking can run on different ports (see `examples/qna/main.py:55`)