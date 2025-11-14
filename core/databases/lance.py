import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Generic

import lancedb
from lancedb.rerankers import (
    RRFReranker,
    LinearCombinationReranker,
    CrossEncoderReranker,
)

from models.embedding_models import BaseEmbedding
from models.reranking_models import Document, BaseReranker

EmbeddingT = TypeVar("EmbeddingT", bound=BaseEmbedding)
RerankerT = TypeVar("RerankerT", bound=BaseReranker)


@dataclass
class Record:
    """레코드"""

    id: Optional[str]
    text: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """검색 결과"""

    record: Record
    distance: Optional[float] = None
    rerank_score: Optional[float] = None


class LanceDB(
    Generic[
        EmbeddingT,
        RerankerT,
    ]
):
    # Fields
    TEXT_FIELD = "text"
    VECTOR_FIELD = "vector"
    METADATA_FIELD = "metadata"

    class QueryType(enum.Enum):
        """쿼리 타입"""

        VECTOR = "vector"
        FTS = "fts"
        HYBRID = "hybrid"

    class RerankType(enum.Enum):
        """1차 리랭킹 타입"""

        RRF = "rrf"
        LinearCombination = "linear_combination"
        CrossEncoder = "cross_encoder"

    def __init__(
        self,
        uri: str,
        table_name: str,
        embedding: EmbeddingT,
        reranker: Optional[RerankerT] = None,
        create_fts_index: bool = True,
    ) -> None:
        self._uri = uri
        self._table_name = table_name
        self._embedding = embedding
        self._reranker = reranker
        self._create_fts_index = create_fts_index
        self._db = lancedb.connect(uri=self._uri)
        try:
            self._table = self._db.open_table(self._table_name)
        except ValueError:
            self._table = None

    @property
    def table(self):
        if self._table is None:
            raise ValueError("Table is not initialized yet. Call add() first.")
        return self._table

    def add(self, records: Sequence[Record]) -> None:
        if not records:
            return

        rows: List[Dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec.text, str):
                raise TypeError(f"text must be str, got {type(rec.text)}")

            vector = self._embedding.embed(rec.text)
            metadata = rec.metadata or {}

            row: Dict[str, Any] = {
                "id": rec.id,
                self.TEXT_FIELD: rec.text,
                self.METADATA_FIELD: metadata,
                self.VECTOR_FIELD: vector,
            }
            rows.append(row)

        if self._table is None:
            # 테이블 생성
            self._table = self._db.create_table(self._table_name, data=rows)
            if self._create_fts_index:
                # FTS 인덱스 생성 (BM25 검색용)
                self._table.create_fts_index(self.TEXT_FIELD)
        else:
            # 기존 테이블에 append
            self._table.add(rows)

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[str] = None,
        query_type: QueryType = QueryType.HYBRID,
        prefilter: bool = True,
        reranker_type: Optional[RerankType] = None,
    ) -> List[SearchResult]:
        if self._table is None:
            raise ValueError("Table is not initialized yet. Call add() first.")
        if not query:
            raise ValueError("query is empty")

        # LanceDB QueryBuilder 구성
        if query_type == self.QueryType.VECTOR:
            # 순수 벡터 검색
            query_vector = self._embedding.embed(query)
            builder = self._table.search(
                query_vector,
                query_type=self.QueryType.VECTOR.value,
                vector_column_name=self.VECTOR_FIELD,
            )
        elif query_type == self.QueryType.FTS:
            # 풀텍스트 검색 (BM25)
            builder = self._table.search(
                query,
                query_type=self.QueryType.FTS.value,
                fts_columns=self.TEXT_FIELD,
            )
        elif query_type == self.QueryType.HYBRID:
            # 명시적 hybrid 패턴
            query_vector = self._embedding.embed(query)
            builder = (
                self._table.search(
                    query_type=self.QueryType.HYBRID.value,
                    vector_column_name=self.VECTOR_FIELD,
                    fts_columns=self.TEXT_FIELD,
                )
                .vector(query_vector)
                .text(query)
            )
        else:
            raise ValueError("query_type must be one of 'vector', 'fts', 'hybrid'")

        # 메타 데이터 조건
        if where is not None:
            builder = builder.where(where, prefilter=prefilter)

        # 1차 리랭킹
        if reranker_type:
            if reranker_type == self.RerankType.CrossEncoder:
                rr = CrossEncoderReranker()
            elif reranker_type == self.RerankType.LinearCombination:
                rr = LinearCombinationReranker(weight=0.7)
            else:
                rr = RRFReranker()
            builder = builder.rerank(reranker=rr)

        rows: List[Dict[str, Any]] = builder.limit(top_k).to_list()

        # 결과 조회
        results: List[SearchResult] = []
        for row in rows:
            record = Record(
                id=row.get("id"),
                text=row.get(self.TEXT_FIELD, ""),
                metadata=row.get(self.METADATA_FIELD, {}) or {},
            )
            distance = row.get("_distance")
            results.append(
                SearchResult(
                    record=record,
                    distance=distance,
                    rerank_score=None,
                )
            )

        # 2차 리랭커 없으면 바로 반환
        if not self._reranker or not results:
            return results

        # 사용자 정의 리랭커 적용 (문서 text 기준)
        docs = [Document(id=r.record.id, text=r.record.text) for r in results]
        scored = self._reranker.rerank(query=query, docs=docs)

        # 리랭커 결과(id -> score) 매핑
        score_map: Dict[str, float] = {s.id: float(s.score) for s in scored}

        # SearchResult rerank_score 주입
        for r in results:
            r.rerank_score = score_map.get(r.record.id)

        # rerank_score 기준 정렬
        results.sort(
            key=lambda r: (r.rerank_score is not None, r.rerank_score or 0.0),
            reverse=True,
        )
        return results[:top_k]
