"""
Q&A 검색 예시

python examples/qna/main.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.databases.lance import Record, LanceDB
from models.embedding_models import LCConfig, LCEmbedding
from models.reranking_models import (
    LCReranker,
    LCConfig as LCRerankerConfig,
    HFReranker,
    HFModel,
    HFConfig
)


def print_results(title: str, results, max_display: int = 3):
    """검색 결과를 출력합니다."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    if not results:
        print("검색 결과가 없습니다.")
        return

    for i, r in enumerate(results[:max_display], start=1):
        rec: Record = r.record
        print(f"\n[{i}] {rec.metadata.get('category', 'N/A')} (ID: {rec.id})")
        print(f"📌 질문: {rec.metadata.get('question', 'N/A')}")
        print(f"💬 답변: {rec.metadata.get('answer', 'N/A')}")

        # 거리와 재랭킹 점수 출력 (None 체크)
        score_parts = []
        if r.distance is not None:
            score_parts.append(f"거리: {r.distance:.4f}")
        if r.rerank_score is not None:
            score_parts.append(f"리랭킹 스코어: {float(r.rerank_score)}")

        if score_parts:
            print(f"   {' | '.join(score_parts)}")

        print("-" * 80)


def main():
    """대화형 검색 모드"""
    # 임베딩 모델 초기화
    embedding_config = LCConfig()
    embedding = LCEmbedding(embedding_config)
    reranker_config = LCRerankerConfig(base_url="http://127.0.0.1:8081")
    reranker = LCReranker(reranker_config)
    # reranker_config = HFConfig(model=HFModel.QWEN3_RERANKER_8B)
    # reranker = HFReranker(reranker_config)

    # LanceDB 연결
    db = LanceDB(
        uri="./.lancedb",
        table_name="qna_dataset",
        embedding=embedding,
        reranker=reranker,
    )

    print("\n" + "=" * 80)
    print("  Q&A 검색 시스템 (종료하려면 'q' 또는 'quit' 입력)")
    print("=" * 80)

    while True:
        query = input("\n🔍 검색어를 입력하세요: ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            print("\n👋 검색을 종료합니다.")
            break

        if not query:
            print("⚠️  검색어를 입력해주세요.")
            continue

        # 데이터 검색
        results = db.search(
            query=query,
            # 조회 데이터 수
            top_k=3,
            # 검색 방식 선택(vector, fts, hybrid 선택 가능)
            query_type=LanceDB.QueryType.HYBRID,
            # # 1차 리랭커 설정(RRF: 기본, LinearCombination, CrossEncoder 선택 가능)
            reranker_type=LanceDB.RerankType.CrossEncoder,
        )

        print_results("검색 결과", results, max_display=5)


if __name__ == "__main__":
    main()
