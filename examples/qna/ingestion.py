"""
데이터셋을 LanceDB에 저장하는 스크립트

python examples/qna/ingestion.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.databases.lance import Record, LanceDB
from models.embedding_models import LCConfig, LCEmbedding


def load_dataset(file_path: str) -> list[dict]:
    """JSON 데이터셋 파일을 로드합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_records(dataset: list[dict]) -> list[Record]:
    """데이터셋을 LanceDB Record 객체로 변환합니다."""
    records = []
    for item in dataset:
        # question과 answer를 결합하여 검색 텍스트로 사용
        text = f"질문: {item['question']}\n답변: {item['answer']}"

        record = Record(
            id=f"qna-{item['pk']}",
            text=text,
            metadata={
                "question": item["question"],
                "answer": item["answer"],
                "category": item["category"],
                "pk": item["pk"],
            },
        )
        records.append(record)

    return records


def main():
    # 데이터셋 경로 설정
    dataset_path = Path(__file__).parent / "dataset.json"

    # 데이터셋 로드
    print("📂 데이터셋 로딩 중...")
    dataset = load_dataset(str(dataset_path))
    print(f"✅ {len(dataset)}개의 Q&A 데이터 로드 완료")

    # 임베딩 모델 초기화
    print("\n🤖 임베딩 모델 초기화 중...")
    config = LCConfig()
    embedding = LCEmbedding(config)

    # LanceDB 초기화
    print("💾 LanceDB 초기화 중...")
    db = LanceDB(
        uri="./.lancedb",
        table_name="qna_dataset",
        embedding=embedding,
        create_fts_index=True,
    )

    # Record 객체 생성
    print("\n🔄 데이터 변환 중...")
    records = create_records(dataset)

    # 데이터 삽입
    print("📥 LanceDB에 데이터 삽입 중...")
    db.add(records)

    print(f"\n✨ 완료! {len(records)}개의 Q&A가 LanceDB에 저장되었습니다.")
    print(f"   테이블 이름: qna_dataset")
    print(f"   저장 경로: ./.lancedb")


if __name__ == "__main__":
    main()
