from dataclasses import dataclass
from typing import Any, Sequence

import requests

from ._base import (
    BaseReranker,
    Config,
    Document,
    RerankOutput,
    RerankError,
)


@dataclass(frozen=True)
class LCConfig(Config):
    """리랭커 설정"""

    model: None = None

    base_url: str = "http://127.0.0.1:8080"
    api_key: str | None = None
    timeout: float = 30.0


class LCReranker(BaseReranker[LCConfig]):
    """llama.cpp 기반 리랭킹 모델"""

    def _validate_config(self) -> None:
        if not self.config.base_url:
            raise ValueError("base_url must not be empty")

    def _setup(self) -> None:
        return None

    def _cleanup(self) -> None:
        return None

    def _rerank_impl(
        self,
        query: str,
        docs: Sequence[Document],
        top_k: int | None,
    ) -> list[RerankOutput]:
        if not docs:
            return []

        url = self.config.base_url.rstrip("/") + "/v1/rerank"
        payload: dict[str, Any] = {
            "query": query,
            "documents": [d.text for d in docs],
            "top_n": top_k,
        }

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RerankError(f"llama-server rerank request failed: {e}") from e

        data = resp.json()
        if not isinstance(data, dict) or "results" not in data or not data["results"]:
            raise RerankError(f"llama-server rerank request failed: {data}")

        items = data["results"]
        raw_results: list[RerankOutput] = []
        for item in items:
            try:
                idx = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue

            if not (0 <= idx < len(docs)):
                continue

            score = item.get("relevance_score")
            if score is None:
                continue

            doc = docs[idx]
            raw_results.append(
                RerankOutput(
                    id=doc.id,
                    text=doc.text,
                    score=float(score),
                    rank=0,
                )
            )

        raw_results.sort(key=lambda r: r.score, reverse=True)
        ranked: list[RerankOutput] = [
            r.model_copy(update={"rank": i}) for i, r in enumerate(raw_results, start=1)
        ]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
