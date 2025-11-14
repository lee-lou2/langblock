from dataclasses import dataclass
from typing import List

import requests

from ._base import BaseEmbedding, Config


@dataclass(frozen=True)
class LCConfig(Config):
    """임베딩 설정"""

    model: None = None

    base_url: str = "http://127.0.0.1:8080"
    api_key: str | None = None
    timeout: float = 30.0


class LCEmbedding(BaseEmbedding[LCConfig]):
    """llama-server 기반 임베딩 모델"""

    def __init__(self, config: LCConfig):
        super().__init__(config)

    def _embed_impl(self, validated_query: str) -> List[float]:
        url = self.config.base_url.rstrip("/") + "/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {"input": validated_query}
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"llama-server embeddings request failed: {e}") from e

        data = resp.json()
        if not isinstance(data, dict) or "data" not in data or not data["data"]:
            raise RuntimeError(f"Unexpected embeddings response: {data}")

        first = data["data"][0]
        embedding = first.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Unexpected embedding field: {first}")

        return [float(x) for x in embedding]
