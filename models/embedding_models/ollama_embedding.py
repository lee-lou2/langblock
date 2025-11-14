import enum
from dataclasses import dataclass
from typing import List

import ollama

from ._base import BaseEmbedding, Config


class Model(enum.Enum):
    QWEN3_EMBEDDING_0_6B = "qwen3-embedding:0.6b"
    QWEN3_EMBEDDING_4B = "qwen3-embedding:4b"
    QWEN3_EMBEDDING_8B = "qwen3-embedding:8b"


@dataclass(frozen=True)
class OLMConfig(Config[Model]):
    """임베딩 설정"""

    model: Model = Model.QWEN3_EMBEDDING_0_6B


class OLMEmbedding(BaseEmbedding[OLMConfig]):
    """올라마 임베딩 모델"""

    def __init__(self, config: OLMConfig):
        super().__init__(config)
        self.model_name = str(config.model.value)

    def _embed_impl(self, validated_query: str) -> List[float]:
        resp = ollama.embed(model=self.model_name, input=validated_query)
        return resp["embeddings"][0]
