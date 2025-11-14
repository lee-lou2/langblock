import abc
import enum
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, List, TypeVar

ModelEnumT = TypeVar("ModelEnumT", bound=enum.Enum)


@dataclass(frozen=True)
class Config(Generic[ModelEnumT]):
    """공용 설정"""

    model: ModelEnumT


ConfigT = TypeVar("ConfigT", bound=Config)


class EmbeddingError(Exception): ...


class BaseEmbedding(Generic[ConfigT], abc.ABC):
    """베이스 임베딩"""

    QUERY_MIN_LEN = 0
    QUERY_MAX_LEN = 8192

    def __init__(self, config: ConfigT):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()

    def _validate_config(self) -> None:
        return None

    def _setup(self) -> None:
        return None

    def _cleanup(self) -> None:
        return None

    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("query is not a string")
        if not query:
            raise ValueError("query is empty")
        if len(query) > self.QUERY_MAX_LEN:
            raise ValueError("query is too long")
        if len(query) < self.QUERY_MIN_LEN:
            raise ValueError("query is too short")

    def _pre_embed(self, query: str) -> str:
        self._validate_query(query)
        return query

    def _post_embed(
        self,
        validated_query: str,
        vector: List[float],
    ) -> List[float]:
        return vector

    @abc.abstractmethod
    def _embed_impl(self, validated_query: str) -> List[float]:
        """실제 임베딩 구현

        Raises:
            EmbeddingError
        """
        raise NotImplementedError

    def embed(self, query: str) -> List[float]:
        """단일 쿼리 임베딩"""

        start = perf_counter()
        self.logger.debug(
            "Start embed: query_len=%d",
            len(query) if isinstance(query, str) else -1,
        )

        try:
            validated_query = self._pre_embed(query)
            vector = self._embed_impl(validated_query)
            final_vector = self._post_embed(validated_query, vector)

            duration_ms = (perf_counter() - start) * 1000
            self.logger.info(
                "Embedded query in %.2f ms (dim=%d)",
                duration_ms,
                len(final_vector),
            )
            return final_vector

        except EmbeddingError:
            self.logger.exception("Embedding failed with EmbeddingError")
            raise
        except Exception as e:
            self.logger.exception("Embedding failed with unexpected error")
            raise EmbeddingError(f"Embedding failed: {e}") from e
