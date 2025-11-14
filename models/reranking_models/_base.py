import abc
import enum
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Generic, TypeVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelEnumT = TypeVar("ModelEnumT", bound=enum.Enum)
ConfigT = TypeVar("ConfigT", bound="Config")


class Document(BaseModel):
    """리랭킹 문서"""

    model_config = ConfigDict(frozen=True, strict=True)

    id: str | None = None
    text: Annotated[str, Field(min_length=1)]


class RerankOutput(BaseModel):
    """리랭킹 출력"""

    model_config = ConfigDict(frozen=False, strict=True)

    id: str | None
    text: str
    score: float
    rank: int = 0

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rank must be non-negative")
        return v


@dataclass(frozen=True)
class Config(Generic[ModelEnumT]):
    """리랭커 설정"""

    model: ModelEnumT


class RerankError(Exception): ...


class BaseReranker(Generic[ConfigT], abc.ABC):
    """베이스 리랭커"""

    def __init__(self, config: ConfigT):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()

    def __enter__(self) -> Self:
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

    def _pre_rerank(
        self,
        query: str,
        docs: Sequence[Document],
        top_k: int | None,
    ) -> tuple[str, Sequence[Document], int | None]:
        return query, docs, top_k

    def _post_rerank(
        self,
        query: str,
        docs: Sequence[Document],
        top_k: int | None,
        results: list[RerankOutput],
    ) -> list[RerankOutput]:
        return results

    @abc.abstractmethod
    def _rerank_impl(
        self,
        query: str,
        docs: Sequence[Document],
        top_k: int | None,
    ) -> list[RerankOutput]:
        """실제 리랭킹 구현

        Raises:
            RerankError
        """
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        docs: Sequence[Document],
        top_k: int | None = None,
    ) -> list[RerankOutput]:
        start = perf_counter()
        self.logger.debug(
            "Start rerank: query=%r, top_k=%r, num_docs=%d",
            query,
            top_k,
            len(docs),
        )

        try:
            q, ds, tk = self._pre_rerank(query, docs, top_k)
            results = self._rerank_impl(q, ds, tk)
            final_results = self._post_rerank(q, ds, tk, results)

            duration_ms = (perf_counter() - start) * 1000
            self.logger.info(
                "Reranked %d docs in %.2f ms",
                len(ds),
                duration_ms,
            )
            return final_results

        except RerankError:
            self.logger.exception("Reranking failed with RerankError")
            raise
        except Exception as e:
            self.logger.exception("Reranking failed with unexpected error")
            raise RerankError(f"Reranking failed: {e}") from e
