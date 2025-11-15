import enum
from dataclasses import dataclass
from typing import Sequence, Literal, Annotated

import torch
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._base import (
    BaseReranker,
    Config,
    Document,
    RerankOutput,
)


class Model(enum.Enum):
    QWEN3_RERANKER_0_6B = "Qwen/Qwen3-Reranker-0.6B"
    QWEN3_RERANKER_4B = "Qwen/Qwen3-Reranker-4B"
    QWEN3_RERANKER_8B = "Qwen/Qwen3-Reranker-8B"


@dataclass(frozen=True)
class HFConfig(Config[Model]):
    """리랭커 설정"""

    model: Model = Field(default=Model.QWEN3_RERANKER_0_6B)

    device: Literal["cuda", "cpu", "mps"] | None = None
    max_length: Annotated[int, Field(gt=0)] = 8192
    batch_size: Annotated[int, Field(gt=0)] = 16
    instruct: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )


class HFReranker(BaseReranker[HFConfig]):
    """허깅페이스 기반 리랭킹 모델"""

    PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and "
        'the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, config: HFConfig):
        super().__init__(config)
        self.model_name = str(self.config.model.value)
        self.device: str = self.config.device or (  # type: ignore
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._tokenizer = None
        self._model = None
        self._tid_no: int | None = None
        self._tid_yes: int | None = None
        self._prefix_ids: list[int] | None = None
        self._suffix_ids: list[int] | None = None

    def _setup(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        self.logger.info("Loading HF model %s on %s", self.model_name, self.device)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        model = (
            AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()
        )

        # 캐시할 토큰/프롬프트
        tid_no = tokenizer.convert_tokens_to_ids("no")
        tid_yes = tokenizer.convert_tokens_to_ids("yes")
        prefix_ids = tokenizer.encode(self.PREFIX, add_special_tokens=False)
        suffix_ids = tokenizer.encode(self.SUFFIX, add_special_tokens=False)

        self._tokenizer = tokenizer
        self._model = model
        self._tid_no = tid_no
        self._tid_yes = tid_yes
        self._prefix_ids = prefix_ids
        self._suffix_ids = suffix_ids

        self.logger.info("Model %s loaded", self.model_name)

    def _cleanup(self) -> None:
        if self._model is not None:
            self.logger.info("Cleaning up model %s", self.model_name)
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _rerank_impl(
        self,
        query: str,
        docs: Sequence[Document],
    ) -> list[RerankOutput]:
        self._setup()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._tid_no is not None
        assert self._tid_yes is not None
        assert self._prefix_ids is not None
        assert self._suffix_ids is not None

        batch_size = self.config.batch_size
        max_len = self.config.max_length
        instruct = self.config.instruct

        raw_results: list[RerankOutput] = []

        for i in range(0, len(docs), batch_size):
            chunk = docs[i : i + batch_size]

            pairs: list[str] = [
                f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {d.text}"
                for d in chunk
            ]

            enc = self._tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_len - len(self._prefix_ids) - len(self._suffix_ids),
            )

            for j, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][j] = self._prefix_ids + ids + self._suffix_ids

            enc = self._tokenizer.pad(
                enc,
                padding=True,
                return_tensors="pt",
                max_length=max_len,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :]
                scores = torch.stack(
                    [logits[:, self._tid_no], logits[:, self._tid_yes]],
                    dim=1,
                )
                probs_yes = torch.softmax(scores, dim=1)[:, 1].tolist()

            for doc, score in zip(chunk, probs_yes):
                raw_results.append(
                    RerankOutput(
                        id=doc.id,
                        text=doc.text,
                        score=float(score),
                    )
                )
        return raw_results
