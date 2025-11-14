from typing import ClassVar

from langchain_openai import ChatOpenAI


class ChatLlamaCpp(ChatOpenAI):
    """
    llama.cpp 모델을 이용한 ChatLLM 클래스

    Args:
        model: "local-model"로 고정(모델은 별도로 구분하지 않음)
        api_key: "not-needed"로 고정(API 키는 필요하지 않음)
        base_url: 온디바이스 모델 서버의 URL
    """

    BASE_URL: ClassVar[str] = "http://localhost:8080/v1"

    def __init__(self, base_url: str = None, **kwargs):
        super().__init__(
            model="local-model",
            base_url=base_url or self.BASE_URL,
            api_key="not-needed",
            **kwargs,
        )
