from typing import ClassVar

from langchain_openai import ChatOpenAI


class ChatOllama(ChatOpenAI):
    """
    Ollama 모델을 이용한 ChatLLM 클래스

    Args:
        model: Ollama에 설치된 모델 이름
        base_url: 온디바이스 모델 서버의 URL
    """

    BASE_URL: ClassVar[str] = "http://localhost:11434/v1/"

    def __init__(self, model: str, base_url: str = None, **kwargs):
        super().__init__(
            model=model,
            base_url=base_url or self.BASE_URL,
            api_key="ollama",
            **kwargs,
        )
