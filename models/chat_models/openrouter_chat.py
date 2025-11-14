import os
from typing import ClassVar

from langchain_openai import ChatOpenAI


class ChatOpenRouter(ChatOpenAI):
    """
    OpenRouter 모델을 이용한 ChatLLM 클래스

    Args:
        model: OpenRouter에서 제공하는 모델 이름
        api_key: OpenRouter API 키 (환경 변수 OPENROUTER_API_KEY에서 가져올 수 있음)
    """

    BASE_URL: ClassVar[str] = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, api_key: str = None, **kwargs):
        api_key = self._get_api_key(api_key)
        super(ChatOpenRouter, self).__init__(
            model=model,
            base_url=self.BASE_URL,
            api_key=api_key,
            **kwargs,
        )

    def _get_api_key(self, api_key: str = None):
        """
        OpenRouter API키 조회

        Args:
            api_key: 파라미터로 전달된 API KEY(Option)

        Returns:
            OpenRouter API Key
        """
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise RuntimeError("OpenRouter API key not set")
        return api_key
