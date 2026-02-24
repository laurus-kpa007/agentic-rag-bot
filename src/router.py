"""Router - 사용자 질문 의도 분류기

경량 LLM 호출로 질문을 3가지 카테고리로 분류한다.
"""

from src.llm_adapter import OllamaAdapter
from src.prompts.router import ROUTER_PROMPT


class Router:
    VALID_ROUTES = {"INTERNAL_SEARCH", "WEB_SEARCH", "CHITCHAT"}

    def __init__(self, llm: OllamaAdapter):
        self.llm = llm

    def classify(self, query: str) -> str:
        """사용자 질문을 분류하여 라우팅 경로를 반환한다."""
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query},
            ]
        )

        route = response.content.strip().upper()

        # 여러 단어가 반환된 경우 첫 번째 유효 라우트 추출
        for word in route.split():
            if word in self.VALID_ROUTES:
                return word

        return "INTERNAL_SEARCH"  # 폴백
