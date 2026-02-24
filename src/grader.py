"""Grader - 검색 결과 관련성 평가 및 쿼리 재작성

Phase 3의 핵심 컴포넌트로, 검색 결과를 이진(PASS/FAIL) 판단하고,
FAIL 시 쿼리를 재작성한다.
"""

from src.llm_adapter import OllamaAdapter
from src.prompts.grader import GRADER_PROMPT
from src.prompts.rewriter import REWRITER_PROMPT


class Grader:
    def __init__(self, llm: OllamaAdapter):
        self.llm = llm

    def evaluate(self, query: str, documents: list[dict]) -> str:
        """검색 결과의 관련성을 PASS/FAIL로 평가한다."""
        if not documents:
            return "FAIL"

        docs_text = "\n\n---\n\n".join(
            f"[문서 {i + 1}]\n{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        )

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": GRADER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"## 사용자 질문\n{query}\n\n"
                        f"## 검색된 문서\n{docs_text}"
                    ),
                },
            ]
        )

        result = response.content.strip().upper()
        # 여러 단어가 반환된 경우 PASS/FAIL 추출
        for word in result.split():
            if word in ("PASS", "FAIL"):
                return word
        return "PASS"  # 확신 없으면 안전 모드


class QueryRewriter:
    def __init__(self, llm: OllamaAdapter):
        self.llm = llm

    def rewrite(self, original_query: str) -> str:
        """원본 질문을 개선된 검색 쿼리로 재작성한다."""
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": REWRITER_PROMPT},
                {"role": "user", "content": f"원본 질문: {original_query}"},
            ]
        )

        rewritten = response.content.strip()
        # 빈 결과 방어
        return rewritten if rewritten else original_query
