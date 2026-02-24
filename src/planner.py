"""Query Planner - 질의 분석 및 최적화

Router 이후, 검색 이전에 위치하여 사용자 질문을 벡터 검색에
최적화된 쿼리로 변환한다.
"""

import json
from dataclasses import dataclass, field

from src.llm_adapter import OllamaAdapter
from src.prompts.planner import PLANNER_PROMPT


@dataclass
class QueryPlan:
    intent: str = ""
    keywords: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    strategy: str = "SINGLE"

    def is_multi(self) -> bool:
        return self.strategy == "MULTI" and len(self.search_queries) > 1


class QueryPlanner:
    def __init__(self, llm: OllamaAdapter):
        self.llm = llm

    def plan(
        self,
        query: str,
        route: str,
        conversation_history: list | None = None,
    ) -> QueryPlan:
        """사용자 질문을 분석하여 최적화된 검색 계획을 반환한다."""
        history_context = ""
        if conversation_history:
            recent = conversation_history[-6:]  # 최근 3턴
            history_context = "\n".join(
                f"{'사용자' if m['role'] == 'user' else '어시스턴트'}: {m['content'][:200]}"
                for m in recent
            )

        user_message = f"## 라우팅 결과\n{route}\n\n"
        if history_context:
            user_message += f"## 대화 히스토리\n{history_context}\n\n"
        user_message += f"## 사용자 질문\n{query}"

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        return self._parse_plan(response.content, query)

    def _parse_plan(self, response_text: str, original_query: str) -> QueryPlan:
        """LLM 응답에서 QueryPlan을 파싱한다."""
        try:
            # JSON 블록 추출
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            plan = QueryPlan(
                intent=data.get("intent", ""),
                keywords=data.get("keywords", []),
                search_queries=data.get("search_queries", [original_query]),
                strategy=data.get("strategy", "SINGLE"),
            )

            # 빈 search_queries 방어
            if not plan.search_queries:
                plan.search_queries = [original_query]

            return plan

        except (json.JSONDecodeError, KeyError, IndexError):
            # 파싱 실패 시 원본 쿼리를 그대로 사용
            return QueryPlan(
                intent=original_query,
                keywords=[],
                search_queries=[original_query],
                strategy="SINGLE",
            )
