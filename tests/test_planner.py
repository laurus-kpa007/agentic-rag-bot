"""Query Planner 단위 테스트"""

import json
from tests.conftest import make_mock_llm, make_text_response
from src.planner import QueryPlanner, QueryPlan


class TestQueryPlan:
    def test_is_multi_single(self):
        plan = QueryPlan(search_queries=["q1"], strategy="SINGLE")
        assert plan.is_multi() is False

    def test_is_multi_multi(self):
        plan = QueryPlan(search_queries=["q1", "q2"], strategy="MULTI")
        assert plan.is_multi() is True

    def test_is_multi_multi_but_one_query(self):
        plan = QueryPlan(search_queries=["q1"], strategy="MULTI")
        assert plan.is_multi() is False


class TestQueryPlanner:
    def test_plan_single_query(self):
        response_json = json.dumps({
            "intent": "휴가 신청 방법 확인",
            "keywords": ["휴가", "신청", "방법"],
            "search_queries": ["휴가 신청 절차 방법"],
            "strategy": "SINGLE",
        })
        llm = make_mock_llm([make_text_response(response_json)])
        planner = QueryPlanner(llm=llm)

        plan = planner.plan("휴가 쓰려면 어떻게 해?", "INTERNAL_SEARCH")

        assert plan.intent == "휴가 신청 방법 확인"
        assert "휴가" in plan.keywords
        assert len(plan.search_queries) == 1
        assert plan.strategy == "SINGLE"
        assert plan.is_multi() is False

    def test_plan_multi_query(self):
        response_json = json.dumps({
            "intent": "휴가 규정과 출장비 정책 확인",
            "keywords": ["휴가", "규정", "출장비"],
            "search_queries": ["휴가 규정 정책", "출장비 정산 방법"],
            "strategy": "MULTI",
        })
        llm = make_mock_llm([make_text_response(response_json)])
        planner = QueryPlanner(llm=llm)

        plan = planner.plan("휴가 규정이랑 출장비 알려줘", "INTERNAL_SEARCH")

        assert len(plan.search_queries) == 2
        assert plan.strategy == "MULTI"
        assert plan.is_multi() is True

    def test_plan_with_json_code_block(self):
        """```json 블록으로 감싸진 응답 파싱."""
        response = '```json\n{"intent": "test", "keywords": [], "search_queries": ["test q"], "strategy": "SINGLE"}\n```'
        llm = make_mock_llm([make_text_response(response)])
        planner = QueryPlanner(llm=llm)

        plan = planner.plan("test", "INTERNAL_SEARCH")
        assert plan.search_queries == ["test q"]

    def test_plan_fallback_on_parse_error(self):
        """파싱 실패 시 원본 쿼리로 폴백."""
        llm = make_mock_llm([make_text_response("이건 JSON이 아닙니다")])
        planner = QueryPlanner(llm=llm)

        plan = planner.plan("원본 질문", "INTERNAL_SEARCH")

        assert plan.search_queries == ["원본 질문"]
        assert plan.strategy == "SINGLE"

    def test_plan_with_conversation_history(self):
        """대화 히스토리가 있을 때 LLM에 전달 확인."""
        response_json = json.dumps({
            "intent": "이전 맥락 반영",
            "keywords": ["휴가"],
            "search_queries": ["휴가 신청"],
            "strategy": "SINGLE",
        })
        llm = make_mock_llm([make_text_response(response_json)])
        planner = QueryPlanner(llm=llm)

        history = [
            {"role": "user", "content": "휴가 규정 알려줘"},
            {"role": "assistant", "content": "휴가는 연 15일입니다."},
        ]
        plan = planner.plan("그거 다시 알려줘", "INTERNAL_SEARCH", history)

        # LLM이 히스토리를 포함한 메시지를 받았는지 확인
        call_args = llm.chat.call_args
        user_msg = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else [])[-1]["content"]
        assert "휴가 규정" in user_msg
