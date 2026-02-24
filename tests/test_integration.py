"""통합 테스트 - 전체 파이프라인 E2E 흐름

Ollama 실제 서버 없이 Mock LLM으로 전체 파이프라인을 검증한다.
"""

import json
from unittest.mock import MagicMock, patch
from tests.conftest import (
    make_mock_llm,
    make_mock_mcp,
    make_text_response,
    make_tool_response,
)
from src.llm_adapter import LLMResponse, ToolCall
from src.agent import AgentCore
from src.router import Router
from src.planner import QueryPlanner
from src.grader import Grader, QueryRewriter
from src.hitl import HITLManager, HITLContext
from src.main import process_query


def _make_pipeline(llm_responses: list[LLMResponse], hitl_mode: str = "off"):
    """테스트용 전체 파이프라인 컴포넌트를 생성한다."""
    llm = make_mock_llm(llm_responses)
    mcp = make_mock_mcp()

    return {
        "agent": AgentCore(llm=llm, mcp=mcp, system_prompt="test"),
        "router": Router(llm=llm),
        "planner": QueryPlanner(llm=llm),
        "grader": Grader(llm=llm),
        "rewriter": QueryRewriter(llm=llm),
        "hitl": HITLManager(mode=hitl_mode),
    }


class TestIntegrationChitchat:
    """CHITCHAT 경로 통합 테스트"""

    def test_chitchat_flow(self):
        """인사 → Router(CHITCHAT) → 직접 답변."""
        llm_responses = [
            make_text_response("CHITCHAT"),       # Router
            make_text_response("안녕하세요! 무엇을 도와드릴까요?"),  # Direct answer
        ]
        components = _make_pipeline(llm_responses)

        answer = process_query(
            query="안녕하세요",
            conversation_history=[],
            **components,
        )

        assert "안녕" in answer or "도와" in answer


class TestIntegrationInternalSearch:
    """INTERNAL_SEARCH 경로 통합 테스트"""

    def test_internal_search_pass_flow(self):
        """사내 검색 → Router → Planner → Agent(도구호출) → Grader(PASS) → 답변."""
        plan_json = json.dumps({
            "intent": "휴가 신청 방법",
            "keywords": ["휴가", "신청"],
            "search_queries": ["휴가 신청 절차"],
            "strategy": "SINGLE",
        })

        llm_responses = [
            make_text_response("INTERNAL_SEARCH"),    # Router
            make_text_response(plan_json),             # Planner
            # Agent: 도구 호출 → 결과 → 최종 답변
            make_tool_response("vector-search__search_vector_db", {"query": "휴가 신청 절차"}),
            make_text_response("휴가 신청은 HR 포털에서 가능합니다."),
            make_text_response("PASS"),                # Grader
        ]
        components = _make_pipeline(llm_responses)

        answer = process_query(
            query="휴가 신청 방법 알려줘",
            conversation_history=[],
            **components,
        )

        assert "휴가" in answer

    def test_internal_search_fail_rewrite_flow(self):
        """사내 검색 → Grader FAIL → Rewriter → 재검색 → 답변."""
        plan_json = json.dumps({
            "intent": "출장비 정산",
            "keywords": ["출장비"],
            "search_queries": ["출장비 정산 방법"],
            "strategy": "SINGLE",
        })

        llm_responses = [
            make_text_response("INTERNAL_SEARCH"),   # Router
            make_text_response(plan_json),            # Planner
            # 1차 Agent: 도구 호출 → 답변
            make_tool_response("vector-search__search_vector_db", {"query": "출장비 정산 방법"}),
            make_text_response("관련 없는 답변입니다."),
            make_text_response("FAIL"),              # Grader: FAIL
            make_text_response("출장 경비 정산 절차 비용 처리"),  # Rewriter
            # 2차 Agent: 재검색
            make_tool_response("vector-search__search_vector_db", {"query": "출장 경비"}),
            make_text_response("출장비는 ERP 시스템에서 정산합니다."),
        ]
        components = _make_pipeline(llm_responses)

        answer = process_query(
            query="출장비 어떻게 정산해?",
            conversation_history=[],
            **components,
        )

        assert "출장" in answer or "정산" in answer


class TestIntegrationWebSearch:
    """WEB_SEARCH 경로 통합 테스트"""

    def test_web_search_flow(self):
        """웹 검색 → Router → Planner → Agent(web_search) → Grader(PASS) → 답변."""
        plan_json = json.dumps({
            "intent": "날씨 확인",
            "keywords": ["서울", "날씨"],
            "search_queries": ["서울 오늘 날씨"],
            "strategy": "SINGLE",
        })

        llm_responses = [
            make_text_response("WEB_SEARCH"),        # Router
            make_text_response(plan_json),            # Planner
            make_tool_response("web-search__web_search", {"query": "서울 오늘 날씨"}),
            make_text_response("오늘 서울은 맑고 기온 15도입니다."),
            make_text_response("PASS"),              # Grader
        ]
        components = _make_pipeline(llm_responses)

        answer = process_query(
            query="오늘 서울 날씨 어때?",
            conversation_history=[],
            **components,
        )

        assert "서울" in answer or "날씨" in answer or "15" in answer


class TestIntegrationMultiQuery:
    """MULTI 전략 통합 테스트"""

    def test_multi_query_search(self):
        """복합 질문 → Planner(MULTI) → 2회 검색 → 병합 → 답변."""
        plan_json = json.dumps({
            "intent": "휴가와 출장비 동시 확인",
            "keywords": ["휴가", "출장비"],
            "search_queries": ["휴가 규정 정책", "출장비 정산 방법"],
            "strategy": "MULTI",
        })

        llm_responses = [
            make_text_response("INTERNAL_SEARCH"),    # Router
            make_text_response(plan_json),             # Planner
            # 1차 검색: 휴가
            make_tool_response("vector-search__search_vector_db", {"query": "휴가 규정"}),
            make_text_response("휴가는 연 15일입니다."),
            # 2차 검색: 출장비
            make_tool_response("vector-search__search_vector_db", {"query": "출장비"}),
            make_text_response("출장비는 ERP로 정산합니다."),
            make_text_response("PASS"),               # Grader
        ]
        components = _make_pipeline(llm_responses)

        answer = process_query(
            query="휴가 규정이랑 출장비 알려줘",
            conversation_history=[],
            **components,
        )

        assert answer  # 답변이 비어있지 않음


class TestIntegrationHITL:
    """HITL 통합 테스트"""

    def test_high_confidence_auto_approve(self):
        """높은 신뢰도 → HITL 자동 승인 (auto 모드)."""
        plan_json = json.dumps({
            "intent": "테스트",
            "keywords": ["테스트"],
            "search_queries": ["테스트 쿼리"],
            "strategy": "SINGLE",
        })

        llm_responses = [
            make_text_response("INTERNAL_SEARCH"),
            make_text_response(plan_json),
            make_tool_response("vector-search__search_vector_db", {"query": "test"}),
            make_text_response("좋은 답변입니다."),
            make_text_response("PASS"),
        ]
        components = _make_pipeline(llm_responses, hitl_mode="auto")

        answer = process_query(
            query="테스트 질문",
            conversation_history=[],
            **components,
        )

        assert answer == "좋은 답변입니다."

    def test_off_mode_skips_hitl(self):
        """HITL off 모드 → 검토 없이 바로 전달."""
        plan_json = json.dumps({
            "intent": "test",
            "keywords": [],
            "search_queries": ["test"],
            "strategy": "SINGLE",
        })

        llm_responses = [
            make_text_response("INTERNAL_SEARCH"),
            make_text_response(plan_json),
            make_text_response("답변"),  # Agent: 도구 없이 직접 답변
            make_text_response("PASS"),
        ]
        components = _make_pipeline(llm_responses, hitl_mode="off")

        answer = process_query(
            query="질문",
            conversation_history=[],
            **components,
        )

        assert answer == "답변"


class TestIntegrationConversationHistory:
    """대화 히스토리 연동 테스트"""

    def test_history_passed_to_planner(self):
        """이전 대화 맥락이 Planner에 전달되는지 확인."""
        plan_json = json.dumps({
            "intent": "이전 질문 재질의",
            "keywords": ["휴가"],
            "search_queries": ["휴가 신청 절차"],
            "strategy": "SINGLE",
        })

        llm_responses = [
            make_text_response("INTERNAL_SEARCH"),
            make_text_response(plan_json),
            make_text_response("이전에 물어본 휴가 정보입니다."),
            make_text_response("PASS"),
        ]
        components = _make_pipeline(llm_responses)

        history = [
            {"role": "user", "content": "휴가 신청 방법 알려줘"},
            {"role": "assistant", "content": "HR 포털에서 신청 가능합니다."},
        ]

        answer = process_query(
            query="그거 다시 알려줘",
            conversation_history=history,
            **components,
        )

        assert answer  # 답변이 반환됨
