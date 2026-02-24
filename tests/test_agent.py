"""Agent Core 단위 테스트"""

import json
from tests.conftest import (
    make_mock_llm,
    make_mock_mcp,
    make_text_response,
    make_tool_response,
)
from src.agent import AgentCore


class TestAgentCore:
    def test_direct_answer_no_tools(self):
        """도구 없이 직접 답변."""
        llm = make_mock_llm([make_text_response("안녕하세요! 반갑습니다.")])
        mcp = make_mock_mcp()
        agent = AgentCore(llm=llm, mcp=mcp, system_prompt="test prompt")

        answer = agent.direct_answer("안녕", [])
        assert "안녕" in answer or "반갑" in answer

    def test_run_text_only_response(self):
        """도구 호출 없이 텍스트만 반환."""
        llm = make_mock_llm([make_text_response("직접 답변입니다.")])
        mcp = make_mock_mcp()
        agent = AgentCore(llm=llm, mcp=mcp, system_prompt="test")

        answer, docs = agent.run([{"role": "user", "content": "안녕"}])
        assert answer == "직접 답변입니다."
        assert docs == []

    def test_run_with_tool_call(self):
        """도구 호출 후 최종 답변 반환."""
        llm = make_mock_llm([
            # 1차: 도구 호출
            make_tool_response(
                "vector-search__search_vector_db",
                {"query": "휴가 신청"},
            ),
            # 2차: 최종 답변
            make_text_response("휴가 신청은 HR 포털에서 가능합니다."),
        ])
        mcp = make_mock_mcp()
        agent = AgentCore(llm=llm, mcp=mcp, system_prompt="test")

        answer, docs = agent.run([{"role": "user", "content": "휴가 신청 방법"}])

        assert "휴가" in answer
        assert mcp.call_tool.called
        assert len(docs) > 0

    def test_run_max_tool_calls_limit(self):
        """최대 도구 호출 횟수 초과 시 실패 메시지."""
        # 계속 도구만 호출하는 응답
        tool_resp = make_tool_response("vector-search__search_vector_db", {"query": "test"})
        llm = make_mock_llm([tool_resp] * 6)
        mcp = make_mock_mcp()
        agent = AgentCore(llm=llm, mcp=mcp, system_prompt="test", max_tool_calls=3)

        answer, _ = agent.run([{"role": "user", "content": "test"}])
        assert "실패" in answer
        assert llm.chat.call_count == 3

    def test_run_with_tool_filter(self):
        """tool_filter로 특정 도구만 활성화."""
        llm = make_mock_llm([make_text_response("답변")])
        mcp = make_mock_mcp()
        agent = AgentCore(llm=llm, mcp=mcp, system_prompt="test")

        agent.run([{"role": "user", "content": "test"}], tool_filter="search_vector_db")

        # LLM에 전달된 도구 목록 확인
        call_args = llm.chat.call_args
        tools = call_args.kwargs.get("tools") or (call_args[0][1] if len(call_args[0]) > 1 else None)
        if tools:
            for tool in tools:
                assert "search_vector_db" in tool["name"]
