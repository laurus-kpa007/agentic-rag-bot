"""테스트용 공통 유틸리티 및 Mock 객체"""

import json
from unittest.mock import MagicMock
from src.llm_adapter import OllamaAdapter, LLMResponse, ToolCall


def make_mock_llm(responses: list[LLMResponse] | None = None) -> OllamaAdapter:
    """순차적으로 응답을 반환하는 Mock LLM을 생성한다."""
    mock = MagicMock(spec=OllamaAdapter)
    if responses:
        mock.chat.side_effect = responses
    return mock


def make_text_response(text: str) -> LLMResponse:
    """텍스트만 반환하는 LLMResponse."""
    return LLMResponse(content=text, tool_calls=[])


def make_tool_response(tool_name: str, arguments: dict, content: str = "") -> LLMResponse:
    """도구 호출을 포함하는 LLMResponse."""
    return LLMResponse(
        content=content,
        tool_calls=[ToolCall(id="call_test1", name=tool_name, arguments=arguments)],
    )


def make_mock_mcp(tools: list[dict] | None = None, call_results: dict | None = None):
    """Mock MCP Client를 생성한다."""
    from src.mcp_client import MCPClient

    mock = MagicMock(spec=MCPClient)

    default_tools = tools or [
        {
            "name": "vector-search__search_vector_db",
            "description": "사내 문서 검색",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"],
            },
        },
        {
            "name": "web-search__web_search",
            "description": "외부 웹 검색",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ]
    mock.get_tools_for_llm.return_value = default_tools

    default_results = call_results or {}

    def mock_call_tool(name, args):
        if name in default_results:
            return default_results[name]
        # 기본: 검색 결과 반환 (distance 낮음 = 유사도 높음)
        docs = [
            {"content": "테스트 문서 내용입니다.", "metadata": {"source": "test.md"}, "distance": 0.1},
            {"content": "관련 문서 두 번째입니다.", "metadata": {"source": "test2.md"}, "distance": 0.15},
            {"content": "관련 문서 세 번째입니다.", "metadata": {"source": "test3.md"}, "distance": 0.2},
        ]
        return json.dumps(
            {"content": [{"type": "text", "text": json.dumps(docs, ensure_ascii=False)}]},
            ensure_ascii=False,
        )

    mock.call_tool.side_effect = mock_call_tool
    return mock
