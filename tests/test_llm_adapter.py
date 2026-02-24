"""LLM Adapter 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock
from src.llm_adapter import OllamaAdapter, LLMResponse, ToolCall


class TestLLMResponse:
    def test_has_tool_calls_empty(self):
        resp = LLMResponse(content="hello")
        assert resp.has_tool_calls() is False

    def test_has_tool_calls_with_tools(self):
        resp = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="search", arguments={"q": "test"})],
        )
        assert resp.has_tool_calls() is True


class TestOllamaAdapter:
    @patch("src.llm_adapter.requests.post")
    def test_chat_text_only(self, mock_post):
        """도구 없이 텍스트만 반환하는 경우."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "안녕하세요!", "role": "assistant"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = OllamaAdapter(model="gemma3:12b")
        result = adapter.chat(messages=[{"role": "user", "content": "안녕"}])

        assert result.content == "안녕하세요!"
        assert result.has_tool_calls() is False

    @patch("src.llm_adapter.requests.post")
    def test_chat_with_tool_calls(self, mock_post):
        """도구 호출을 포함하는 응답."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_vector_db",
                            "arguments": {"query": "휴가 신청", "top_k": 3},
                        }
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = OllamaAdapter()
        result = adapter.chat(
            messages=[{"role": "user", "content": "test"}],
            tools=[{"name": "search", "description": "desc", "parameters": {}}],
        )

        assert result.has_tool_calls() is True
        assert result.tool_calls[0].name == "search_vector_db"
        assert result.tool_calls[0].arguments == {"query": "휴가 신청", "top_k": 3}

    @patch("src.llm_adapter.requests.post")
    def test_chat_with_string_arguments(self, mock_post):
        """arguments가 문자열로 오는 경우 JSON 파싱."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"query": "test"}',
                        }
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = OllamaAdapter()
        result = adapter.chat(messages=[{"role": "user", "content": "test"}])

        assert result.tool_calls[0].arguments == {"query": "test"}

    @patch("src.llm_adapter.requests.post")
    def test_chat_sends_correct_payload(self, mock_post):
        """올바른 페이로드를 전송하는지 확인."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = OllamaAdapter(model="test-model", base_url="http://test:11434")
        adapter.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t1", "description": "d1", "parameters": {"type": "object"}}],
        )

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "t1"
