"""MCP 서버 프로토콜 단위 테스트

handle_request 함수를 직접 임포트하여 JSON-RPC 프로토콜을 검증한다.
vector_search_server는 lazy init이므로 import 시 모델을 로딩하지 않는다.
"""

from src.mcp_servers.vector_search_server import handle_request as vs_handle
from src.mcp_servers.web_search_server import handle_request as ws_handle


class TestVectorSearchServer:
    def test_initialize(self):
        result = vs_handle({"method": "initialize", "params": {}})
        assert result["protocolVersion"] == "2024-11-05"
        assert "tools" in result["capabilities"]

    def test_tools_list(self):
        result = vs_handle({"method": "tools/list", "params": {}})
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search_vector_db"

    def test_unknown_method(self):
        result = vs_handle({"method": "unknown/method", "params": {}})
        assert result == {}

    def test_notifications_initialized(self):
        result = vs_handle({"method": "notifications/initialized", "params": {}})
        assert result == {}


class TestWebSearchServer:
    def test_initialize(self):
        result = ws_handle({"method": "initialize", "params": {}})
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "web-search"

    def test_tools_list(self):
        result = ws_handle({"method": "tools/list", "params": {}})
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "web_search"
