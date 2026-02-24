"""MCP Client - MCP 서버 연결 및 도구 관리

시작 시 mcp_config.json의 서버에 연결하여 도구 목록을 수집하고,
Agent Core가 도구를 호출할 때 해당 MCP 서버로 요청을 중계한다.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MCPTool:
    server_name: str
    name: str
    description: str
    parameters: dict

    @property
    def full_name(self) -> str:
        return f"{self.server_name}__{self.name}"

    def to_llm_tool(self) -> dict:
        return {
            "name": self.full_name,
            "description": self.description,
            "parameters": self.parameters,
        }


class MCPClient:
    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = Path(config_path)
        self.servers: dict[str, subprocess.Popen] = {}
        self.tools: dict[str, MCPTool] = {}

    def connect_all(self):
        """mcp_config.json의 모든 서버에 연결하여 도구를 수집한다."""
        if not self.config_path.exists():
            print("  [MCP] 설정 파일을 찾을 수 없습니다:", self.config_path)
            return

        config = json.loads(self.config_path.read_text())
        for name, cfg in config.get("mcpServers", {}).items():
            try:
                proc = subprocess.Popen(
                    [cfg["command"]] + cfg.get("args", []),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**__import__("os").environ, **cfg.get("env", {})},
                )
                self.servers[name] = proc

                # 초기화 핸드셰이크
                self._send(proc, "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "agentic-rag-bot"},
                })

                # 도구 목록 수집
                result = self._send(proc, "tools/list", {})
                for t in result.get("tools", []):
                    tool = MCPTool(
                        server_name=name,
                        name=t["name"],
                        description=t.get("description", ""),
                        parameters=t.get("inputSchema", {}),
                    )
                    self.tools[tool.full_name] = tool

                print(f"  [MCP] {name} 연결 완료 (도구 {len(result.get('tools', []))}개)")
            except Exception as e:
                print(f"  [MCP] {name} 연결 실패: {e}")

    def get_tools_for_llm(self) -> list[dict]:
        """LLM에 전달할 도구 스키마 목록을 반환한다."""
        return [t.to_llm_tool() for t in self.tools.values()]

    def call_tool(self, full_name: str, arguments: dict) -> str:
        """MCP 서버에 도구 호출을 중계한다."""
        tool = self.tools.get(full_name)
        if not tool:
            return json.dumps({"error": f"도구 '{full_name}'을 찾을 수 없습니다."})

        proc = self.servers.get(tool.server_name)
        if not proc:
            return json.dumps({"error": f"서버 '{tool.server_name}'에 연결되지 않았습니다."})

        result = self._send(proc, "tools/call", {
            "name": tool.name,
            "arguments": arguments,
        })
        return json.dumps(result, ensure_ascii=False)

    def _send(self, proc: subprocess.Popen, method: str, params: dict) -> dict:
        """MCP 서버에 JSON-RPC 요청을 보내고 응답을 받는다."""
        req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        proc.stdin.write((json.dumps(req) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline().decode()
        if not line:
            return {}
        return json.loads(line).get("result", {})

    def disconnect_all(self):
        """모든 MCP 서버 프로세스를 종료한다."""
        for name, proc in self.servers.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self.servers.clear()
        self.tools.clear()
