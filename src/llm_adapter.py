"""LLM Adapter - Ollama 래퍼

Ollama의 /api/chat 엔드포인트를 통해 Tool Calling을 수행한다.
"""

import json
import uuid
import urllib3

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class OllamaAdapter:
    def __init__(self, model: str = "qwen3:14b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list, tools: list | None = None) -> LLMResponse:
        """Ollama /api/chat 호출. OpenAI 호환 tool calling 형식."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
            verify=False,
        )
        resp.raise_for_status()
        msg = resp.json().get("message", {})

        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                ToolCall(id=call_id, name=fn.get("name", ""), arguments=args)
            )

        return LLMResponse(
            content=msg.get("content", ""),
            tool_calls=tool_calls,
        )
