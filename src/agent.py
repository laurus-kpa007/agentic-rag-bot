"""Agent Core - Ollama + MCP 기반 Tool Calling 루프

핵심 로직:
1. 사용자 메시지 + MCP 도구 목록을 Ollama에게 전송
2. 응답에 tool_calls가 있으면 MCP를 통해 도구 실행
3. 실행 결과를 다시 Ollama에게 전송 (반복)
4. text 응답이 나오면 최종 답변으로 반환
"""

from src.llm_adapter import OllamaAdapter, LLMResponse
from src.mcp_client import MCPClient


class AgentCore:
    def __init__(
        self,
        llm: OllamaAdapter,
        mcp: MCPClient,
        system_prompt: str,
        max_tool_calls: int = 5,
    ):
        self.llm = llm
        self.mcp = mcp
        self.system_prompt = system_prompt
        self.max_tool_calls = max_tool_calls

    def run(self, messages: list, tool_filter: str | None = None) -> tuple[str, list[dict]]:
        """에이전트 루프를 실행하여 최종 답변과 수집된 문서를 반환한다.

        Args:
            messages: 대화 메시지 리스트
            tool_filter: 특정 도구만 활성화 (예: "search_vector_db")

        Returns:
            (answer, documents): 최종 답변 문자열과 검색된 문서 리스트
        """
        tools = self._get_filtered_tools(tool_filter)
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        collected_documents = []

        for _ in range(self.max_tool_calls):
            response = self.llm.chat(full_messages, tools=tools if tools else None)

            if not response.has_tool_calls():
                return response.content, collected_documents

            # assistant 응답 추가
            assistant_msg = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in response.tool_calls
                ]
            full_messages.append(assistant_msg)

            # 도구 실행 및 결과 수집
            for tc in response.tool_calls:
                result = self.mcp.call_tool(tc.name, tc.arguments)
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                # 검색 결과 수집
                self._collect_documents(result, collected_documents)

        return "답변 생성에 실패했습니다. 다시 시도해 주세요.", collected_documents

    def answer_with_context(
        self, query: str, documents: list[dict], conversation_history: list,
    ) -> str:
        """사전 검색된 문서를 바탕으로 답변을 생성한다 (도구 호출 없이).

        Planner가 최적화한 쿼리로 직접 검색한 결과를 LLM에게 전달하여
        답변 생성에만 집중하도록 한다.
        """
        context_parts = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "")
            header = f"[문서 {i + 1}]" + (f" (출처: {source})" if source else "")
            context_parts.append(f"{header}\n{content}")
        context = "\n\n---\n\n".join(context_parts)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)
        messages.append({
            "role": "user",
            "content": (
                f"{query}\n\n"
                f"## 검색된 참고 문서\n{context}\n\n"
                f"위 문서를 참고하여 사용자 질문에 답변하세요. "
                f"문서에 관련 정보가 있다면 반드시 해당 내용을 기반으로 답변하세요."
            ),
        })

        response = self.llm.chat(messages)
        return response.content

    def direct_answer(self, query: str, conversation_history: list) -> str:
        """도구 없이 LLM 직접 답변 (CHITCHAT용)."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})

        response = self.llm.chat(messages)
        return response.content

    def _get_filtered_tools(self, tool_filter: str | None) -> list[dict]:
        """tool_filter에 맞는 도구만 반환한다."""
        all_tools = self.mcp.get_tools_for_llm()
        if not tool_filter:
            return all_tools
        return [t for t in all_tools if tool_filter in t["name"]]

    def _collect_documents(self, result_json: str, documents: list[dict]):
        """도구 실행 결과에서 문서를 추출한다."""
        try:
            import json
            result = json.loads(result_json)
            # MCP 서버의 content 배열에서 text 추출
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if item.get("type") == "text":
                        docs = json.loads(item["text"])
                        if isinstance(docs, list):
                            documents.extend(docs)
            elif isinstance(result, list):
                documents.extend(result)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"  [Agent] 문서 수집 실패: {e}")
