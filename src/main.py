"""main.py - Simple Agentic RAG 진입점

Phase 1~4를 모두 통합한 최종 실행 파일이다.
- Phase 1: Planner 최적화 쿼리로 직접 검색 + LLM 답변 생성
- Phase 2: Router 패턴
- Phase 2.5: Query Planner
- Phase 3: 단일 피드백 루프 (CRAG)
- Phase 4: Human in the Loop
"""

import json

from src.config import Config
from src.llm_adapter import OllamaAdapter
from src.mcp_client import MCPClient
from src.agent import AgentCore
from src.router import Router
from src.planner import QueryPlanner
from src.grader import Grader, QueryRewriter
from src.hitl import HITLManager, HITLContext, FeedbackStore
from src.prompts.system import SYSTEM_PROMPT


def main():
    config = Config()

    # 핵심 인프라 초기화
    llm = OllamaAdapter(model=config.llm_model, base_url=config.ollama_url)
    mcp = MCPClient(config_path=config.mcp_config_path)

    print("Simple Agentic RAG Bot 시작 중...")
    mcp.connect_all()

    # 파이프라인 컴포넌트 초기화 (모두 같은 LLM 인스턴스 공유)
    agent = AgentCore(
        llm=llm, mcp=mcp,
        system_prompt=SYSTEM_PROMPT,
        max_tool_calls=config.max_tool_calls,
    )
    router = Router(llm=llm)
    planner = QueryPlanner(llm=llm)
    grader = Grader(llm=llm)
    rewriter = QueryRewriter(llm=llm)
    hitl = HITLManager(mode=config.hitl_mode)
    feedback_store = FeedbackStore()

    conversation_history = []

    print("준비 완료! (종료: quit)\n")

    try:
        while True:
            query = input("[사용자] ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "종료"):
                break

            answer = process_query(
                query=query,
                conversation_history=conversation_history,
                agent=agent,
                router=router,
                planner=planner,
                grader=grader,
                rewriter=rewriter,
                hitl=hitl,
            )

            print(f"\n[봇] {answer}\n")

            # 대화 히스토리 관리 (최근 N턴)
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})
            max_messages = config.max_history_turns * 2
            conversation_history = conversation_history[-max_messages:]

            # 사후 피드백 수집
            feedback = hitl.collect_feedback(query, answer)
            if feedback:
                feedback_store.save(feedback)

    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        mcp.disconnect_all()


def _find_search_tool(mcp, route: str) -> str | None:
    """라우트에 맞는 MCP 검색 도구 이름을 찾는다."""
    keyword = "search_vector_db" if route == "INTERNAL_SEARCH" else "web_search"
    for tool in mcp.get_tools_for_llm():
        if keyword in tool["name"]:
            return tool["name"]
    return None


def _parse_mcp_results(result_json: str) -> list[dict]:
    """MCP 도구 호출 결과에서 문서 리스트를 추출한다."""
    try:
        result = json.loads(result_json)
        if isinstance(result, dict) and "content" in result:
            for item in result["content"]:
                if item.get("type") == "text":
                    docs = json.loads(item["text"])
                    if isinstance(docs, list):
                        return docs
        elif isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return []


def _dedup_documents(documents: list[dict]) -> list[dict]:
    """문서 중복 제거."""
    seen = set()
    result = []
    for doc in documents:
        key = doc.get("content", "")[:100]
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def _direct_search(mcp, tool_name: str, queries: list[str], top_k: int = 5) -> list[dict]:
    """Planner의 최적화된 쿼리로 MCP 검색을 직접 수행한다."""
    all_docs = []
    for sq in queries:
        result_json = mcp.call_tool(tool_name, {"query": sq, "top_k": top_k})
        docs = _parse_mcp_results(result_json)
        print(f"  [검색] '{sq}' → {len(docs)}건")
        for i, doc in enumerate(docs):
            dist = doc.get("distance", "?")
            preview = doc.get("content", "")[:80].replace("\n", " ")
            print(f"    [{i + 1}] (거리={dist}) {preview}...")
        all_docs.extend(docs)
    return _dedup_documents(all_docs)


def process_query(
    query: str,
    conversation_history: list,
    agent: AgentCore,
    router: Router,
    planner: QueryPlanner,
    grader: Grader,
    rewriter: QueryRewriter,
    hitl: HITLManager,
) -> str:
    """하나의 사용자 질문을 전체 파이프라인으로 처리한다.

    핵심 변경: Planner가 최적화한 검색어로 직접 MCP 검색을 수행한 뒤,
    검색 결과를 LLM에게 전달하여 답변 생성에만 집중하도록 한다.
    (기존: LLM이 도구 호출 쿼리를 독립적으로 결정 → Planner 쿼리 무시 가능)
    """

    # Phase 2: 라우팅
    route = router.classify(query)
    print(f"  [라우팅] {route}")

    if route == "CHITCHAT":
        return agent.direct_answer(query, conversation_history)

    # Phase 2.5: 질의 분석 & 최적화
    plan = planner.plan(query, route, conversation_history)
    print(f"  [플래닝] 의도: {plan.intent}")
    print(f"  [플래닝] 검색어: {plan.search_queries}")

    # Phase 1: Planner 쿼리로 직접 검색
    tool_name = _find_search_tool(agent.mcp, route)
    tool_filter = "search_vector_db" if route == "INTERNAL_SEARCH" else "web_search"
    documents = []

    if tool_name:
        documents = _direct_search(agent.mcp, tool_name, plan.search_queries)

    # 검색 결과 기반 답변 생성
    if documents:
        answer = agent.answer_with_context(query, documents, conversation_history)
    else:
        # 폴백: 기존 Agent 루프 (도구 호출 포함)
        messages = conversation_history + [{"role": "user", "content": query}]
        answer, documents = agent.run(messages, tool_filter=tool_filter)

    # Phase 3: 검색 결과 평가
    retry_count = 0
    grade = "PASS"
    if documents:
        grade = grader.evaluate(query, documents)
        print(f"  [평가] {grade}")

        if grade == "FAIL":
            rewritten = rewriter.rewrite(query)
            print(f"  [재작성] {rewritten}")
            # 재작성된 쿼리로 직접 재검색
            if tool_name:
                new_docs = _direct_search(agent.mcp, tool_name, [rewritten])
                if new_docs:
                    documents = new_docs
                    answer = agent.answer_with_context(query, documents, conversation_history)
            retry_count = 1
            grade = "PASS"  # 재검색 후 강제 진행

    # Phase 4: Human in the Loop
    confidence = hitl.calculator.calculate(
        grader_result=grade,
        vector_scores=[d.get("distance", 0.5) for d in documents],
        retry_count=retry_count,
        doc_count=len(documents),
    )

    context = HITLContext(
        query=query,
        answer=answer,
        confidence=confidence,
        documents=documents,
        route=route,
        search_queries=plan.search_queries,
    )

    decision = hitl.request_review(context)

    if decision.action == "approve":
        return answer
    elif decision.action == "edit":
        return decision.edited_answer
    elif decision.action == "retry":
        # HITL 재시도도 직접 검색으로
        if tool_name:
            retry_docs = _direct_search(agent.mcp, tool_name, [decision.new_query])
            if retry_docs:
                return agent.answer_with_context(
                    decision.new_query, retry_docs, conversation_history,
                )
        msgs = conversation_history + [{"role": "user", "content": decision.new_query}]
        new_answer, _ = agent.run(msgs, tool_filter=tool_filter)
        return new_answer
    elif decision.action == "reject":
        return "답변이 거부되었습니다. 다른 방법으로 질문해 주세요."
    else:
        return answer


if __name__ == "__main__":
    main()
