"""main.py - Simple Agentic RAG 진입점

Phase 1~4를 모두 통합한 최종 실행 파일이다.
- Phase 1: 네이티브 Tool Calling (Ollama + MCP)
- Phase 2: Router 패턴
- Phase 2.5: Query Planner
- Phase 3: 단일 피드백 루프 (CRAG)
- Phase 4: Human in the Loop
"""

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
    """하나의 사용자 질문을 전체 파이프라인으로 처리한다."""

    # Phase 2: 라우팅
    route = router.classify(query)
    print(f"  [라우팅] {route}")

    if route == "CHITCHAT":
        return agent.direct_answer(query, conversation_history)

    # Phase 2.5: 질의 분석 & 최적화
    plan = planner.plan(query, route, conversation_history)
    print(f"  [플래닝] 의도: {plan.intent}")
    print(f"  [플래닝] 검색어: {plan.search_queries}")

    # 도구 필터 결정
    tool_filter = (
        "search_vector_db" if route == "INTERNAL_SEARCH" else "web_search"
    )

    # Phase 1: Tool Calling 기반 검색
    # 원본 질문을 함께 전달하여 LLM이 맥락을 이해하도록 함
    search_hint = plan.search_queries[0]
    if search_hint != query:
        user_content = f"{query}\n\n(검색 키워드 힌트: {search_hint})"
    else:
        user_content = query

    messages = conversation_history + [
        {"role": "user", "content": user_content}
    ]

    if plan.is_multi():
        all_documents = []
        answer = ""
        for sq in plan.search_queries:
            if sq != query:
                content = f"{query}\n\n(검색 키워드 힌트: {sq})"
            else:
                content = query
            msgs = conversation_history + [{"role": "user", "content": content}]
            ans, docs = agent.run(msgs, tool_filter=tool_filter)
            all_documents.extend(docs)
            answer = ans  # 마지막 답변 사용

        # 중복 제거
        seen = set()
        documents = []
        for doc in all_documents:
            key = doc.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                documents.append(doc)
    else:
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
            msgs = conversation_history + [{"role": "user", "content": rewritten}]
            answer, documents = agent.run(msgs, tool_filter=tool_filter)
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
        msgs = conversation_history + [{"role": "user", "content": decision.new_query}]
        new_answer, _ = agent.run(msgs, tool_filter=tool_filter)
        return new_answer
    elif decision.action == "reject":
        return "답변이 거부되었습니다. 다른 방법으로 질문해 주세요."
    else:
        return answer


if __name__ == "__main__":
    main()
