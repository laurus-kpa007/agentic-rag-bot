"""RAG 파이프라인 단계별 진단 스크립트

각 단계에서 무슨 일이 일어나는지 로그를 찍어서
어디서 정보가 누락되는지 파악한다.

사용법:
  python diagnose.py "질문 내용"
  python diagnose.py  (기본 질문으로 테스트)
"""

import json
import os
import sys

os.environ["ANONYMIZED_TELEMETRY"] = "False"

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bona/bge-m3-korean:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def header(step: str):
    print(f"\n{'='*60}")
    print(f"  {step}")
    print(f"{'='*60}")


def sub(label: str, value):
    print(f"  [{label}] {value}")


def diagnose(query: str):
    print(f"\n진단 쿼리: \"{query}\"")

    # ── STEP 0: ChromaDB 상태 확인 ──
    header("STEP 0: ChromaDB 컬렉션 상태")
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collections = client.list_collections()
        sub("컬렉션 목록", [c.name for c in collections])
    except Exception as e:
        sub("ERROR", f"컬렉션 목록 조회 실패: {e}")
        return

    from src.retriever import _NoOpEF
    _noop_ef = _NoOpEF()

    try:
        children_col = client.get_collection("children", embedding_function=_noop_ef)
        parents_col = client.get_collection("parents", embedding_function=_noop_ef)
        child_count = children_col.count()
        parent_count = parents_col.count()
        sub("Children 수", child_count)
        sub("Parents 수", parent_count)
    except Exception as e:
        sub("ERROR", f"컬렉션 로딩 실패: {e}")
        return

    if child_count == 0:
        sub("PROBLEM", "Children 컬렉션이 비어있습니다! `python -m src.vectorstore.ingest` 를 먼저 실행하세요.")
        return

    # 저장된 문서 샘플 보기
    sample = children_col.get(limit=3)
    sub("Children 샘플 (3개)", "")
    for i, doc in enumerate(sample["documents"]):
        meta = sample["metadatas"][i]
        print(f"    [{i}] source={meta.get('source')}, parent_id={meta.get('parent_id')}")
        print(f"        내용(앞100자): {doc[:100]}...")
        print(f"        keywords: {meta.get('keywords', '')[:80]}")

    # ── STEP 1: 임베딩 테스트 ──
    header("STEP 1: 쿼리 임베딩")
    from src.embedding import OllamaEmbedder
    embedder = OllamaEmbedder(model=EMBEDDING_MODEL)
    try:
        query_emb = embedder.encode(query)
        sub("임베딩 차원", query_emb.shape)
        sub("임베딩 샘플", query_emb[:5])
        sub("결과", "OK")
    except Exception as e:
        sub("ERROR", f"임베딩 실패: {e}")
        return

    # ── STEP 2: 순수 벡터 검색 (top_k=10) ──
    header("STEP 2: 순수 벡터 검색 (Children, top_k=10)")
    vector_raw = children_col.query(
        query_embeddings=[query_emb.tolist()],
        n_results=min(10, child_count),
    )
    vector_ids = vector_raw["ids"][0] if vector_raw["ids"] else []
    vector_distances = vector_raw["distances"][0] if vector_raw.get("distances") else []

    if not vector_ids:
        sub("PROBLEM", "벡터 검색 결과 0건!")
    else:
        sub("결과 수", len(vector_ids))
        for i, doc_id in enumerate(vector_ids):
            dist = vector_distances[i] if i < len(vector_distances) else "N/A"
            doc_text = vector_raw["documents"][0][i]
            meta = vector_raw["metadatas"][0][i]
            similarity = 1 - dist if isinstance(dist, float) else "N/A"
            print(f"    [{i+1}] distance={dist:.4f}, similarity={similarity:.4f}")
            print(f"        source={meta.get('source')}, parent_id={meta.get('parent_id')}")
            print(f"        내용(앞120자): {doc_text[:120]}")
            print()

    # ── STEP 3: Parent Lookup 테스트 ──
    header("STEP 3: Parent Lookup (상위 3개 Child의 Parent)")
    for i, doc_id in enumerate(vector_ids[:3]):
        meta = vector_raw["metadatas"][0][i]
        parent_id = meta.get("parent_id", "")
        if parent_id:
            try:
                parent_data = parents_col.get(ids=[parent_id])
                if parent_data["documents"]:
                    parent_text = parent_data["documents"][0]
                    sub(f"Parent[{i+1}] ({parent_id})", f"길이={len(parent_text)}자")
                    print(f"        내용(앞200자): {parent_text[:200]}")
                    print()
                else:
                    sub(f"Parent[{i+1}] ({parent_id})", "문서 없음!")
            except Exception as e:
                sub(f"Parent[{i+1}] ERROR", str(e))
        else:
            sub(f"Parent[{i+1}]", "parent_id 없음!")

    # ── STEP 4: BM25 검색 테스트 ──
    header("STEP 4: BM25 검색 (top_k=10)")
    from src.retriever import BM25
    bm25 = BM25()

    all_child_data = children_col.get(limit=child_count)
    bm25_docs = []
    for i, doc in enumerate(all_child_data["documents"]):
        meta = all_child_data["metadatas"][i] if all_child_data["metadatas"] else {}
        bm25_docs.append({
            "content": doc,
            "keywords": meta.get("keywords", ""),
        })
    bm25.index(bm25_docs)

    bm25_tokens = bm25._tokenize(query)
    sub("쿼리 토큰", bm25_tokens)

    bm25_results = bm25.search(query, top_k=10)
    sub("결과 수", len([r for r in bm25_results if r[1] > 0]))
    for rank, (doc_idx, score) in enumerate(bm25_results[:5]):
        if score <= 0:
            continue
        doc_text = all_child_data["documents"][doc_idx]
        meta = all_child_data["metadatas"][doc_idx]
        print(f"    [{rank+1}] score={score:.4f}, source={meta.get('source')}")
        print(f"        내용(앞120자): {doc_text[:120]}")
        print()

    # ── STEP 5: Hybrid Search (AdvancedRetriever) ──
    header("STEP 5: Hybrid Search (AdvancedRetriever, top_k=3 기본)")
    from src.retriever import AdvancedRetriever
    retriever = AdvancedRetriever(chroma_client=client, embedder=embedder)

    results_3 = retriever.search(query=query, top_k=3)
    sub("결과 수 (top_k=3)", len(results_3))
    for i, r in enumerate(results_3):
        print(f"    [{i+1}] distance={r.distance:.4f}, rrf={r.rrf_score:.6f}")
        print(f"        source={r.metadata.get('source')}, parent_id={r.metadata.get('parent_id')}")
        print(f"        child(앞100자): {r.content[:100]}")
        print(f"        parent(앞200자): {r.parent_content[:200]}")
        print()

    # top_k=5로도 비교
    retriever2 = AdvancedRetriever(chroma_client=client, embedder=embedder)
    results_5 = retriever2.search(query=query, top_k=5)
    sub("결과 수 (top_k=5)", len(results_5))
    extra = [r for r in results_5[3:]]
    if extra:
        sub("top_k=3에서 누락된 문서들", "")
        for i, r in enumerate(extra):
            print(f"    [{i+4}] distance={r.distance:.4f}, rrf={r.rrf_score:.6f}")
            print(f"        source={r.metadata.get('source')}")
            print(f"        parent(앞200자): {r.parent_content[:200]}")
            print()

    # ── STEP 6: Planner 테스트 ──
    header("STEP 6: Query Planner 변환 결과")
    from src.llm_adapter import OllamaAdapter
    from src.planner import QueryPlanner
    llm = OllamaAdapter(model=LLM_MODEL, base_url=OLLAMA_URL)
    planner = QueryPlanner(llm=llm)

    plan = planner.plan(query, "INTERNAL_SEARCH", [])
    sub("원본 쿼리", query)
    sub("변환된 검색어", plan.search_queries)
    sub("추출 키워드", plan.keywords)
    sub("의도", plan.intent)
    sub("전략", plan.strategy)

    # 변환된 쿼리로 다시 검색해서 비교
    if plan.search_queries and plan.search_queries[0] != query:
        planned_query = plan.search_queries[0]
        sub("COMPARE", f"변환 쿼리 \"{planned_query}\" 로 재검색")
        retriever3 = AdvancedRetriever(chroma_client=client, embedder=embedder)
        results_planned = retriever3.search(query=planned_query, top_k=3)
        sub("변환 쿼리 결과 수", len(results_planned))
        for i, r in enumerate(results_planned):
            print(f"    [{i+1}] distance={r.distance:.4f}")
            print(f"        parent(앞150자): {r.parent_content[:150]}")
            print()

        # 원본 vs 변환 비교
        original_parents = {r.metadata.get('parent_id') for r in results_3}
        planned_parents = {r.metadata.get('parent_id') for r in results_planned}
        if original_parents != planned_parents:
            sub("DIFF", f"원본과 변환 쿼리의 검색 결과가 다릅니다!")
            sub("원본에만 있는 Parent", original_parents - planned_parents)
            sub("변환에만 있는 Parent", planned_parents - original_parents)
        else:
            sub("SAME", "원본과 변환 쿼리의 검색 결과가 동일합니다")

    # ── STEP 7: 실제 Agent에 전달되는 메시지 확인 ──
    header("STEP 7: Agent에 전달되는 메시지 (문제 확인)")
    from src.prompts.system import SYSTEM_PROMPT
    sub("시스템 프롬프트 길이", f"{len(SYSTEM_PROMPT)}자")
    sub("시스템 프롬프트 내용", "")
    print(f"    {SYSTEM_PROMPT[:300]}...")
    print()
    sub("ISSUE 1", "GENERATOR_PROMPT가 정의되어 있지만 사용되지 않음!")
    sub("ISSUE 2", f"Agent에 전달되는 user 메시지 = planner 쿼리 \"{plan.search_queries[0]}\" (원본 \"{query}\" 아님!)")
    sub("ISSUE 3", "시스템 프롬프트에 '검색 결과를 근거로 답변하라'는 명시적 지시가 약함")

    # ── STEP 8: 전체 문서에서 키워드 직접 검색 ──
    header("STEP 8: 전체 Children에서 키워드 직접 매칭")
    search_terms = bm25_tokens[:5]
    sub("검색 키워드", search_terms)

    matched_count = 0
    for i, doc in enumerate(all_child_data["documents"]):
        if any(term in doc for term in search_terms):
            matched_count += 1
    sub("키워드 포함 Child 수", f"{matched_count} / {child_count}")

    if matched_count > 0 and len(results_3) == 0:
        sub("PROBLEM", "키워드가 포함된 문서가 있는데 벡터 검색이 못 찾고 있습니다!")

    # ── 요약 ──
    header("진단 요약")
    print()
    issues = []

    if child_count == 0:
        issues.append("CRITICAL: ChromaDB가 비어있습니다")

    if len(results_3) == 0:
        issues.append("CRITICAL: 검색 결과가 0건입니다")

    if results_3 and results_3[0].distance > 0.5:
        issues.append(f"WARNING: 최상위 결과의 distance가 높습니다 ({results_3[0].distance:.4f}). 임베딩 품질 또는 쿼리 문제 가능성")

    if plan.search_queries and plan.search_queries[0] != query:
        issues.append(f"WARNING: Planner가 쿼리를 변환함 \"{query}\" → \"{plan.search_queries[0]}\"")

    issues.append("FIX NEEDED: GENERATOR_PROMPT가 사용되지 않아 LLM이 검색 결과를 무시할 수 있음")
    issues.append("FIX NEEDED: Agent에 원본 질문이 아닌 planner 쿼리만 전달됨")
    issues.append("FIX NEEDED: top_k=3 기본값이 너무 낮음 (5 권장)")

    for i, issue in enumerate(issues):
        print(f"  {i+1}. {issue}")

    print()


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "연차 휴가 며칠 쓸 수 있어?"
    diagnose(query)
