"""Advanced Retriever - Hybrid Search + RRF + LLM Reranking

기본 벡터 검색을 대체하는 고급 검색 모듈이다.
1. Vector Search: 의미적 유사도 기반 검색
2. BM25 Search: 키워드 매칭 기반 검색
3. RRF (Reciprocal Rank Fusion): 두 결과를 순위 기반으로 합산
4. LLM Reranking: 상위 결과를 LLM으로 관련도 재정렬
5. Parent Lookup: Child 매칭 → Parent 컨텍스트 확장
"""

import json
import math
import os
import re
from dataclasses import dataclass, field

os.environ["ANONYMIZED_TELEMETRY"] = "False"


class _NoOpEF:
    """chromadb 기본 EF(onnx 다운로드)를 방지하는 더미."""

    def __call__(self, input):
        return [[0.0] * 10 for _ in input]


_noop_ef = _NoOpEF()


@dataclass
class RetrievalResult:
    content: str
    parent_content: str
    metadata: dict = field(default_factory=dict)
    vector_rank: int = 0
    bm25_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: str = ""  # "HIGH" | "MEDIUM" | "LOW"
    distance: float = 0.0


class BM25:
    """순수 Python BM25 구현 (외부 의존성 없음)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths: list[int] = []
        self.term_freqs: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        self.documents: list[dict] = []

    def index(self, documents: list[dict]):
        """문서 목록을 BM25 인덱스에 추가한다."""
        self.documents = documents
        self.doc_count = len(documents)

        total_length = 0
        for doc in documents:
            # keywords 메타데이터 또는 본문에서 토큰 추출
            text = doc.get("keywords", "") + " " + doc.get("content", "")
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)

            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avg_dl = total_length / max(self.doc_count, 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """쿼리에 대해 BM25 스코어가 높은 문서 인덱스를 반환한다."""
        query_tokens = self._tokenize(query)
        scores = []

        for i in range(self.doc_count):
            score = 0.0
            dl = self.doc_lengths[i]

            for token in query_tokens:
                if token not in self.term_freqs[i]:
                    continue
                tf = self.term_freqs[i][token]
                df = self.doc_freq.get(token, 0)
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                score += idf * numerator / denominator

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """한국어 + 영어 토큰 추출."""
        return re.findall(r"[가-힣a-zA-Z0-9]{2,}", text.lower())


def reciprocal_rank_fusion(
    vector_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """RRF로 두 검색 결과의 순위를 합산한다.

    RRF_score(doc) = Σ 1/(k + rank_i)
    """
    rrf_scores: dict[int, float] = {}

    for rank, (doc_idx, _) in enumerate(vector_results):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)

    for rank, (doc_idx, _) in enumerate(bm25_results):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class AdvancedRetriever:
    """Hybrid Search + RRF + Parent Lookup + Optional LLM Reranking."""

    def __init__(self, chroma_client, embedder, llm=None):
        self.chroma = chroma_client
        self.embedder = embedder
        self.llm = llm  # Optional: LLM reranking용
        self.bm25 = BM25()
        self._bm25_indexed = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = False,
    ) -> list[RetrievalResult]:
        """Advanced Hybrid Search를 수행한다."""
        try:
            children_col = self.chroma.get_collection(
                "children", embedding_function=_noop_ef,
            )
            parents_col = self.chroma.get_collection(
                "parents", embedding_function=_noop_ef,
            )
        except Exception as e:
            print(f"  [Retriever] 컬렉션 로딩 실패: {e}")
            return []

        # BM25 인덱스 구축 (최초 1회)
        if not self._bm25_indexed:
            self._build_bm25_index(children_col)

        # 검색 후보 수 (RRF 합산 전) — 더 넓은 후보군 확보
        candidate_k = min(top_k * 4, 20)

        # 1. Vector Search
        query_embedding = self.embedder.encode(query).tolist()
        vector_raw = children_col.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
        )
        vector_ids = vector_raw["ids"][0] if vector_raw["ids"] else []
        vector_distances = vector_raw["distances"][0] if vector_raw.get("distances") else []

        # ID → index 매핑
        id_to_data = {}
        for i, doc_id in enumerate(vector_ids):
            id_to_data[doc_id] = {
                "content": vector_raw["documents"][0][i],
                "metadata": vector_raw["metadatas"][0][i],
                "distance": vector_distances[i] if i < len(vector_distances) else 1.0,
            }

        vector_results = [(i, 1.0 - (vector_distances[i] if i < len(vector_distances) else 1.0))
                          for i in range(len(vector_ids))]

        # 2. BM25 Search
        bm25_results = self.bm25.search(query, top_k=candidate_k)

        # BM25 결과의 ID 가져오기
        bm25_id_map = {}
        all_child_data = children_col.get(limit=children_col.count())
        all_ids = all_child_data["ids"]
        for rank_idx, (doc_idx, score) in enumerate(bm25_results):
            if doc_idx < len(all_ids):
                bm25_doc_id = all_ids[doc_idx]
                bm25_id_map[doc_idx] = bm25_doc_id
                if bm25_doc_id not in id_to_data:
                    # BM25에서만 나온 결과도 수집
                    idx_in_all = doc_idx
                    id_to_data[bm25_doc_id] = {
                        "content": all_child_data["documents"][idx_in_all],
                        "metadata": all_child_data["metadatas"][idx_in_all],
                        "distance": 0.5,  # BM25 전용은 distance 없음
                    }

        # 3. RRF 합산 — 통합 인덱스 기반
        # vector는 0~N-1 인덱스, BM25는 전체 컬렉션 인덱스 → 통합 필요
        unified_vector = []
        for i, doc_id in enumerate(vector_ids):
            unified_vector.append((doc_id, 1.0 - id_to_data[doc_id]["distance"]))

        unified_bm25 = []
        for doc_idx, score in bm25_results:
            if doc_idx in bm25_id_map:
                unified_bm25.append((bm25_id_map[doc_idx], score))

        rrf_scores: dict[str, float] = {}
        k = 60
        for rank, (doc_id, _) in enumerate(unified_vector):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(unified_bm25):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. 상위 결과 수집 + Parent Lookup
        results = []
        seen_parents = set()

        for doc_id, rrf_score in sorted_rrf[:top_k * 2]:  # 중복 Parent 제거 후 top_k
            if doc_id not in id_to_data:
                continue

            data = id_to_data[doc_id]
            parent_id = data["metadata"].get("parent_id", "")

            # 같은 Parent에서 나온 Child 중복 방지
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            # Parent 원본 텍스트 가져오기
            parent_content = data["content"]  # fallback
            if parent_id:
                try:
                    parent_data = parents_col.get(ids=[parent_id])
                    if parent_data["documents"]:
                        parent_content = parent_data["documents"][0]
                except Exception as e:
                    print(f"  [Retriever] Parent 조회 실패 (parent_id={parent_id}): {e}")

            results.append(RetrievalResult(
                content=data["content"],
                parent_content=parent_content,
                metadata=data["metadata"],
                distance=data["distance"],
                rrf_score=rrf_score,
            ))

            if len(results) >= top_k:
                break

        # 5. Optional LLM Reranking
        if use_reranking and self.llm and results:
            results = self._llm_rerank(query, results)

        return results

    def _build_bm25_index(self, children_col):
        """Children 컬렉션으로 BM25 인덱스를 구축한다."""
        try:
            count = children_col.count()
            if count == 0:
                return
            all_data = children_col.get(limit=count)
            documents = []
            for i, doc in enumerate(all_data["documents"]):
                meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
                documents.append({
                    "content": doc,
                    "keywords": meta.get("keywords", ""),
                })
            self.bm25.index(documents)
            self._bm25_indexed = True
        except Exception as e:
            print(f"  [Retriever] BM25 인덱스 구축 실패: {e}")

    def _llm_rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """LLM으로 검색 결과의 관련도를 재정렬한다."""
        docs_text = "\n".join(
            f"[문서 {i+1}] {r.parent_content[:300]}"
            for i, r in enumerate(results)
        )

        prompt = f"""다음 문서들을 사용자 질문과의 관련도 순으로 정렬하세요.
각 문서 번호를 HIGH, MEDIUM, LOW로 평가하세요.

## 사용자 질문
{query}

## 문서 목록
{docs_text}

## 출력 형식 (JSON)
[{{"doc": 1, "relevance": "HIGH"}}, {{"doc": 2, "relevance": "LOW"}}]
JSON만 출력하세요."""

        try:
            response = self.llm.chat(messages=[
                {"role": "user", "content": prompt}
            ])
            text = response.content.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]
            rankings = json.loads(text)

            relevance_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            for ranking in rankings:
                idx = ranking.get("doc", 0) - 1
                if 0 <= idx < len(results):
                    results[idx].rerank_score = ranking.get("relevance", "MEDIUM")

            results.sort(key=lambda r: relevance_order.get(r.rerank_score, 1))
        except Exception as e:
            print(f"  [Retriever] LLM Reranking 실패 (RRF 순서 유지): {e}")

        return results
