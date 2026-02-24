"""Advanced Retriever 단위 테스트

BM25, RRF, AdvancedRetriever를 검증한다.
"""

from unittest.mock import MagicMock, patch
from src.retriever import BM25, reciprocal_rank_fusion, AdvancedRetriever, RetrievalResult


class TestBM25:
    def test_index_and_search(self):
        """BM25 인덱싱 및 검색 기본 동작."""
        bm25 = BM25()
        docs = [
            {"content": "연차 휴가 신청 방법", "keywords": "연차 휴가"},
            {"content": "출장비 정산 절차", "keywords": "출장비 정산"},
            {"content": "신입사원 온보딩 가이드", "keywords": "온보딩 신입사원"},
        ]
        bm25.index(docs)

        results = bm25.search("휴가 신청", top_k=3)
        assert len(results) > 0
        # 첫 번째 결과가 "연차 휴가" 문서여야 함
        assert results[0][0] == 0

    def test_search_returns_ranked_results(self):
        """BM25 결과가 스코어 내림차순."""
        bm25 = BM25()
        docs = [
            {"content": "파이썬 프로그래밍 가이드", "keywords": "파이썬"},
            {"content": "파이썬 웹 개발 파이썬 튜토리얼", "keywords": "파이썬 웹개발"},
        ]
        bm25.index(docs)

        results = bm25.search("파이썬", top_k=2)
        assert len(results) == 2
        # 스코어 내림차순
        assert results[0][1] >= results[1][1]

    def test_search_no_match(self):
        """매칭 없는 쿼리 → 스코어 0."""
        bm25 = BM25()
        docs = [{"content": "연차 휴가 정책", "keywords": "연차"}]
        bm25.index(docs)

        results = bm25.search("블록체인 기술", top_k=3)
        assert len(results) >= 1
        assert results[0][1] == 0.0

    def test_empty_corpus(self):
        """빈 코퍼스에서 검색 → 빈 결과."""
        bm25 = BM25()
        bm25.index([])
        results = bm25.search("테스트", top_k=3)
        assert results == []

    def test_tokenizer(self):
        """한국어 + 영어 토큰 추출."""
        bm25 = BM25()
        tokens = bm25._tokenize("Hello 세계 AI 테스트 a b")
        assert "hello" in tokens
        assert "세계" in tokens
        assert "테스트" in tokens
        # 1글자 필터링
        assert "a" not in tokens
        assert "b" not in tokens


class TestRRF:
    def test_single_source(self):
        """단일 소스 RRF 스코어."""
        vector = [(0, 0.9), (1, 0.7)]
        bm25 = []
        results = reciprocal_rank_fusion(vector, bm25)
        assert len(results) == 2
        assert results[0][0] == 0  # rank 1이 더 높은 RRF 스코어

    def test_both_sources_boost(self):
        """두 소스에 모두 등장하는 문서는 높은 RRF 스코어."""
        vector = [(0, 0.9), (1, 0.7), (2, 0.5)]
        bm25 = [(1, 3.0), (0, 2.0), (3, 1.0)]

        results = reciprocal_rank_fusion(vector, bm25)
        rrf_dict = dict(results)

        # doc 0: vector rank 1 + bm25 rank 2 → 두 소스에서 높은 순위
        # doc 1: vector rank 2 + bm25 rank 1 → 두 소스에서 높은 순위
        # 둘 다 양쪽에 있으므로 doc 2, doc 3보다 높아야 함
        assert rrf_dict[0] > rrf_dict.get(3, 0)
        assert rrf_dict[1] > rrf_dict.get(3, 0)

    def test_empty_results(self):
        """빈 결과 → 빈 RRF."""
        results = reciprocal_rank_fusion([], [])
        assert results == []


class TestAdvancedRetriever:
    def _make_mock_chroma(self):
        """Mock ChromaDB 클라이언트를 생성한다."""
        mock_client = MagicMock()

        # Children 컬렉션
        children_col = MagicMock()
        children_col.count.return_value = 3
        children_col.get.return_value = {
            "ids": ["doc_p0_c0", "doc_p0_c1", "doc_p1_c0"],
            "documents": [
                "[출처: doc.txt] 연차 휴가 신청은 HR 포털에서",
                "[출처: doc.txt] 출장비 정산은 ERP 시스템에서",
                "[출처: doc.txt] 신입사원 온보딩 첫 주 OJT",
            ],
            "metadatas": [
                {"parent_id": "doc_p0", "keywords": "연차 휴가 신청 HR"},
                {"parent_id": "doc_p0", "keywords": "출장비 정산 ERP"},
                {"parent_id": "doc_p1", "keywords": "신입사원 온보딩 OJT"},
            ],
        }
        children_col.query.return_value = {
            "ids": [["doc_p0_c0", "doc_p1_c0"]],
            "documents": [[
                "[출처: doc.txt] 연차 휴가 신청은 HR 포털에서",
                "[출처: doc.txt] 신입사원 온보딩 첫 주 OJT",
            ]],
            "metadatas": [[
                {"parent_id": "doc_p0", "keywords": "연차 휴가 신청 HR"},
                {"parent_id": "doc_p1", "keywords": "신입사원 온보딩 OJT"},
            ]],
            "distances": [[0.1, 0.4]],
        }

        # Parents 컬렉션
        parents_col = MagicMock()
        parents_col.get.return_value = {
            "ids": ["doc_p0"],
            "documents": ["연차 휴가 신청은 HR 포털에서 가능합니다. 연간 15일의 연차가 부여됩니다."],
            "metadatas": [{"source": "doc.txt"}],
        }

        def get_collection(name):
            if name == "children":
                return children_col
            elif name == "parents":
                return parents_col
            raise Exception(f"Collection {name} not found")

        mock_client.get_collection = get_collection
        return mock_client, children_col, parents_col

    def test_search_returns_results(self):
        """기본 검색이 결과를 반환."""
        mock_client, _, _ = self._make_mock_chroma()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        retriever = AdvancedRetriever(chroma_client=mock_client, embedder=mock_embedder)
        results = retriever.search("휴가 신청", top_k=2)

        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)

    def test_parent_lookup(self):
        """검색 결과에 Parent 컨텐츠가 포함."""
        mock_client, _, parents_col = self._make_mock_chroma()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        retriever = AdvancedRetriever(chroma_client=mock_client, embedder=mock_embedder)
        results = retriever.search("휴가 신청", top_k=2)

        # Parent 컨텐츠가 반환됨
        assert any("연간 15일" in r.parent_content for r in results)

    def test_dedup_by_parent(self):
        """같은 Parent의 Child가 중복 반환되지 않음."""
        mock_client, children_col, parents_col = self._make_mock_chroma()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        # 벡터 검색이 같은 parent의 child 2개를 반환
        children_col.query.return_value = {
            "ids": [["doc_p0_c0", "doc_p0_c1"]],
            "documents": [[
                "[출처: doc.txt] 연차 휴가 신청은 HR 포털에서",
                "[출처: doc.txt] 출장비 정산은 ERP 시스템에서",
            ]],
            "metadatas": [[
                {"parent_id": "doc_p0", "keywords": "연차 휴가"},
                {"parent_id": "doc_p0", "keywords": "출장비 정산"},
            ]],
            "distances": [[0.1, 0.2]],
        }

        retriever = AdvancedRetriever(chroma_client=mock_client, embedder=mock_embedder)
        results = retriever.search("휴가", top_k=5)

        parent_ids = [r.metadata.get("parent_id") for r in results]
        assert len(parent_ids) == len(set(parent_ids))  # 중복 없음

    def test_empty_collection(self):
        """빈 컬렉션 → 빈 결과."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")

        mock_embedder = MagicMock()
        retriever = AdvancedRetriever(chroma_client=mock_client, embedder=mock_embedder)
        results = retriever.search("테스트", top_k=3)

        assert results == []

    def test_rrf_score_in_results(self):
        """결과에 RRF 스코어가 포함."""
        mock_client, _, _ = self._make_mock_chroma()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        retriever = AdvancedRetriever(chroma_client=mock_client, embedder=mock_embedder)
        results = retriever.search("휴가", top_k=2)

        for r in results:
            assert r.rrf_score > 0
