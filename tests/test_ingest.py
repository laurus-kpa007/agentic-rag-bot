"""Advanced 문서 인제스트 파이프라인 테스트

Parent-Child Chunking + Contextual Headers를 검증한다.
"""

import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
from src.vectorstore.ingest import (
    chunk_text,
    split_into_chunks,
    extract_title,
    extract_keywords,
    make_contextual_header,
    ingest_documents,
    PARENT_CHUNK_SIZE,
    PARENT_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
)


def _make_dynamic_embedder():
    """입력 개수에 맞춰 임베딩을 동적으로 반환하는 Mock."""
    mock = MagicMock()
    mock.encode.side_effect = lambda texts: np.array([[0.1] * 384] * len(texts))
    return mock


class TestSplitIntoChunks:
    def test_short_text(self):
        """짧은 텍스트 → 청크 1개."""
        chunks = split_into_chunks("짧은 텍스트입니다.", 200, 30)
        assert len(chunks) == 1

    def test_long_text_chunking(self):
        """긴 텍스트 → 여러 청크로 분할."""
        text = "A" * 1000
        chunks = split_into_chunks(text, 200, 30)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_overlap(self):
        """청크 간 오버랩 확인."""
        text = "A" * 600
        chunks = split_into_chunks(text, 200, 30)
        if len(chunks) >= 2:
            assert chunks[0][-30:] == chunks[1][:30]

    def test_empty_text(self):
        chunks = split_into_chunks("", 200, 30)
        assert chunks == []

    def test_whitespace_only_filtered(self):
        """공백만 있는 청크는 필터링."""
        text = "내용" + " " * 400 + "내용2"
        chunks = split_into_chunks(text, 200, 30)
        for chunk in chunks:
            assert chunk.strip()


class TestChunkTextBackcompat:
    """chunk_text 함수 호환성 테스트 (기존 API 유지 확인)."""

    def test_short_text(self):
        text = "짧은 텍스트입니다."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text(self):
        text = "A" * 1200
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []


class TestExtractTitle:
    def test_markdown_header(self):
        text = "# 온보딩 가이드\n\n내용입니다."
        assert extract_title(text, "guide.md") == "온보딩 가이드"

    def test_no_header_uses_first_line(self):
        text = "첫 번째 줄 텍스트\n두 번째 줄"
        result = extract_title(text, "doc.txt")
        assert "첫 번째 줄" in result

    def test_empty_text_uses_filename(self):
        assert extract_title("", "fallback.txt") == "fallback.txt"


class TestExtractKeywords:
    def test_extracts_words(self):
        text = "연차 휴가 신청은 HR 포털에서 가능합니다"
        keywords = extract_keywords(text)
        assert "연차" in keywords
        assert "휴가" in keywords
        assert "HR" in keywords or "hr" in [k.lower() for k in keywords]

    def test_filters_stopwords(self):
        text = "있다 없다 하는 되는 것이 수가"
        keywords = extract_keywords(text)
        assert len(keywords) == 0

    def test_max_20_keywords(self):
        text = " ".join([f"키워드{i}" for i in range(50)])
        keywords = extract_keywords(text)
        assert len(keywords) <= 20


class TestContextualHeader:
    def test_header_format(self):
        header = make_contextual_header("policy.md", "휴가 정책", 0, 3)
        assert "[출처: policy.md" in header
        assert "휴가 정책" in header
        assert "섹션 1/3" in header


class TestIngestDocuments:
    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_ingest_creates_two_collections(self, mock_st_class):
        """인제스트 시 children + parents 2개 컬렉션 생성."""
        mock_st_class.return_value = _make_dynamic_embedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = os.path.join(tmpdir, "documents")
            chroma_dir = os.path.join(tmpdir, "chroma")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "policy.txt"), "w") as f:
                f.write("휴가 정책: 연차는 연 15일이며 HR 포털에서 신청합니다." * 5)

            count = ingest_documents(docs_dir=docs_dir, chroma_dir=chroma_dir)
            assert count >= 1

            # 컬렉션 존재 확인
            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            children = client.get_collection("children")
            parents = client.get_collection("parents")
            assert children.count() >= 1
            assert parents.count() >= 1

    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_child_has_parent_id(self, mock_st_class):
        """Child 메타데이터에 parent_id가 존재."""
        mock_st_class.return_value = _make_dynamic_embedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = os.path.join(tmpdir, "documents")
            chroma_dir = os.path.join(tmpdir, "chroma")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "test.txt"), "w") as f:
                f.write("테스트 문서 내용입니다. " * 30)

            ingest_documents(docs_dir=docs_dir, chroma_dir=chroma_dir)

            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            children = client.get_collection("children")
            data = children.get(limit=1)
            assert "parent_id" in data["metadatas"][0]

    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_child_has_contextual_header(self, mock_st_class):
        """Child 문서에 컨텍스트 헤더가 포함."""
        mock_st_class.return_value = _make_dynamic_embedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = os.path.join(tmpdir, "documents")
            chroma_dir = os.path.join(tmpdir, "chroma")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "guide.md"), "w") as f:
                f.write("# 온보딩 가이드\n\n신입사원은 첫 주에 OJT를 진행합니다. " * 10)

            ingest_documents(docs_dir=docs_dir, chroma_dir=chroma_dir)

            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            children = client.get_collection("children")
            data = children.get(limit=1)
            assert "[출처:" in data["documents"][0]

    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_ingest_md_files(self, mock_st_class):
        """markdown 파일 인제스트."""
        mock_st_class.return_value = _make_dynamic_embedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = os.path.join(tmpdir, "documents")
            chroma_dir = os.path.join(tmpdir, "chroma")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "guide.md"), "w") as f:
                f.write("# 온보딩 가이드\n\n신입사원은 첫 주에 OJT를 진행합니다.")

            count = ingest_documents(docs_dir=docs_dir, chroma_dir=chroma_dir)
            assert count >= 1

    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_ingest_empty_directory(self, mock_st_class):
        """빈 디렉토리 → 0개 인제스트."""
        mock_embedder = MagicMock()
        mock_st_class.return_value = mock_embedder

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = os.path.join(tmpdir, "empty_docs")
            chroma_dir = os.path.join(tmpdir, "chroma")
            os.makedirs(docs_dir)

            count = ingest_documents(docs_dir=docs_dir, chroma_dir=chroma_dir)
            assert count == 0

    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_child_has_keywords_metadata(self, mock_st_class):
        """Child 메타데이터에 keywords가 포함."""
        mock_st_class.return_value = _make_dynamic_embedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = os.path.join(tmpdir, "documents")
            chroma_dir = os.path.join(tmpdir, "chroma")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "policy.txt"), "w") as f:
                f.write("연차 휴가 정책 안내: 모든 직원에게 연 15일의 연차가 부여됩니다.")

            ingest_documents(docs_dir=docs_dir, chroma_dir=chroma_dir)

            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            children = client.get_collection("children")
            data = children.get(limit=1)
            assert "keywords" in data["metadatas"][0]
