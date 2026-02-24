"""문서 인제스트 파이프라인 테스트"""

import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
from src.vectorstore.ingest import chunk_text, ingest_documents


class TestChunkText:
    def test_short_text(self):
        """500자 이하 텍스트 → 청크 1개."""
        text = "짧은 텍스트입니다."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunking(self):
        """500자 초과 텍스트 → 여러 청크로 분할."""
        text = "A" * 1200
        chunks = chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_overlap(self):
        """청크 간 50자 오버랩 확인."""
        text = "A" * 1000
        chunks = chunk_text(text)
        if len(chunks) >= 2:
            assert chunks[0][-50:] == chunks[1][:50]

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_whitespace_only_chunks_filtered(self):
        """공백만 있는 청크는 필터링."""
        text = "내용" + " " * 600 + "내용2"
        chunks = chunk_text(text)
        for chunk in chunks:
            assert chunk.strip()


class TestIngestDocuments:
    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_ingest_txt_files(self, mock_st_class):
        """txt 파일 인제스트 정상 동작."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1] * 384])
        mock_st_class.return_value = mock_embedder

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = os.path.join(tmpdir, "documents")
            chroma_dir = os.path.join(tmpdir, "chroma")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "policy.txt"), "w") as f:
                f.write("휴가 정책: 연차는 연 15일이며 HR 포털에서 신청합니다.")

            count = ingest_documents(
                docs_dir=docs_dir,
                chroma_dir=chroma_dir,
            )
            assert count >= 1

    @patch("src.vectorstore.ingest.SentenceTransformer")
    def test_ingest_md_files(self, mock_st_class):
        """markdown 파일 인제스트."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1] * 384])
        mock_st_class.return_value = mock_embedder

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
