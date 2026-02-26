"""Advanced 문서 인제스트 파이프라인

Parent-Child Chunking + Contextual Headers를 적용한다.
- Parent 청크 (800자): LLM에 전달할 충분한 컨텍스트
- Child 청크 (400자): 벡터 검색 정밀도 최적화
- Contextual Header: 각 Child에 문서명/섹션 정보 삽입
- 임베딩은 헤더 없는 원본 텍스트로, 저장은 헤더 포함 텍스트로 분리
"""

import glob
import math
import os
import re

import chromadb

from src.embedding import OllamaEmbedder

os.environ["ANONYMIZED_TELEMETRY"] = "False"


class _NoOpEmbeddingFunction(chromadb.EmbeddingFunction):
    """chromadb가 기본 임베딩 함수(onnx 다운로드)를 사용하지 않도록 하는 더미 EF."""

    def __init__(self):
        pass

    def __call__(self, input):
        return [[0.0] * 10 for _ in input]

# Parent-Child 청크 파라미터
PARENT_CHUNK_SIZE = 800
PARENT_OVERLAP = 100
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 50


def extract_title(text: str, filename: str) -> str:
    """문서에서 제목을 추출한다. Markdown 헤더 또는 첫 줄 사용."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line and not line.startswith("#"):
            return line[:50]
    return filename


def _find_sentence_boundary(text: str, chunk_start: int, chunk_end: int) -> int:
    """chunk_end 근처에서 문장 경계를 찾아 반환한다.

    chunk_end에서 역방향으로 청크 크기의 20% 이내를 탐색하여
    가장 가까운 문장 종결 위치를 반환한다.
    """
    tolerance = (chunk_end - chunk_start) // 5
    search_limit = max(chunk_start, chunk_end - tolerance)

    for i in range(chunk_end - 1, search_limit - 1, -1):
        if text[i] in '.!?\n':
            return i + 1

    return chunk_end


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """텍스트를 문장 경계를 고려하여 청크로 분할한다."""
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # 끝이 아니면 문장 경계로 스냅
        if end < text_len:
            end = _find_sentence_boundary(text, start, end)

        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """하위 호환용 청크 함수."""
    if not text or not text.strip():
        return []
    return split_into_chunks(text, chunk_size, overlap)


def make_contextual_header(filename: str, title: str, parent_idx: int, total_parents: int) -> str:
    """Child 청크 앞에 붙일 컨텍스트 헤더를 생성한다."""
    return f"[출처: {filename} | {title} | 섹션 {parent_idx + 1}/{total_parents}]\n"


def extract_keywords(text: str) -> list[str]:
    """텍스트에서 BM25 검색용 키워드를 추출한다."""
    # 한국어 + 영어 단어 추출 (2자 이상)
    words = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)
    # 빈도 기반 상위 키워드 (stopwords 간단 필터)
    stopwords = {"있다", "없다", "하는", "되는", "것이", "수가", "등의", "위한", "대한", "통해"}
    filtered = [w for w in words if w not in stopwords]
    return list(dict.fromkeys(filtered))[:20]  # 중복 제거, 최대 20개


def ingest_documents(
    docs_dir: str = "./data/documents",
    chroma_dir: str = "./data/chroma",
    embedding_model: str = "bona/bge-m3-korean:latest",
):
    """Advanced RAG 방식으로 문서를 인제스트한다.

    Parent-Child 이중 청크 + Contextual Headers + BM25 키워드 메타데이터.
    임베딩은 헤더 없는 원본 텍스트로, 저장은 헤더 포함 텍스트로 분리한다.
    """
    embedder = OllamaEmbedder(model=embedding_model)
    client = chromadb.PersistentClient(path=chroma_dir)

    # 기존 컬렉션 삭제
    for name in ["children", "parents", "documents"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    _noop_ef = _NoOpEmbeddingFunction()
    children_col = client.create_collection(
        name="children", metadata={"hnsw:space": "cosine"},
        embedding_function=_noop_ef,
    )
    parents_col = client.create_collection(
        name="parents", embedding_function=_noop_ef,
    )

    parent_chunks = []
    parent_metadatas = []
    parent_ids = []

    child_chunks_for_embedding = []  # 임베딩용 (헤더 없는 원본)
    child_chunks_for_storage = []    # 저장용 (헤더 포함)
    child_metadatas = []
    child_ids = []

    extensions = ["*.txt", "*.md"]

    for ext in extensions:
        for filepath in glob.glob(
            os.path.join(docs_dir, "**", ext), recursive=True
        ):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            filename = os.path.basename(filepath)
            title = extract_title(text, filename)

            # 1단계: Parent 청크 생성
            parents = split_into_chunks(text, PARENT_CHUNK_SIZE, PARENT_OVERLAP)
            total_parents = len(parents)

            for p_idx, parent_text in enumerate(parents):
                parent_id = f"{filename}_p{p_idx}"
                parent_keywords = extract_keywords(parent_text)

                parent_chunks.append(parent_text)
                parent_metadatas.append({
                    "source": filename,
                    "title": title,
                    "parent_index": p_idx,
                    "keywords": " ".join(parent_keywords),
                })
                parent_ids.append(parent_id)

                # 2단계: 각 Parent에서 Child 청크 생성
                children = split_into_chunks(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)

                for c_idx, child_text in enumerate(children):
                    header = make_contextual_header(filename, title, p_idx, total_parents)
                    enriched_child = header + child_text

                    child_id = f"{filename}_p{p_idx}_c{c_idx}"
                    child_keywords = extract_keywords(child_text)

                    child_chunks_for_embedding.append(child_text)
                    child_chunks_for_storage.append(enriched_child)
                    child_metadatas.append({
                        "source": filename,
                        "title": title,
                        "parent_id": parent_id,
                        "parent_index": p_idx,
                        "child_index": c_idx,
                        "keywords": " ".join(child_keywords),
                    })
                    child_ids.append(child_id)

    if not child_chunks_for_storage:
        print("인제스트할 문서가 없습니다.")
        return 0

    # 임베딩은 헤더 없는 원본 텍스트로 생성
    child_embeddings = embedder.encode(child_chunks_for_embedding).tolist()
    children_col.add(
        documents=child_chunks_for_storage,
        embeddings=child_embeddings,
        metadatas=child_metadatas,
        ids=child_ids,
    )

    # Parent 청크 저장 (임베딩 불필요 — 텍스트 저장용)
    parents_col.add(
        documents=parent_chunks,
        metadatas=parent_metadatas,
        ids=parent_ids,
    )

    print(f"인제스트 완료: Parent {len(parent_chunks)}개, Child {len(child_chunks_for_storage)}개")
    return len(child_chunks_for_storage)


if __name__ == "__main__":
    ingest_documents()
