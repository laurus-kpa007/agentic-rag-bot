"""문서 인제스트 파이프라인

data/documents/ 폴더의 파일을 읽어 ChromaDB에 벡터로 저장한다.
"""

import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str) -> list[str]:
    """텍스트를 고정 크기 청크로 분할한다."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def ingest_documents(
    docs_dir: str = "./data/documents",
    chroma_dir: str = "./data/chroma",
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """문서 디렉토리의 모든 파일을 벡터 DB에 인제스트한다."""
    embedder = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=chroma_dir)

    # 기존 컬렉션이 있으면 삭제 후 재생성
    try:
        client.delete_collection("documents")
    except Exception:
        pass

    collection = client.create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    all_metadatas = []
    all_ids = []

    extensions = ["*.txt", "*.md"]

    for ext in extensions:
        for filepath in glob.glob(
            os.path.join(docs_dir, "**", ext), recursive=True
        ):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)
            filename = os.path.basename(filepath)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": filename, "chunk_index": i})
                all_ids.append(f"{filename}_{i}")

    if not all_chunks:
        print("인제스트할 문서가 없습니다.")
        return 0

    # 배치 임베딩 및 저장
    embeddings = embedder.encode(all_chunks).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print(f"총 {len(all_chunks)}개 청크를 인제스트했습니다.")
    return len(all_chunks)


if __name__ == "__main__":
    ingest_documents()
