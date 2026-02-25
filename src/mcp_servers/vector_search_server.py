"""Vector Search MCP Server - Advanced RAG 검색 도구를 MCP로 제공

stdio를 통해 JSON-RPC 메시지를 주고받는 MCP 서버이다.
Hybrid Search (Vector + BM25) + RRF + Parent Lookup을 수행한다.
"""

import json
import os
import ssl
import sys

# SSL 인증서 검증 비활성화 (사내 프록시/인증서 이슈 대응)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["ANONYMIZED_TELEMETRY"] = "False"

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bona/bge-m3-korean:latest")

# Lazy 초기화 (서버 시작 시가 아니라 첫 검색 호출 시 로딩)
_embedder = None
_chroma = None
_retriever = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from src.embedding import OllamaEmbedder
        _embedder = OllamaEmbedder(model=EMBEDDING_MODEL)
    return _embedder


def _get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.retriever import AdvancedRetriever
        _retriever = AdvancedRetriever(
            chroma_client=_get_chroma(),
            embedder=_get_embedder(),
        )
    return _retriever

TOOLS = [
    {
        "name": "search_vector_db",
        "description": "사내 문서 데이터베이스에서 관련 문서를 검색합니다. Hybrid Search(벡터+BM25) + RRF 스코어 퓨전으로 정확한 결과를 반환합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 쿼리 문자열"},
                "top_k": {
                    "type": "integer",
                    "description": "반환할 최대 문서 수",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    }
]


def search(query: str, top_k: int = 3) -> dict:
    retriever = _get_retriever()
    results = retriever.search(query=query, top_k=top_k)

    if not results:
        return {
            "content": [{"type": "text", "text": json.dumps([], ensure_ascii=False)}]
        }

    docs = []
    for r in results:
        docs.append({
            "content": r.parent_content,
            "metadata": r.metadata,
            "distance": r.distance,
        })

    return {
        "content": [{"type": "text", "text": json.dumps(docs, ensure_ascii=False)}]
    }


def handle_request(req: dict) -> dict:
    method = req.get("method", "")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "vector-search", "version": "1.0.0"},
        }
    elif method == "notifications/initialized":
        return {}
    elif method == "tools/list":
        return {"tools": TOOLS}
    elif method == "tools/call":
        params = req.get("params", {})
        args = params.get("arguments", {})
        return search(args.get("query", ""), args.get("top_k", 3))
    else:
        return {}


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            result = handle_request(req)
            response = {
                "jsonrpc": "2.0",
                "id": req.get("id"),
                "result": result,
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            error_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)},
            }
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
