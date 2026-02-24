"""Vector Search MCP Server - 벡터 DB 검색 도구를 MCP로 제공

stdio를 통해 JSON-RPC 메시지를 주고받는 MCP 서버이다.
"""

import json
import os
import sys

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Lazy 초기화 (서버 시작 시가 아니라 첫 검색 호출 시 로딩)
_embedder = None
_chroma = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma

TOOLS = [
    {
        "name": "search_vector_db",
        "description": "사내 문서 데이터베이스에서 관련 문서를 검색합니다. 사내 정책, 가이드라인, 매뉴얼, 업무 절차에 대한 질문일 때 사용하세요.",
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
    try:
        col = _get_chroma().get_collection("documents")
    except Exception:
        return {
            "content": [{"type": "text", "text": json.dumps([], ensure_ascii=False)}]
        }

    query_embedding = _get_embedder().encode(query).tolist()
    results = col.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i] if results.get("distances") else 0.0
        docs.append(
            {
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "distance": distance,
            }
        )

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
