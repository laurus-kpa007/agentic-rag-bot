"""Web Search MCP Server - 외부 웹 검색 도구를 MCP로 제공

stdio를 통해 JSON-RPC 메시지를 주고받는 MCP 서버이다.
DuckDuckGo 검색을 사용하여 외부 API 키 없이 동작한다.
"""

import json
import sys
import urllib.parse
import urllib.request


TOOLS = [
    {
        "name": "web_search",
        "description": "외부 웹에서 최신 정보를 검색합니다. 실시간 데이터, 뉴스, 최신 기술 트렌드 등 사내 문서에 없는 정보가 필요할 때 사용하세요.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 쿼리 문자열"}
            },
            "required": ["query"],
        },
    }
]


def web_search(query: str) -> dict:
    """DuckDuckGo Instant Answer API를 사용한 간단한 웹 검색."""
    try:
        encoded = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1})
        url = f"https://api.duckduckgo.com/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "AgenticRAGBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []
        # Abstract 결과
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", ""),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"][:500],
            })
        # RelatedTopics 결과
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")[:500],
                })

        if not results:
            results.append({
                "title": "검색 결과 없음",
                "url": "",
                "snippet": f"'{query}'에 대한 검색 결과를 찾지 못했습니다.",
            })

        return {
            "content": [
                {"type": "text", "text": json.dumps(results, ensure_ascii=False)}
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        [{"title": "검색 오류", "url": "", "snippet": str(e)}],
                        ensure_ascii=False,
                    ),
                }
            ]
        }


def handle_request(req: dict) -> dict:
    method = req.get("method", "")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "web-search", "version": "1.0.0"},
        }
    elif method == "notifications/initialized":
        return {}
    elif method == "tools/list":
        return {"tools": TOOLS}
    elif method == "tools/call":
        params = req.get("params", {})
        args = params.get("arguments", {})
        return web_search(args.get("query", ""))
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
