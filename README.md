# Simple Agentic RAG Bot

로컬 Ollama LLM과 MCP 도구 시스템을 활용한 자율형 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 아키텍처

```
사용자 질문
    ↓
[Router] → CHITCHAT → 직접 답변
    ↓
[Query Planner] → 쿼리 최적화
    ↓
[Agent Core] → MCP 도구 호출 (벡터 검색 / 웹 검색)
    ↓
[Grader] → PASS → [HITL] → 답변 전달
    ↓ FAIL
[Rewriter] → 재검색 (1회)
```

### 핵심 컴포넌트

| 컴포넌트 | 설명 | Phase |
|----------|------|-------|
| **Agent Core** | Ollama Tool Calling 루프 | 1 |
| **Router** | 질문 의도 분류 (INTERNAL_SEARCH / WEB_SEARCH / CHITCHAT) | 2 |
| **Query Planner** | 쿼리 최적화, 맥락 해소, 복합 질문 분해 | 2.5 |
| **Grader** | 검색 결과 관련성 평가 (PASS/FAIL) | 3 |
| **HITL** | 신뢰도 기반 사람 개입 + 피드백 수집 | 4 |

### 기술 스택

- **LLM**: Ollama + gemma3:12b (로컬, 무료)
- **도구 시스템**: MCP (Model Context Protocol) 플러그인 방식
- **벡터 DB**: ChromaDB (로컬)
- **임베딩**: sentence-transformers (all-MiniLM-L6-v2)

## 시작하기

### 1. 사전 준비

```bash
# Ollama 설치 및 모델 다운로드
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:12b
```

### 2. 설치

```bash
git clone <repo-url> && cd agentic-rag-bot
pip install -r requirements.txt
cp .env.example .env
```

### 3. 문서 인제스트

`data/documents/` 폴더에 검색할 문서(.txt, .md)를 넣고 인제스트합니다.

```bash
python -m src.vectorstore.ingest
```

### 4. 실행

```bash
python -m src.main
```

```
Simple Agentic RAG Bot 시작 중...
  [MCP] vector-search 연결 완료 (도구 1개)
  [MCP] web-search 연결 완료 (도구 1개)
준비 완료! (종료: quit)

[사용자] 휴가 신청 방법 알려줘
  [라우팅] INTERNAL_SEARCH
  [플래닝] 의도: 휴가 신청 방법 확인
  [플래닝] 검색어: ['휴가 신청 절차 방법']
  [평가] PASS

[봇] 휴가 신청은 HR 포털에서 가능합니다...
```

## 프로젝트 구조

```
agentic-rag-bot/
├── src/
│   ├── main.py                 # 진입점 (Phase 1~4 통합)
│   ├── agent.py                # Agent Core (Tool Calling 루프)
│   ├── llm_adapter.py          # OllamaAdapter (LLM 추상화)
│   ├── mcp_client.py           # MCP 클라이언트
│   ├── router.py               # Router (의도 분류)
│   ├── planner.py              # Query Planner (쿼리 최적화)
│   ├── grader.py               # Grader + QueryRewriter
│   ├── hitl.py                 # HITL + 피드백 수집
│   ├── config.py               # 설정 관리
│   ├── prompts/                # 역할별 분리된 프롬프트
│   │   ├── system.py
│   │   ├── router.py
│   │   ├── planner.py
│   │   ├── grader.py
│   │   ├── rewriter.py
│   │   └── generator.py
│   ├── mcp_servers/            # 내장 MCP 서버 (플러그인)
│   │   ├── vector_search_server.py
│   │   └── web_search_server.py
│   └── vectorstore/
│       └── ingest.py           # 문서 인제스트
├── tests/                      # 단위 + 통합 테스트 (69개)
├── docs/                       # 설계 문서
├── data/
│   ├── documents/              # 검색할 원본 문서
│   └── chroma/                 # ChromaDB 저장소 (자동 생성)
├── mcp_config.json             # MCP 서버 설정
├── requirements.txt
├── .env.example
└── .gitignore
```

## 설정

`.env` 파일로 설정을 관리합니다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `LLM_MODEL` | `gemma3:12b` | 사용할 LLM 모델 |
| `MCP_CONFIG_PATH` | `mcp_config.json` | MCP 서버 설정 파일 경로 |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB 저장 경로 |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 임베딩 모델 |
| `HITL_MODE` | `auto` | HITL 모드 (`auto`/`strict`/`off`) |

### 모델 교체

`.env`의 `LLM_MODEL`만 변경하면 됩니다.

| 모델 | VRAM | 한국어 | 특징 |
|------|------|--------|------|
| `gemma3:12b` | ~8GB | 양호 | 기본값 |
| `qwen2.5:14b` | ~10GB | 우수 | 한국어 우선 시 |
| `gemma3:27b` | ~18GB | 우수 | GPU 여유 있을 때 |

## MCP 도구 확장

새 도구를 추가하려면 `mcp_config.json`에 서버를 등록하면 됩니다. 코드 변경 **0줄**.

```json
{
  "mcpServers": {
    "vector-search": { "command": "python", "args": ["src/mcp_servers/vector_search_server.py"] },
    "web-search": { "command": "python", "args": ["src/mcp_servers/web_search_server.py"] },
    "slack": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-slack"],
      "env": { "SLACK_TOKEN": "${SLACK_TOKEN}" }
    }
  }
}
```

## 테스트

```bash
python -m pytest tests/ -v
```

**69개 테스트** (단위 + 통합):
- `test_llm_adapter.py` - LLM 어댑터
- `test_router.py` - 라우터 분류
- `test_planner.py` - 쿼리 플래너
- `test_grader.py` - 검색 결과 평가
- `test_agent.py` - 에이전트 코어
- `test_hitl.py` - HITL 신뢰도/피드백
- `test_mcp_servers.py` - MCP 서버 프로토콜
- `test_ingest.py` - 문서 인제스트
- `test_integration.py` - 전체 파이프라인 E2E

## 설계 문서

- [전략 문서](docs/strategy.md) - 구현 전략, 원칙, 단계별 계획
- [아키텍처](docs/architecture.md) - 시스템 구조, 컴포넌트 설계
- [구현 가이드](docs/implementation-guide.md) - 코드 스케치, 테스트 전략
- [API/데이터 흐름](docs/api-and-data-flow.md) - 시퀀스 다이어그램, 에러 처리
- [Query Planner](docs/query-planner.md) - 쿼리 분석기 설계
- [HITL](docs/human-in-the-loop.md) - 사람 개입 메커니즘
- [Ollama + MCP](docs/mcp-integration.md) - 로컬 LLM + 플러그인 도구 설계
