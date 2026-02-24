# Simple Agentic RAG - 아키텍처 설계 문서

## 1. 시스템 전체 아키텍처

### 1.1 고수준 아키텍처 (High-Level Architecture)

```mermaid
graph TB
    User["👤 사용자"]

    subgraph "Simple Agentic RAG System"
        direction TB

        Router["🔀 Router<br/>(의도 분류기)"]

        subgraph "처리 경로"
            direction LR
            Path1["📄 사내 문서 검색<br/>(Vector DB RAG)"]
            Path2["🌐 외부 웹 검색<br/>(Web Search)"]
            Path3["💬 단순 대화<br/>(Direct LLM)"]
        end

        Planner["🧠 Query Planner<br/>(질의 분석 & 최적화)"]

        Agent["🤖 Agent Core<br/>(Tool Calling 엔진)"]

        subgraph "도구 (Tools)"
            direction LR
            Tool1["search_vector_db()"]
            Tool2["web_search()"]
        end

        Grader["📊 Grader<br/>(검색 결과 평가기)"]

        HITL["🙋 HITL<br/>(Human in the Loop)"]

        Generator["✍️ Generator<br/>(답변 생성기)"]
    end

    subgraph "외부 서비스"
        LLM["Claude API"]
        VectorDB["ChromaDB"]
        WebAPI["Web Search API"]
    end

    User -->|"질문"| Router
    Router -->|"사내 문서"| Path1
    Router -->|"웹 검색"| Path2
    Router -->|"일상 대화"| Path3

    Path1 --> Planner
    Path2 --> Planner
    Path3 --> Generator

    Planner -->|"최적화된 쿼리"| Agent

    Agent -->|"도구 호출"| Tool1
    Agent -->|"도구 호출"| Tool2

    Tool1 --> VectorDB
    Tool2 --> WebAPI

    Agent -->|"검색 결과"| Grader
    Grader -->|"Pass"| HITL
    Grader -->|"Fail"| Agent

    HITL -->|"승인"| Generator
    HITL -->|"수정/재검색"| Agent

    Agent --> LLM
    Router --> LLM
    Planner --> LLM
    Grader --> LLM
    Generator --> LLM

    Generator -->|"최종 답변"| User
```

> **참고**: Query Planner와 HITL의 상세 설계는 별도 문서를 참조하세요.
> - [Query Planner 설계](./query-planner.md)
> - [Human in the Loop 설계](./human-in-the-loop.md)

### 1.2 Phase별 아키텍처 진화

```mermaid
graph LR
    subgraph "Phase 1: Tool Calling"
        P1_User["사용자"] --> P1_Agent["Agent<br/>(Tool Calling)"]
        P1_Agent --> P1_Tools["Tools"]
        P1_Agent --> P1_Gen["답변 생성"]
    end

    subgraph "Phase 2: + Router"
        P2_User["사용자"] --> P2_Router["Router"]
        P2_Router --> P2_Agent["Agent<br/>(Tool Calling)"]
        P2_Router --> P2_Direct["Direct 답변"]
        P2_Agent --> P2_Tools["Tools"]
        P2_Agent --> P2_Gen["답변 생성"]
    end

    subgraph "Phase 2.5: + Query Planner"
        P25_User["사용자"] --> P25_Router["Router"]
        P25_Router --> P25_Planner["Query<br/>Planner"]
        P25_Router --> P25_Direct["Direct 답변"]
        P25_Planner --> P25_Agent["Agent"]
        P25_Agent --> P25_Gen["답변 생성"]
    end

    subgraph "Phase 3: + Feedback Loop"
        P3_User["사용자"] --> P3_Router["Router"]
        P3_Router --> P3_Planner["Query<br/>Planner"]
        P3_Router --> P3_Direct["Direct 답변"]
        P3_Planner --> P3_Agent["Agent"]
        P3_Agent --> P3_Grader["Grader"]
        P3_Grader -->|"Pass"| P3_Gen["답변 생성"]
        P3_Grader -->|"Fail"| P3_Agent
    end

    subgraph "Phase 4: + HITL"
        P4_User["사용자"] --> P4_Router["Router"]
        P4_Router --> P4_Planner["Query<br/>Planner"]
        P4_Router --> P4_Direct["Direct 답변"]
        P4_Planner --> P4_Agent["Agent"]
        P4_Agent --> P4_Grader["Grader"]
        P4_Grader --> P4_HITL["🙋 HITL"]
        P4_HITL --> P4_Gen["답변 생성"]
    end

    style P1_Agent fill:#4CAF50,color:#fff
    style P2_Router fill:#2196F3,color:#fff
    style P25_Planner fill:#E91E63,color:#fff
    style P3_Grader fill:#FF9800,color:#fff
    style P4_HITL fill:#9C27B0,color:#fff
```

---

## 2. 핵심 컴포넌트 상세 설계

### 2.1 Agent Core (에이전트 코어)

에이전트의 핵심 루프를 담당하는 중앙 컴포넌트이다. LLM의 네이티브 Tool Calling을 통해 도구 호출 여부를 자율적으로 판단한다.

```mermaid
stateDiagram-v2
    [*] --> ReceiveQuery: 사용자 질문 수신

    ReceiveQuery --> CallLLM: LLM에게 질문 + 도구 정의 전달

    CallLLM --> CheckResponse: LLM 응답 확인

    CheckResponse --> ExecuteTool: tool_use 블록 존재
    CheckResponse --> ReturnAnswer: text 블록만 존재 (도구 불필요)

    ExecuteTool --> CallLLM: 도구 실행 결과를 LLM에게 재전달

    ReturnAnswer --> [*]: 최종 답변 반환

    note right of CallLLM
        LLM이 자율적으로
        도구 호출 여부를 판단
    end note

    note right of ExecuteTool
        search_vector_db() 또는
        web_search() 실행
    end note
```

**핵심 설계 원칙:**
- 최대 도구 호출 횟수: **3회**로 제한 (무한 루프 방지)
- 대화 히스토리: 최근 **10턴**만 유지 (토큰 절약)
- 도구 정의: JSON Schema 형식으로 LLM에게 전달

### 2.2 Router (라우터)

사용자 질문을 분석하여 최적의 처리 경로로 분기하는 게이트키퍼 역할을 한다.

```mermaid
flowchart TD
    Input["사용자 질문 입력"]

    Input --> RouterLLM["Router LLM 호출<br/>(경량 프롬프트)"]

    RouterLLM --> Decision{"분류 결과"}

    Decision -->|"INTERNAL_SEARCH"| InternalPath["사내 문서 벡터 검색<br/>search_vector_db()"]
    Decision -->|"WEB_SEARCH"| WebPath["외부 웹 검색<br/>web_search()"]
    Decision -->|"CHITCHAT"| ChitchatPath["LLM 직접 응답<br/>(검색 없이)"]

    InternalPath --> AgentCore["Agent Core"]
    WebPath --> AgentCore
    ChitchatPath --> DirectGen["Direct Generator"]

    AgentCore --> Response["최종 응답"]
    DirectGen --> Response

    style RouterLLM fill:#2196F3,color:#fff
    style Decision fill:#FF9800,color:#fff
    style InternalPath fill:#4CAF50,color:#fff
    style WebPath fill:#9C27B0,color:#fff
    style ChitchatPath fill:#607D8B,color:#fff
```

**라우팅 분류 기준:**

| 카테고리 | 트리거 조건 | 예시 |
|----------|------------|------|
| `INTERNAL_SEARCH` | 사내 문서, 정책, 가이드라인 관련 질문 | "휴가 신청 절차가 어떻게 돼?" |
| `WEB_SEARCH` | 최신 정보, 외부 데이터 필요 | "오늘 코스피 지수 알려줘" |
| `CHITCHAT` | 일반 인사, 잡담, 간단한 지식 질문 | "안녕하세요", "파이썬이 뭐야?" |

### 2.3 Query Planner (질의 분석기)

Router 이후, 검색 이전에 위치하여 사용자 질문을 벡터 검색에 최적화된 쿼리로 변환한다. 대화 맥락 해소, 핵심어 추출, 복합 질문 분해를 수행한다.

> **상세 설계**: [Query Planner 설계 문서](./query-planner.md) 참조

```mermaid
flowchart TD
    Input["원본 질문 + 대화 히스토리"]
    Input --> Planner["Query Planner LLM 호출"]
    Planner --> Output["QueryPlan 출력"]

    Output --> Intent["intent: 의도 요약"]
    Output --> Keywords["keywords: 핵심어 목록"]
    Output --> Queries["search_queries: 최적화 쿼리 (1~2개)"]
    Output --> Strategy["strategy: SINGLE / MULTI"]

    style Planner fill:#E91E63,color:#fff
```

**핵심 기능:**
- **맥락 해소**: "그거 다시 알려줘" → 이전 질문 맥락 복원
- **쿼리 최적화**: 구어체 → 키워드 중심 명사구 변환
- **복합 질문 분해**: 최대 2개 서브쿼리로 분리

### 2.4 Grader (검색 결과 평가기)

검색된 문서가 사용자 질문에 답하기에 충분한지 이진(Pass/Fail) 판단을 수행한다.

```mermaid
flowchart TD
    Input["검색 결과 + 원본 질문"]

    Input --> GraderLLM["Grader LLM 호출"]

    GraderLLM --> Evaluation{"평가 결과"}

    Evaluation -->|"PASS"| Generate["답변 생성<br/>(검색 결과 활용)"]
    Evaluation -->|"FAIL"| CheckRetry{"재시도 횟수 확인"}

    CheckRetry -->|"retry < 1"| Rewrite["쿼리 재작성<br/>(Rewriter)"]
    CheckRetry -->|"retry >= 1"| Fallback["폴백 응답 생성<br/>'정확한 정보를 찾지 못했습니다'"]

    Rewrite --> ReSearch["재검색 실행"]
    ReSearch --> GraderLLM

    Generate --> Output["최종 답변"]
    Fallback --> Output

    style GraderLLM fill:#FF9800,color:#fff
    style Evaluation fill:#F44336,color:#fff
    style Generate fill:#4CAF50,color:#fff
    style Rewrite fill:#2196F3,color:#fff
    style Fallback fill:#607D8B,color:#fff
```

**평가 기준 (프롬프트로 제어):**
- **PASS 조건**: 검색 결과가 질문의 핵심 키워드에 관련된 정보를 포함
- **FAIL 조건**: 검색 결과가 질문과 무관하거나 정보가 불충분

### 2.5 Human in the Loop (HITL)

에이전트의 신뢰도가 낮을 때 사람에게 판단을 위임하고, 답변 후 피드백을 수집하는 메커니즘이다.

> **상세 설계**: [Human in the Loop 설계 문서](./human-in-the-loop.md) 참조

```mermaid
flowchart LR
    Answer["에이전트 답변"] --> Confidence{"신뢰도 점수"}

    Confidence -->|"HIGH ≥0.8"| Auto["자동 전달"]
    Confidence -->|"MEDIUM 0.5~0.8"| Soft["경고 표시<br/>+ 자동 승인 옵션"]
    Confidence -->|"LOW <0.5"| Hard["필수 검토<br/>승인/수정/재검색/거부"]

    Auto --> Deliver["답변 전달"]
    Soft --> Deliver
    Hard -->|"사용자 결정"| Deliver

    Deliver --> Feedback["👍/👎 피드백 수집"]

    style Hard fill:#F44336,color:#fff
    style Soft fill:#FF9800,color:#fff
    style Auto fill:#4CAF50,color:#fff
```

**HITL 모드:**
- `auto`: 신뢰도 기반 자동 트리거 (기본값)
- `strict`: 모든 검색 답변에 필수 검토
- `off`: HITL 비활성화

### 2.6 도구(Tools) 설계

```mermaid
classDiagram
    class ToolInterface {
        <<interface>>
        +name: str
        +description: str
        +parameters: dict
        +execute(params: dict) dict
    }

    class VectorSearchTool {
        +name = "search_vector_db"
        +description = "사내 문서 벡터 DB 검색"
        +execute(query: str, top_k: int) list~Document~
    }

    class WebSearchTool {
        +name = "web_search"
        +description = "외부 웹 검색"
        +execute(query: str) list~SearchResult~
    }

    class Document {
        +content: str
        +metadata: dict
        +score: float
    }

    class SearchResult {
        +title: str
        +url: str
        +snippet: str
    }

    ToolInterface <|.. VectorSearchTool
    ToolInterface <|.. WebSearchTool
    VectorSearchTool --> Document
    WebSearchTool --> SearchResult
```

**Tool 정의 스키마 (Claude API 형식):**

```json
{
  "name": "search_vector_db",
  "description": "사내 문서 데이터베이스에서 관련 문서를 검색합니다. 사내 정책, 가이드라인, 매뉴얼 등에 대한 질문일 때 사용합니다.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "검색할 쿼리 문자열"
      },
      "top_k": {
        "type": "integer",
        "description": "반환할 최대 문서 수 (기본값: 3)",
        "default": 3
      }
    },
    "required": ["query"]
  }
}
```

---

## 3. 벡터 스토어 설계

### 3.1 문서 인제스트 파이프라인

```mermaid
flowchart LR
    subgraph "입력 소스"
        PDF["PDF 파일"]
        MD["Markdown 파일"]
        TXT["텍스트 파일"]
    end

    subgraph "전처리 파이프라인"
        Loader["Document Loader<br/>(파일 읽기)"]
        Splitter["Text Splitter<br/>(청크 분할)"]
        Embedder["Embedding Model<br/>(all-MiniLM-L6-v2)"]
    end

    subgraph "저장소"
        ChromaDB["ChromaDB<br/>(벡터 저장소)"]
    end

    PDF --> Loader
    MD --> Loader
    TXT --> Loader

    Loader --> Splitter
    Splitter --> Embedder
    Embedder --> ChromaDB

    style Loader fill:#4CAF50,color:#fff
    style Splitter fill:#2196F3,color:#fff
    style Embedder fill:#FF9800,color:#fff
    style ChromaDB fill:#9C27B0,color:#fff
```

### 3.2 청크 전략

| 파라미터 | 값 | 근거 |
|----------|------|------|
| `chunk_size` | 500자 | 한국어 기준 의미 단위 유지에 적합 |
| `chunk_overlap` | 50자 | 문맥 연속성 보장 |
| `separators` | `["\n\n", "\n", ". ", " "]` | 단락 → 줄바꿈 → 문장 → 공백 순으로 분할 |

### 3.3 검색 전략

```mermaid
sequenceDiagram
    participant Agent as Agent Core
    participant Embed as Embedding Model
    participant Chroma as ChromaDB

    Agent->>Embed: 쿼리 임베딩 요청
    Embed-->>Agent: 쿼리 벡터 반환

    Agent->>Chroma: similarity_search(query_vector, top_k=3)
    Chroma-->>Agent: 상위 3개 문서 + 유사도 점수 반환

    Note over Agent: score >= 0.7인 문서만 필터링<br/>(낮은 유사도 결과 제외)
```

---

## 4. 프롬프트 아키텍처

### 4.1 프롬프트 분리 전략

하나의 거대한 시스템 프롬프트 대신, 역할별로 프롬프트를 분리하여 각 단계의 성능을 최적화한다.

```mermaid
graph TD
    subgraph "프롬프트 구조"
        direction TB

        SysPrompt["시스템 프롬프트<br/>(agent.py)"]
        RouterPrompt["라우터 프롬프트<br/>(prompts/router.py)"]
        PlannerPrompt["플래너 프롬프트<br/>(prompts/planner.py)"]
        GraderPrompt["평가 프롬프트<br/>(prompts/grader.py)"]
        RewriterPrompt["재작성 프롬프트<br/>(prompts/rewriter.py)"]
        GeneratorPrompt["답변 생성 프롬프트<br/>(prompts/generator.py)"]
    end

    SysPrompt -->|"에이전트 전체 역할 정의"| Agent["Agent Core"]
    RouterPrompt -->|"의도 분류 지시"| Router["Router"]
    PlannerPrompt -->|"질의 분석 & 최적화 지시"| Planner["Query Planner"]
    GraderPrompt -->|"문서 평가 지시"| Grader["Grader"]
    RewriterPrompt -->|"쿼리 변환 지시"| Rewriter["Rewriter"]
    GeneratorPrompt -->|"답변 포맷 지시"| Generator["Generator"]

    style SysPrompt fill:#9C27B0,color:#fff
    style RouterPrompt fill:#2196F3,color:#fff
    style PlannerPrompt fill:#E91E63,color:#fff
    style GraderPrompt fill:#FF9800,color:#fff
    style RewriterPrompt fill:#F44336,color:#fff
    style GeneratorPrompt fill:#4CAF50,color:#fff
```

### 4.2 각 프롬프트의 역할

| 프롬프트 | 입력 | 출력 | 호출 빈도 |
|----------|------|------|-----------|
| **System** | 없음 (상시 적용) | 에이전트 행동 규칙 | 매 대화 |
| **Router** | 사용자 질문 | `INTERNAL_SEARCH` / `WEB_SEARCH` / `CHITCHAT` | 매 질문 |
| **Planner** | 질문 + 대화 히스토리 + 라우팅 결과 | QueryPlan (JSON: intent, keywords, queries, strategy) | 검색 필요 시 |
| **Grader** | 질문 + 검색 결과 | `PASS` / `FAIL` | 검색 발생 시 |
| **Rewriter** | 원본 질문 + 실패 사유 | 개선된 검색 쿼리 | Grader FAIL 시 |
| **Generator** | 질문 + (검색 결과) | 자연어 답변 | 매 답변 |

---

## 5. 에러 처리 및 폴백 전략

```mermaid
flowchart TD
    Start["요청 처리 시작"]

    Start --> TryRouter{"Router 호출 성공?"}

    TryRouter -->|"성공"| Route["경로 분기"]
    TryRouter -->|"실패"| FallbackRoute["기본값: INTERNAL_SEARCH"]

    Route --> TryTool{"도구 실행 성공?"}
    FallbackRoute --> TryTool

    TryTool -->|"성공"| TryGrade{"Grader 평가 성공?"}
    TryTool -->|"실패 (API 오류)"| RetryTool{"재시도 가능?<br/>(max 2회)"}

    RetryTool -->|"예"| TryTool
    RetryTool -->|"아니오"| ErrorResponse["오류 응답 반환"]

    TryGrade -->|"PASS"| GenerateAnswer["답변 생성"]
    TryGrade -->|"FAIL"| TryRewrite["쿼리 재작성 + 재검색"]
    TryGrade -->|"Grader 오류"| GenerateAnswer

    TryRewrite --> GenerateAnswer

    GenerateAnswer --> End["최종 응답"]
    ErrorResponse --> End

    style Start fill:#4CAF50,color:#fff
    style ErrorResponse fill:#F44336,color:#fff
    style End fill:#607D8B,color:#fff
```

### 폴백 규칙

1. **Router 실패 시**: `INTERNAL_SEARCH`를 기본값으로 사용
2. **도구 실행 실패 시**: 최대 2회 재시도 후 오류 응답
3. **Grader 실패 시**: 검색 결과를 그대로 사용하여 답변 생성 (안전 모드)
4. **LLM API 전체 장애 시**: "현재 서비스를 이용할 수 없습니다" 정적 응답

---

## 6. 보안 고려사항

| 영역 | 위협 | 대응 |
|------|------|------|
| **프롬프트 인젝션** | 사용자가 시스템 프롬프트를 조작 | 입력 검증, 시스템/사용자 프롬프트 분리 |
| **API 키 노출** | 환경 변수 유출 | `.env` 파일 사용, `.gitignore`에 포함 |
| **데이터 유출** | 벡터 DB 내 민감 정보 | 인제스트 시 PII 필터링 |
| **토큰 남용** | 악의적 대량 요청 | 요청 Rate Limiting 적용 |
