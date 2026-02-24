# Simple Agentic RAG - API 및 데이터 흐름 문서

## 1. 외부 API 인터페이스

### 1.1 사용하는 서비스 목록

```mermaid
graph LR
    subgraph "Simple Agentic RAG"
        App["Application"]
    end

    subgraph "로컬 LLM"
        Ollama["Ollama<br/>(gemma3:12b)"]
    end

    subgraph "MCP Servers"
        VectorMCP["vector-search<br/>MCP Server"]
        WebMCP["web-search<br/>MCP Server"]
    end

    subgraph "로컬 서비스"
        Chroma["ChromaDB<br/>(로컬 벡터 DB)"]
        STModel["SentenceTransformers<br/>(로컬 임베딩)"]
    end

    App -->|"POST /api/chat"| Ollama
    App -->|"stdio JSON-RPC"| VectorMCP
    App -->|"stdio JSON-RPC"| WebMCP
    VectorMCP -->|"query()"| Chroma
    VectorMCP -->|"encode()"| STModel

    style Ollama fill:#4CAF50,color:#fff
    style VectorMCP fill:#2196F3,color:#fff
    style WebMCP fill:#9C27B0,color:#fff
    style Chroma fill:#FF9800,color:#fff
```

### 1.2 Ollama API 호출 패턴

시스템 전체에서 Ollama API는 **4가지 역할**로 호출된다. 모든 호출은 `OllamaAdapter.chat()`을 통해 통일된 인터페이스로 수행된다.

```mermaid
graph TD
    subgraph "Ollama API 호출 유형 (POST /api/chat)"
        direction TB

        Call1["① Router 호출<br/>도구: 없음"]
        Call2["② Agent 호출<br/>도구: MCP에서 수집한 전체 도구"]
        Call3["③ Grader 호출<br/>도구: 없음"]
        Call4["④ Rewriter 호출<br/>도구: 없음"]
    end

    Call1 -->|"출력"| R1["INTERNAL_SEARCH | WEB_SEARCH | CHITCHAT"]
    Call2 -->|"출력"| R2["tool_calls 또는 text 응답"]
    Call3 -->|"출력"| R3["PASS | FAIL"]
    Call4 -->|"출력"| R4["재작성된 검색 쿼리"]

    style Call1 fill:#2196F3,color:#fff
    style Call2 fill:#4CAF50,color:#fff
    style Call3 fill:#FF9800,color:#fff
    style Call4 fill:#F44336,color:#fff
```

### 1.3 리소스 사용 분석

로컬 Ollama를 사용하므로 API 비용은 **무료**이다. 대신 로컬 GPU 리소스를 사용한다.

| 호출 유형 | 모델 | 입력 토큰 (예상) | 출력 토큰 (예상) | 호출 빈도 |
|-----------|------|------------------|------------------|-----------|
| Router | gemma3:12b | ~200 | ~5 | 매 질문 |
| Agent (검색 포함) | gemma3:12b | ~800 | ~500 | 검색 필요 시 |
| Agent (직접 답변) | gemma3:12b | ~300 | ~300 | CHITCHAT 시 |
| Grader | gemma3:12b | ~1000 | ~3 | 검색 후 |
| Rewriter | gemma3:12b | ~150 | ~30 | Grader FAIL 시 |

**1회 질문당 최대 LLM 호출 수**: 4회 (Router → Agent → Grader → Rewriter+재검색)
**1회 질문당 최소 LLM 호출 수**: 2회 (Router → Direct Answer)

---

## 2. 데이터 흐름 상세

### 2.1 전체 데이터 흐름 (Happy Path)

사내 문서 검색이 성공적으로 완료되는 이상적인 경로이다.

```mermaid
sequenceDiagram
    actor User as 사용자
    participant Main as main.py
    participant Router as Router
    participant Agent as Agent Core
    participant LLM as Ollama (gemma3:12b)
    participant MCP as MCP Client
    participant Server as MCP Server (vector)
    participant Grader as Grader

    User->>Main: "휴가 신청 방법 알려줘"

    rect rgb(200, 220, 255)
        Note over Main,Router: Phase 2: 라우팅
        Main->>Router: classify("휴가 신청 방법 알려줘")
        Router->>LLM: chat(system=ROUTER_PROMPT)
        LLM-->>Router: "INTERNAL_SEARCH"
        Router-->>Main: INTERNAL_SEARCH
    end

    rect rgb(200, 255, 200)
        Note over Main,Server: Phase 1: Tool Calling (MCP)
        Main->>Agent: run(messages)
        Agent->>LLM: chat(messages, tools=MCP 도구 목록)
        LLM-->>Agent: tool_calls: search_vector_db(query="휴가 신청")
        Agent->>MCP: call_tool("vector-search__search_vector_db", args)
        MCP->>Server: JSON-RPC tools/call
        Server-->>MCP: [문서1, 문서2, 문서3]
        MCP-->>Agent: 검색 결과
        Agent->>LLM: 도구 결과 전달
        LLM-->>Agent: "휴가 신청은 다음과 같이..."
    end

    rect rgb(255, 240, 200)
        Note over Main,Grader: Phase 3: 평가
        Main->>Grader: evaluate(query, documents)
        Grader->>LLM: chat(system=GRADER_PROMPT)
        LLM-->>Grader: "PASS"
        Grader-->>Main: PASS
    end

    Main-->>User: "휴가 신청은 다음과 같이..."
```

### 2.2 재검색 흐름 (Grader FAIL Path)

검색 결과가 부적절하여 쿼리를 재작성하고 재검색하는 경로이다.

```mermaid
sequenceDiagram
    actor User as 사용자
    participant Main as main.py
    participant Agent as Agent Core
    participant LLM as Ollama (gemma3:12b)
    participant MCP as MCP Client
    participant Grader as Grader
    participant Rewriter as Rewriter

    User->>Main: "신입사원 출퇴근 규정이 뭐야?"

    Note over Main: Router → INTERNAL_SEARCH (생략)

    rect rgb(255, 200, 200)
        Note over Main,MCP: 1차 검색 (결과 부족)
        Main->>Agent: run(messages)
        Agent->>LLM: chat(messages, tools)
        LLM-->>Agent: tool_calls: search_vector_db
        Agent->>MCP: call_tool(...)
        MCP-->>Agent: [관련 없는 문서들]
        Agent->>LLM: 도구 결과 전달
        LLM-->>Agent: 부정확한 답변
    end

    rect rgb(255, 240, 200)
        Note over Main,Grader: 평가: FAIL
        Main->>Grader: evaluate(query, documents)
        Grader->>LLM: chat(system=GRADER_PROMPT)
        LLM-->>Grader: "FAIL"
        Grader-->>Main: "FAIL"
    end

    rect rgb(200, 220, 255)
        Note over Main,Rewriter: 쿼리 재작성
        Main->>Rewriter: rewrite("신입사원 출퇴근 규정이 뭐야?")
        Rewriter->>LLM: chat(system=REWRITER_PROMPT)
        LLM-->>Rewriter: "신규 입사자 근무시간 출퇴근 복무 규정"
        Rewriter-->>Main: 재작성된 쿼리
    end

    rect rgb(200, 255, 200)
        Note over Main,MCP: 2차 검색 (재작성 쿼리)
        Main->>Agent: run(재작성된 쿼리)
        Agent->>LLM: chat(messages, tools)
        LLM-->>Agent: tool_calls: search_vector_db
        Agent->>MCP: call_tool(...)
        MCP-->>Agent: [관련 문서 발견!]
        Agent->>LLM: 도구 결과 전달
        LLM-->>Agent: 정확한 답변
    end

    Main-->>User: "신입사원 근무시간 규정은..."
```

### 2.3 단순 대화 흐름 (CHITCHAT Path)

검색이 필요 없는 일상 대화 처리 경로이다.

```mermaid
sequenceDiagram
    actor User as 사용자
    participant Main as main.py
    participant Router as Router
    participant LLM as Ollama (gemma3:12b)

    User->>Main: "안녕하세요! 오늘 기분이 좋아요"

    Main->>Router: classify("안녕하세요! 오늘 기분이 좋아요")
    Router->>LLM: chat(system=ROUTER_PROMPT)
    LLM-->>Router: "CHITCHAT"
    Router-->>Main: CHITCHAT

    Note over Main: 검색 도구 없이 LLM 직접 호출

    Main->>LLM: chat(도구 없음)
    LLM-->>Main: "안녕하세요! 좋은 하루 보내고 계시군요..."

    Main-->>User: "안녕하세요! 좋은 하루 보내고 계시군요..."

    Note over User,LLM: LLM 호출: 총 2회 (로컬, 무료)<br/>(Router 1회 + 답변 생성 1회)
```

---

## 3. 내부 데이터 구조

### 3.1 주요 데이터 모델

```mermaid
classDiagram
    class UserQuery {
        +text: str
        +timestamp: datetime
    }

    class RouterResult {
        +route: str
        +raw_response: str
    }

    class SearchResult {
        +documents: list~Document~
        +query_used: str
        +tool_name: str
    }

    class Document {
        +content: str
        +metadata: dict
        +distance: float
    }

    class GradeResult {
        +verdict: str
        +retry_count: int
    }

    class AgentResponse {
        +answer: str
        +documents: list~Document~
        +route: str
        +grade: str
        +was_rewritten: bool
    }

    UserQuery --> RouterResult: "분류"
    RouterResult --> SearchResult: "검색"
    SearchResult --> Document: "포함"
    SearchResult --> GradeResult: "평가"
    GradeResult --> AgentResponse: "최종 응답"
```

### 3.2 대화 히스토리 구조

```mermaid
graph LR
    subgraph "conversation_history (최대 20개 = 10턴)"
        M1["user: '안녕'"]
        M2["assistant: '안녕하세요!'"]
        M3["user: '휴가 규정 알려줘'"]
        M4["assistant: '휴가 규정은...'"]
        M5["..."]
        M6["user: 최신 질문"]
        M7["assistant: 최신 답변"]
    end

    subgraph "LLM에 전송되는 메시지"
        SYS["system: SYSTEM_PROMPT"]
        HIST["messages: conversation_history"]
        TOOLS["tools: MCP에서 수집한 도구 목록"]
    end

    M1 --> HIST
    M7 --> HIST
    SYS --> LLM["Ollama (gemma3:12b)"]
    HIST --> LLM
    TOOLS --> LLM

    style SYS fill:#9C27B0,color:#fff
    style HIST fill:#4CAF50,color:#fff
    style TOOLS fill:#FF9800,color:#fff
```

---

## 4. 상태 전이 다이어그램

### 4.1 전체 시스템 상태 머신

```mermaid
stateDiagram-v2
    [*] --> Idle: 시스템 시작

    Idle --> Routing: 사용자 질문 수신

    Routing --> DirectAnswer: CHITCHAT
    Routing --> Searching: INTERNAL_SEARCH / WEB_SEARCH
    Routing --> Searching: 라우터 오류 (폴백)

    Searching --> Grading: 검색 완료
    Searching --> ErrorState: 도구 실행 오류 (재시도 초과)

    Grading --> Generating: PASS
    Grading --> Rewriting: FAIL (retry < 1)
    Grading --> Generating: FAIL (retry >= 1) 또는 Grader 오류

    Rewriting --> Searching: 재작성된 쿼리로 재검색

    DirectAnswer --> Responding: 답변 생성 완료
    Generating --> Responding: 답변 생성 완료

    Responding --> Idle: 답변 출력 완료
    ErrorState --> Responding: 오류 메시지 생성

    state Searching {
        [*] --> CallLLM
        CallLLM --> CheckToolUse
        CheckToolUse --> ExecuteTool: tool_use 존재
        CheckToolUse --> ReturnResult: text만 존재
        ExecuteTool --> CallLLM: 결과 전달
        ReturnResult --> [*]
    }
```

### 4.2 도구 호출 횟수 제어 상태

```mermaid
stateDiagram-v2
    [*] --> ToolCall_0: 에이전트 루프 시작

    ToolCall_0 --> ToolCall_1: 1번째 도구 호출
    ToolCall_1 --> ToolCall_2: 2번째 도구 호출
    ToolCall_2 --> ToolCall_3: 3번째 도구 호출

    ToolCall_0 --> TextResponse: LLM이 직접 답변
    ToolCall_1 --> TextResponse: LLM이 직접 답변
    ToolCall_2 --> TextResponse: LLM이 직접 답변
    ToolCall_3 --> ForcedStop: MAX_TOOL_CALLS 도달

    TextResponse --> [*]: 정상 답변 반환
    ForcedStop --> [*]: 오류 메시지 반환

    note right of ToolCall_3
        MAX_TOOL_CALLS = 3
        무한 루프 방지
    end note
```

---

## 5. 인제스트 데이터 흐름

### 5.1 문서 → 벡터 변환 상세

```mermaid
flowchart TD
    subgraph "입력"
        File["원본 파일<br/>(company_policy.md)"]
    end

    subgraph "1단계: 파일 읽기"
        Raw["원본 텍스트<br/>'# 회사 정책\n\n## 1. 휴가 규정\n...'<br/>(전체 3,200자)"]
    end

    subgraph "2단계: 청크 분할"
        C1["청크 1<br/>(500자)"]
        C2["청크 2<br/>(500자)"]
        C3["청크 3<br/>(500자)"]
        C4["..."]
        C5["청크 N<br/>(≤500자)"]
    end

    subgraph "3단계: 임베딩"
        E1["벡터 1<br/>[0.12, -0.34, ...]<br/>(384차원)"]
        E2["벡터 2<br/>[0.56, 0.78, ...]<br/>(384차원)"]
        E3["벡터 3<br/>[-0.23, 0.45, ...]<br/>(384차원)"]
        E4["..."]
        E5["벡터 N"]
    end

    subgraph "4단계: 저장"
        DB["ChromaDB Collection<br/>'documents'"]
        Meta["메타데이터<br/>{source, chunk_index}"]
    end

    File --> Raw
    Raw --> C1 & C2 & C3 & C4 & C5

    C1 --> E1
    C2 --> E2
    C3 --> E3
    C5 --> E5

    E1 & E2 & E3 & E5 --> DB
    DB --- Meta

    style File fill:#607D8B,color:#fff
    style DB fill:#9C27B0,color:#fff
    style Meta fill:#FF9800,color:#fff
```

### 5.2 검색 시 데이터 흐름

```mermaid
flowchart LR
    Query["검색 쿼리<br/>'휴가 신청 방법'"]

    Query --> Embed["쿼리 임베딩<br/>[0.45, -0.12, ...]<br/>(384차원)"]

    Embed --> Search["코사인 유사도 검색<br/>(ChromaDB)"]

    Search --> Results["상위 3개 결과"]

    Results --> Doc1["문서 1<br/>유사도: 0.89<br/>source: policy.md"]
    Results --> Doc2["문서 2<br/>유사도: 0.76<br/>source: guide.md"]
    Results --> Doc3["문서 3<br/>유사도: 0.71<br/>source: faq.md"]

    Doc1 --> Filter{"유사도 ≥ 0.7?"}
    Doc2 --> Filter
    Doc3 --> Filter

    Filter -->|"통과"| Final["필터링된 결과<br/>Agent에게 전달"]

    style Query fill:#2196F3,color:#fff
    style Embed fill:#FF9800,color:#fff
    style Search fill:#9C27B0,color:#fff
    style Final fill:#4CAF50,color:#fff
```

---

## 6. 에러 흐름 및 재시도 메커니즘

### 6.1 LLM/MCP 호출 재시도 전략

```mermaid
flowchart TD
    Start["Ollama / MCP 호출 시작"]

    Start --> Try1["1차 시도"]

    Try1 -->|"성공"| Success["정상 처리"]
    Try1 -->|"실패"| Wait1["대기 1초"]

    Wait1 --> Try2["2차 시도"]

    Try2 -->|"성공"| Success
    Try2 -->|"실패"| Wait2["대기 2초"]

    Wait2 --> Try3["3차 시도 (최종)"]

    Try3 -->|"성공"| Success
    Try3 -->|"실패"| Fail["오류 반환"]

    Success --> Continue["다음 단계 진행"]
    Fail --> Fallback["폴백 처리"]

    style Success fill:#4CAF50,color:#fff
    style Fail fill:#F44336,color:#fff
    style Fallback fill:#FF9800,color:#fff
```

### 6.2 전체 에러 처리 매트릭스

| 컴포넌트 | 에러 유형 | 재시도 | 폴백 동작 |
|----------|----------|--------|-----------|
| **Router** | Ollama 오류 | 2회 | `INTERNAL_SEARCH` 기본값 사용 |
| **Router** | 파싱 오류 | 0회 | `INTERNAL_SEARCH` 기본값 사용 |
| **Agent** | Ollama 오류 | 2회 | 오류 메시지 반환 |
| **Agent** | MCP 도구 실행 오류 | 1회 | 도구 없이 직접 답변 시도 |
| **MCP Server** | 서버 프로세스 크래시 | 1회 | 해당 도구 비활성화, 오류 메시지 |
| **VectorDB** | 검색 오류 | 1회 | 빈 결과로 진행 |
| **WebSearch** | 외부 API 오류 | 2회 | "검색 결과를 가져올 수 없습니다" |
| **Grader** | Ollama 오류 | 1회 | `PASS` (안전 모드) |
| **Grader** | 파싱 오류 | 0회 | `PASS` (안전 모드) |
| **Rewriter** | Ollama 오류 | 1회 | 원본 쿼리 그대로 재검색 |

---

## 7. 성능 최적화 포인트

### 7.1 토큰 절약 전략

```mermaid
graph TD
    subgraph "토큰 절약 기법"
        A["Router에 경량 프롬프트 사용<br/>(max_tokens: 50)"]
        B["Grader에 최소 출력 설정<br/>(max_tokens: 10)"]
        C["대화 히스토리 10턴 제한"]
        D["검색 결과 상위 3개만 전달"]
        E["청크 크기 500자 제한"]
    end

    A --> Save["토큰 비용 절감"]
    B --> Save
    C --> Save
    D --> Save
    E --> Save

    style Save fill:#4CAF50,color:#fff
```

### 7.2 응답 속도 최적화

| 최적화 기법 | 설명 | 예상 효과 |
|-------------|------|-----------|
| CHITCHAT 조기 반환 | 검색 불필요 질문은 Router → 직접 답변 | API 호출 2회로 감소 |
| 경량 Router 프롬프트 | Router 응답 생성 최소화 | ~100ms 절약 |
| top_k=3 제한 | 검색 결과 수 제한 | 컨텍스트 토큰 감소 |
| 임베딩 모델 캐싱 | SentenceTransformer 싱글턴 | 모델 로딩 1회만 |
| ChromaDB 로컬 실행 | 네트워크 지연 없음 | ~50ms 절약 |
