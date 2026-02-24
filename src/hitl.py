"""Human in the Loop - 사람 개입 및 피드백 수집

에이전트의 신뢰도가 낮을 때 사람에게 판단을 위임하고,
답변 후 피드백을 수집하는 메커니즘이다.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class HITLContext:
    query: str
    answer: str
    confidence: float
    documents: list[dict] = field(default_factory=list)
    route: str = ""
    search_queries: list[str] = field(default_factory=list)


@dataclass
class HITLDecision:
    action: str  # "approve" | "edit" | "retry" | "reject"
    edited_answer: str = ""
    new_query: str = ""


@dataclass
class Feedback:
    query: str
    answer: str
    rating: str  # "up" | "down"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ConfidenceCalculator:
    """가중 평균으로 신뢰도 점수를 계산한다."""

    WEIGHTS = {
        "vector_similarity": 0.3,
        "grader_pass": 0.3,
        "doc_count_ratio": 0.2,
        "no_retry": 0.2,
    }

    def calculate(
        self,
        grader_result: str = "PASS",
        vector_scores: list[float] | None = None,
        retry_count: int = 0,
        doc_count: int = 0,
        expected_docs: int = 3,
    ) -> float:
        scores = {}

        # 벡터 유사도 (distance가 작을수록 유사 → 1-distance로 변환)
        if vector_scores:
            avg_sim = sum(1 - min(s, 1.0) for s in vector_scores) / len(vector_scores)
            scores["vector_similarity"] = max(0.0, min(1.0, avg_sim))
        else:
            scores["vector_similarity"] = 0.5

        # Grader 결과
        scores["grader_pass"] = 1.0 if grader_result == "PASS" else 0.0

        # 검색 결과 수 비율
        if expected_docs > 0:
            scores["doc_count_ratio"] = min(1.0, doc_count / expected_docs)
        else:
            scores["doc_count_ratio"] = 0.5

        # 재시도 없음 보너스
        scores["no_retry"] = 1.0 if retry_count == 0 else 0.3

        # 가중 평균
        total = sum(
            scores[key] * self.WEIGHTS[key] for key in self.WEIGHTS
        )
        return round(total, 3)


class HITLManager:
    """Human in the Loop 관리자."""

    def __init__(self, mode: str = "auto"):
        self.mode = mode  # "auto" | "strict" | "off"
        self.calculator = ConfidenceCalculator()

    def should_intervene(self, confidence: float) -> str:
        """신뢰도에 따른 개입 수준을 결정한다."""
        if self.mode == "off":
            return "none"
        if self.mode == "strict":
            return "hard"

        # auto 모드
        if confidence >= 0.8:
            return "none"
        elif confidence >= 0.5:
            return "soft"
        else:
            return "hard"

    def request_review(self, context: HITLContext) -> HITLDecision:
        """사용자에게 검토를 요청하고 결정을 반환한다."""
        intervention = self.should_intervene(context.confidence)

        if intervention == "none":
            return HITLDecision(action="approve")

        # CLI에서 사용자 입력
        print(f"\n  [HITL] 신뢰도: {context.confidence:.1%}")
        if intervention == "soft":
            print("  [HITL] 신뢰도가 보통입니다. 답변을 확인해 주세요.")
        else:
            print("  [HITL] 신뢰도가 낮습니다. 반드시 검토가 필요합니다.")

        print(f"  [미리보기] {context.answer[:200]}...")
        print()
        print("  [1] 승인  [2] 수정  [3] 재검색  [4] 거부")

        try:
            choice = input("  선택> ").strip()
        except (EOFError, KeyboardInterrupt):
            return HITLDecision(action="approve")

        if choice == "2":
            edited = input("  수정된 답변> ").strip()
            return HITLDecision(action="edit", edited_answer=edited)
        elif choice == "3":
            new_q = input("  새 검색어> ").strip()
            return HITLDecision(action="retry", new_query=new_q)
        elif choice == "4":
            return HITLDecision(action="reject")
        else:
            return HITLDecision(action="approve")

    def collect_feedback(self, query: str, answer: str) -> Feedback | None:
        """답변 후 사용자 피드백을 수집한다."""
        if self.mode == "off":
            return None

        try:
            fb = input("  피드백 (👍 1 / 👎 2 / 건너뛰기 Enter): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if fb == "1":
            return Feedback(query=query, answer=answer, rating="up")
        elif fb == "2":
            return Feedback(query=query, answer=answer, rating="down")
        return None


class FeedbackStore:
    """피드백을 JSONL 파일로 저장한다."""

    def __init__(self, filepath: str = "./data/feedback.jsonl"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def save(self, feedback: Feedback):
        with open(self.filepath, "a", encoding="utf-8") as f:
            data = {
                "query": feedback.query,
                "answer": feedback.answer,
                "rating": feedback.rating,
                "timestamp": feedback.timestamp,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict]:
        if not self.filepath.exists():
            return []
        entries = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
