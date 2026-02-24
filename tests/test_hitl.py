"""HITL (Human in the Loop) 단위 테스트"""

import json
import tempfile
from pathlib import Path
from src.hitl import (
    ConfidenceCalculator,
    HITLManager,
    HITLContext,
    HITLDecision,
    Feedback,
    FeedbackStore,
)


class TestConfidenceCalculator:
    def setup_method(self):
        self.calc = ConfidenceCalculator()

    def test_high_confidence(self):
        """PASS, 높은 유사도, 재시도 없음 → 높은 신뢰도."""
        score = self.calc.calculate(
            grader_result="PASS",
            vector_scores=[0.1, 0.2, 0.15],  # distance 낮음 = 유사도 높음
            retry_count=0,
            doc_count=3,
        )
        assert score >= 0.8

    def test_low_confidence(self):
        """FAIL, 낮은 유사도, 재시도 있음 → 낮은 신뢰도."""
        score = self.calc.calculate(
            grader_result="FAIL",
            vector_scores=[0.9, 0.95],  # distance 높음 = 유사도 낮음
            retry_count=1,
            doc_count=1,
        )
        assert score < 0.5

    def test_medium_confidence(self):
        """중간 수준의 입력 → 중간 신뢰도."""
        score = self.calc.calculate(
            grader_result="PASS",
            vector_scores=[0.5],
            retry_count=1,
            doc_count=2,
        )
        assert 0.3 <= score <= 0.8

    def test_no_vector_scores(self):
        """벡터 점수가 없을 때 기본값 사용."""
        score = self.calc.calculate(grader_result="PASS")
        assert 0.0 <= score <= 1.0

    def test_score_bounds(self):
        """신뢰도 점수가 0~1 범위인지 확인."""
        for grader in ("PASS", "FAIL"):
            for retry in (0, 1, 2):
                score = self.calc.calculate(
                    grader_result=grader,
                    vector_scores=[0.5],
                    retry_count=retry,
                    doc_count=2,
                )
                assert 0.0 <= score <= 1.0


class TestHITLManager:
    def test_off_mode_no_intervention(self):
        manager = HITLManager(mode="off")
        assert manager.should_intervene(0.1) == "none"

    def test_strict_mode_always_hard(self):
        manager = HITLManager(mode="strict")
        assert manager.should_intervene(0.99) == "hard"

    def test_auto_high_confidence(self):
        manager = HITLManager(mode="auto")
        assert manager.should_intervene(0.85) == "none"

    def test_auto_medium_confidence(self):
        manager = HITLManager(mode="auto")
        assert manager.should_intervene(0.6) == "soft"

    def test_auto_low_confidence(self):
        manager = HITLManager(mode="auto")
        assert manager.should_intervene(0.3) == "hard"

    def test_auto_boundary_high(self):
        manager = HITLManager(mode="auto")
        assert manager.should_intervene(0.8) == "none"

    def test_auto_boundary_low(self):
        manager = HITLManager(mode="auto")
        assert manager.should_intervene(0.5) == "soft"

    def test_request_review_auto_approve(self):
        """HIGH 신뢰도 → 자동 승인."""
        manager = HITLManager(mode="auto")
        context = HITLContext(query="q", answer="a", confidence=0.9)
        decision = manager.request_review(context)
        assert decision.action == "approve"


class TestFeedbackStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "feedback.jsonl")
            store = FeedbackStore(filepath=filepath)

            fb1 = Feedback(query="q1", answer="a1", rating="up")
            fb2 = Feedback(query="q2", answer="a2", rating="down")
            store.save(fb1)
            store.save(fb2)

            entries = store.load_all()
            assert len(entries) == 2
            assert entries[0]["rating"] == "up"
            assert entries[1]["rating"] == "down"

    def test_load_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "nonexistent.jsonl")
            store = FeedbackStore(filepath=filepath)
            assert store.load_all() == []
