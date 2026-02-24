"""Grader 및 QueryRewriter 단위 테스트"""

from tests.conftest import make_mock_llm, make_text_response
from src.grader import Grader, QueryRewriter


class TestGrader:
    def test_evaluate_pass(self):
        llm = make_mock_llm([make_text_response("PASS")])
        grader = Grader(llm=llm)

        docs = [{"content": "휴가 신청은 HR 포털에서 합니다."}]
        result = grader.evaluate("휴가 신청 방법", docs)
        assert result == "PASS"

    def test_evaluate_fail(self):
        llm = make_mock_llm([make_text_response("FAIL")])
        grader = Grader(llm=llm)

        docs = [{"content": "회사 연혁 소개 문서입니다."}]
        result = grader.evaluate("출장비 정산", docs)
        assert result == "FAIL"

    def test_evaluate_empty_documents(self):
        llm = make_mock_llm()
        grader = Grader(llm=llm)

        result = grader.evaluate("아무 질문", [])
        assert result == "FAIL"

    def test_evaluate_fallback_on_invalid(self):
        """유효하지 않은 응답은 PASS로 폴백 (안전 모드)."""
        llm = make_mock_llm([make_text_response("잘 모르겠습니다")])
        grader = Grader(llm=llm)

        docs = [{"content": "테스트 문서"}]
        result = grader.evaluate("질문", docs)
        assert result == "PASS"

    def test_evaluate_extracts_from_verbose(self):
        """여러 단어 응답에서 FAIL 추출."""
        llm = make_mock_llm([make_text_response("판단 결과: FAIL 입니다")])
        grader = Grader(llm=llm)

        docs = [{"content": "무관한 문서"}]
        result = grader.evaluate("질문", docs)
        assert result == "FAIL"


class TestQueryRewriter:
    def test_rewrite(self):
        llm = make_mock_llm([make_text_response("연차 휴가 신청 절차 방법 가이드")])
        rewriter = QueryRewriter(llm=llm)

        result = rewriter.rewrite("회사에서 연차 쓰려면 어떻게 해야 해?")
        assert "연차" in result
        assert "절차" in result

    def test_rewrite_fallback_on_empty(self):
        """빈 응답 시 원본 쿼리 반환."""
        llm = make_mock_llm([make_text_response("")])
        rewriter = QueryRewriter(llm=llm)

        result = rewriter.rewrite("원본 질문")
        assert result == "원본 질문"
