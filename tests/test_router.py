"""Router 단위 테스트"""

from tests.conftest import make_mock_llm, make_text_response
from src.router import Router


class TestRouter:
    def test_classify_internal_search(self):
        llm = make_mock_llm([make_text_response("INTERNAL_SEARCH")])
        router = Router(llm=llm)
        assert router.classify("휴가 신청 방법 알려줘") == "INTERNAL_SEARCH"

    def test_classify_web_search(self):
        llm = make_mock_llm([make_text_response("WEB_SEARCH")])
        router = Router(llm=llm)
        assert router.classify("오늘 서울 날씨 어때?") == "WEB_SEARCH"

    def test_classify_chitchat(self):
        llm = make_mock_llm([make_text_response("CHITCHAT")])
        router = Router(llm=llm)
        assert router.classify("안녕하세요!") == "CHITCHAT"

    def test_classify_fallback_on_invalid(self):
        """유효하지 않은 응답 시 INTERNAL_SEARCH로 폴백."""
        llm = make_mock_llm([make_text_response("UNKNOWN_ROUTE")])
        router = Router(llm=llm)
        assert router.classify("아무말") == "INTERNAL_SEARCH"

    def test_classify_extracts_from_verbose_response(self):
        """여러 단어가 포함된 응답에서 유효 라우트 추출."""
        llm = make_mock_llm([make_text_response("결과는 WEB_SEARCH 입니다")])
        router = Router(llm=llm)
        assert router.classify("주가 알려줘") == "WEB_SEARCH"

    def test_classify_case_insensitive(self):
        """소문자 응답도 처리."""
        llm = make_mock_llm([make_text_response("chitchat")])
        router = Router(llm=llm)
        assert router.classify("ㅎㅇ") == "CHITCHAT"
