"""Calculator MCP Server 테스트

안전한 수식 평가기, 소득세 계산, MCP 프로토콜을 검증한다.
"""

from src.mcp_servers.calculator_server import (
    safe_calculate,
    calculate_income_tax,
    handle_request,
)


class TestSafeCalculate:
    """안전한 수식 평가기 테스트."""

    def test_basic_arithmetic(self):
        assert safe_calculate("2 + 3") == 5
        assert safe_calculate("10 - 4") == 6
        assert safe_calculate("3 * 7") == 21
        assert safe_calculate("15 / 4") == 3.75

    def test_complex_expression(self):
        assert safe_calculate("(100 + 200) * 3") == 900
        assert safe_calculate("50000000 * 0.24 - 5760000") == 6240000.0

    def test_floor_division_and_mod(self):
        assert safe_calculate("17 // 5") == 3
        assert safe_calculate("17 % 5") == 2

    def test_power(self):
        assert safe_calculate("2 ** 10") == 1024

    def test_unary_operators(self):
        assert safe_calculate("-5 + 10") == 5

    def test_functions(self):
        assert safe_calculate("round(3.14159, 2)") == 3.14
        assert safe_calculate("abs(-42)") == 42
        assert safe_calculate("max(10, 20, 30)") == 30
        assert safe_calculate("min(10, 20, 30)") == 10
        assert safe_calculate("sqrt(144)") == 12.0
        assert safe_calculate("ceil(3.1)") == 4
        assert safe_calculate("floor(3.9)") == 3

    def test_nested_expressions(self):
        assert safe_calculate("round(50000000 * 0.15 - 1260000)") == 6240000

    def test_rejects_dangerous_code(self):
        """위험한 코드 실행을 차단."""
        import pytest

        with pytest.raises(ValueError):
            safe_calculate("__import__('os').system('ls')")
        with pytest.raises(ValueError):
            safe_calculate("open('/etc/passwd').read()")

    def test_rejects_string_literals(self):
        import pytest

        with pytest.raises(ValueError):
            safe_calculate("'hello'")


class TestCalculateIncomeTax:
    """종합소득세 계산 테스트."""

    def test_lowest_bracket(self):
        """1,400만원 이하 → 6%."""
        result = calculate_income_tax(10_000_000)
        assert result["적용세율"] == "6%"
        assert result["산출세액"] == 600_000

    def test_second_bracket(self):
        """1,400만~5,000만원 → 15%."""
        result = calculate_income_tax(30_000_000)
        assert result["적용세율"] == "15%"
        assert result["산출세액"] == 30_000_000 * 0.15 - 1_260_000

    def test_third_bracket(self):
        """5,000만~8,800만원 → 24%."""
        result = calculate_income_tax(60_000_000)
        assert result["적용세율"] == "24%"
        assert result["산출세액"] == int(60_000_000 * 0.24 - 5_760_000)

    def test_high_income(self):
        """1.5억~3억 → 38%."""
        result = calculate_income_tax(200_000_000)
        assert result["적용세율"] == "38%"
        expected = int(200_000_000 * 0.38 - 19_940_000)
        assert result["산출세액"] == expected

    def test_zero_income(self):
        result = calculate_income_tax(0)
        assert result["산출세액"] == 0

    def test_negative_income(self):
        result = calculate_income_tax(-100)
        assert result["산출세액"] == 0

    def test_result_has_explanation(self):
        """결과에 설명이 포함."""
        result = calculate_income_tax(50_000_000)
        assert "설명" in result
        assert "계산식" in result


class TestCalculatorMCPProtocol:
    """MCP 프로토콜 테스트."""

    def test_initialize(self):
        result = handle_request({"method": "initialize", "params": {}})
        assert result["serverInfo"]["name"] == "calculator"

    def test_tools_list(self):
        result = handle_request({"method": "tools/list", "params": {}})
        tool_names = [t["name"] for t in result["tools"]]
        assert "calculate" in tool_names
        assert "calculate_income_tax" in tool_names

    def test_calculate_tool_call(self):
        result = handle_request({
            "method": "tools/call",
            "params": {
                "name": "calculate",
                "arguments": {"expression": "100 + 200"},
            },
        })
        import json
        text = result["content"][0]["text"]
        data = json.loads(text)
        assert data["result"] == 300

    def test_income_tax_tool_call(self):
        result = handle_request({
            "method": "tools/call",
            "params": {
                "name": "calculate_income_tax",
                "arguments": {"taxable_income": 60000000},
            },
        })
        import json
        text = result["content"][0]["text"]
        data = json.loads(text)
        assert data["적용세율"] == "24%"

    def test_unknown_tool(self):
        result = handle_request({
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {},
            },
        })
        import json
        text = result["content"][0]["text"]
        data = json.loads(text)
        assert "error" in data

    def test_notifications_initialized(self):
        result = handle_request({"method": "notifications/initialized"})
        assert result == {}
