"""Calculator MCP Server - 수학 계산 도구를 MCP로 제공

stdio를 통해 JSON-RPC 메시지를 주고받는 MCP 서버이다.
안전한 수식 평가와 세금 계산 기능을 제공한다.
"""

import ast
import json
import math
import operator
import sys

# Windows CP949 → UTF-8 인코딩 강제 (UnicodeDecodeError 방지)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# 허용할 연산자 매핑
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 허용할 함수 매핑
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "min": min,
    "max": max,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
}


def _safe_eval_node(node):
    """AST 노드를 안전하게 평가한다. eval() 대신 사용."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"허용되지 않는 상수: {node.value}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"허용되지 않는 연산자: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"허용되지 않는 단항 연산자: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("함수 호출만 허용됩니다.")
        func_name = node.func.id
        if func_name not in _FUNCTIONS:
            raise ValueError(f"허용되지 않는 함수: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[func_name](*args)
    else:
        raise ValueError(f"허용되지 않는 표현식: {type(node).__name__}")


def safe_calculate(expression: str) -> float:
    """수식 문자열을 안전하게 계산한다."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval_node(tree)


# 2024 한국 종합소득세 누진세율표
INCOME_TAX_BRACKETS = [
    (14_000_000, 0.06, 0),
    (50_000_000, 0.15, 1_260_000),
    (88_000_000, 0.24, 5_760_000),
    (150_000_000, 0.35, 15_440_000),
    (300_000_000, 0.38, 19_940_000),
    (500_000_000, 0.40, 25_940_000),
    (1_000_000_000, 0.42, 35_940_000),
    (float("inf"), 0.45, 65_940_000),
]


def calculate_income_tax(taxable_income: float) -> dict:
    """과세표준 기준 종합소득세를 계산한다."""
    if taxable_income <= 0:
        return {
            "과세표준": 0,
            "적용세율": "0%",
            "산출세액": 0,
            "누진공제": 0,
            "설명": "과세표준이 0 이하입니다.",
        }

    for upper, rate, deduction in INCOME_TAX_BRACKETS:
        if taxable_income <= upper:
            tax = taxable_income * rate - deduction
            return {
                "과세표준": int(taxable_income),
                "적용세율": f"{rate * 100:.0f}%",
                "산출세액": int(tax),
                "누진공제": int(deduction),
                "계산식": f"{int(taxable_income):,} × {rate * 100:.0f}% - {int(deduction):,} = {int(tax):,}",
                "설명": f"과세표준 {int(taxable_income):,}원에 대한 종합소득세는 {int(tax):,}원입니다.",
            }

    return {"error": "계산 오류"}


TOOLS = [
    {
        "name": "calculate",
        "description": "수학 수식을 계산합니다. 사칙연산, 거듭제곱, 제곱근, 반올림 등을 지원합니다. 세금, 급여, 비용 등 숫자 계산이 필요할 때 사용하세요.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "계산할 수식 (예: '50000000 * 0.24 - 5760000', 'sqrt(144)', 'round(3.14159, 2)')",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "calculate_income_tax",
        "description": "한국 종합소득세를 계산합니다. 과세표준 금액을 입력하면 누진세율을 적용하여 산출세액을 반환합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "taxable_income": {
                    "type": "number",
                    "description": "과세표준 금액 (원 단위, 예: 50000000)",
                },
            },
            "required": ["taxable_income"],
        },
    },
]


def handle_calculate(expression: str) -> dict:
    """수식 계산 도구 핸들러."""
    try:
        result = safe_calculate(expression)
        output = {
            "expression": expression,
            "result": result,
            "formatted": f"{result:,.2f}" if isinstance(result, float) else f"{result:,}",
        }
    except Exception as e:
        output = {"expression": expression, "error": str(e)}

    return {
        "content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}]
    }


def handle_income_tax(taxable_income: float) -> dict:
    """소득세 계산 도구 핸들러."""
    result = calculate_income_tax(taxable_income)
    return {
        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
    }


def handle_request(req: dict) -> dict:
    method = req.get("method", "")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "calculator", "version": "1.0.0"},
        }
    elif method == "notifications/initialized":
        return {}
    elif method == "tools/list":
        return {"tools": TOOLS}
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "calculate":
            return handle_calculate(args.get("expression", "0"))
        elif tool_name == "calculate_income_tax":
            return handle_income_tax(args.get("taxable_income", 0))
        else:
            return {
                "content": [
                    {"type": "text", "text": json.dumps({"error": f"알 수 없는 도구: {tool_name}"}, ensure_ascii=False)}
                ]
            }
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
