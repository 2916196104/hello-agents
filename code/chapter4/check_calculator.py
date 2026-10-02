"""离线验证计算器和 ReAct 调用流程，不消耗 API 额度。"""
import runpy
from pathlib import Path
from unittest.mock import patch

from tools import calculator


def check():
    for expression, expected in (
        ("(123 + 456) × 789 / 12", "38069.25"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("-3 * (+2)", "-6"),
        ("1.5 ÷ 0.5", "3.0"),
    ):
        assert calculator(expression) == expected, expression
    assert calculator("1 / 0") == "错误：除数不能为0。"
    for expression in ("", "1 +", "True + 1", "2 ** 1000000", "abs(-1)",
                       "__import__('os').getcwd()", "(1).__class__", "[1][0]",
                       "1e309", "1e308 * 10", "1" * 201,
                       "+".join(["1"] * 40)):
        assert calculator(expression).startswith("错误："), expression

    class ScriptedLLM:
        calls = 0

        def think(self, messages):
            self.calls += 1
            prompt = messages[0]["content"]
            assert "- Calculator:" in prompt
            if self.calls == 1:
                return "Thought: 使用计算器。\nAction: Calculator[(123 + 456) × 789 / 12]"
            assert "Observation: 38069.25" in prompt
            return "Thought: 已获得结果。\nAction: Finish[38069.25]"

    llm = ScriptedLLM()
    with patch("llm_client.HelloAgentsLLM", return_value=llm):
        namespace = runpy.run_path(str(Path(__file__).with_name("ReAct.py")), run_name="__main__")
    assert llm.calls == 2
    assert namespace["agent"].history[-1] == "Observation: 38069.25"
    print("计算器与 ReAct 调用流程验证通过。")


if __name__ == "__main__":
    check()
