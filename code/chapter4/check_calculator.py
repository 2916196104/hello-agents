"""离线验证计算器和 ReAct 调用流程，不消耗 API 额度。"""
import runpy
from pathlib import Path
from unittest.mock import patch

from tools import ToolExecutor, calculator
from ReAct import ReActAgent


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

    executor = ToolExecutor()
    executor.registerTool("Calculator", "输入非空四则运算表达式，如 1 + 2。", calculator)
    assert executor.executeTool("Missing", "1 + 2")[0] is False
    assert executor.executeTool("Calculator", " ")[0] is False
    assert executor.executeTool("Calculator", 123)[0] is False
    assert executor.executeTool("Calculator", "1 / 0")[0] is False
    assert executor.executeTool("Calculator", "1 + 2") == (True, "3")

    def broken_tool(value):
        raise ValueError("参数无效")

    executor.registerTool("Broken", "模拟参数错误。", broken_tool)
    assert executor.executeTool("Broken", "input")[0] is False
    executor.registerTool("Empty", "模拟无结果。", lambda value: None)
    assert executor.executeTool("Empty", "input")[0] is False
    executor.registerTool("Search", "模拟搜索失败。", lambda value: "搜索时发生错误: timeout")
    assert executor.executeTool("Search", "input")[0] is False

    class ReplayLLM:
        def __init__(self, actions):
            self.actions = iter(actions)
            self.prompts = []

        def think(self, messages):
            self.prompts.append(messages[0]["content"])
            action = next(self.actions)
            return f"Action: {action}" if action is not None else "没有按格式输出动作"

    # 工具名错误 -> 参数错误 -> 修正成功 -> 最终答案。
    replay = ReplayLLM(["Calclator[1 + 2]", "Calculator[1 / 0]",
                        "Calculator[1 + 2]", "Finish[3]"])
    agent = ReActAgent(replay, executor)
    assert agent.run("计算 1 + 2") == "3"
    assert "未找到工具 'Calclator'" in replay.prompts[1]
    assert "连续失败 1/3 次" in replay.prompts[1]
    assert "- Calculator:" in replay.prompts[1]
    assert "除数不能为0" in replay.prompts[2]
    assert "不要重复失败的调用" in replay.prompts[2]
    assert "Observation: 3" in replay.prompts[3]

    # 错误格式、空参数、工具异常累计失败，达到上限就停止。
    for bad_action in ("Missing[1]", "Calculator[]", "Calculator[1+]",
                       "Calculator[1+2]garbage", "Broken[input]", "Finish[]", None):
        replay = ReplayLLM([bad_action] * 3 + ["Finish[不应执行]"])
        agent = ReActAgent(replay, executor)
        assert agent.run("计算") is None
        assert len(replay.prompts) == 3
        assert "停止执行" in agent.history[-1]

    # 成功调用重置连续失败次数，新的 run 不继承上一任务的错误。
    replay = ReplayLLM(["Missing[1]", "Calculator[1+2]", "Missing[1]",
                        "Calculator[1+2]", "Finish[3]"])
    agent = ReActAgent(replay, executor, max_tool_failures=2)
    assert agent.run("计算") == "3"
    replay = ReplayLLM(["Missing[1]", "Missing[1]", "Calculator[1+2]", "Finish[3]"])
    agent = ReActAgent(replay, executor, max_tool_failures=2)
    assert agent.run("第一个任务") is None
    assert agent.run("第二个任务") == "3"
    assert "连续失败" not in replay.prompts[2]

    replay = ReplayLLM(["Calculator[1+2]"])
    assert ReActAgent(replay, executor, max_steps=1).run("计算") is None
    print("计算器、工具失败纠正与停止机制验证通过。")


if __name__ == "__main__":
    check()
