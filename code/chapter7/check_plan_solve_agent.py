"""无需模型密钥或网络的回归检查：直接运行本文件即可。"""

import contextlib
import io
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from hello_agents import Config, HelloAgentsLLM
from my_plan_solve_agent import MyPlanAndSolveAgent, Planner


QUESTION = "周一卖出15个苹果，周二是两倍，周三比周二少5个，三天共多少个？"
PLAN = '["计算周二销量", "计算周三销量", "求三天总和"]'


class FakeLLM(HelloAgentsLLM):
    """用预设响应代替真实请求，记录发给模型的上下文。"""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.provider = "test"

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class PlanAndSolveChecks(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_supported_plan_formats(self):
        for response in (
            PLAN,
            f"```python\n{PLAN}\n```",
            f"```json\n{PLAN}\n```",
            f"```\n{PLAN}\n```",
            f"这是计划：\n```Python\n{PLAN}\n```\n请按顺序执行。",
            f"python\n{PLAN}",
        ):
            with self.subTest(response=response):
                self.assertEqual(
                    Planner._parse_plan(response),
                    ["计算周二销量", "计算周三销量", "求三天总和"],
                )

    def test_invalid_plans_are_rejected_without_execution(self):
        for response in ("", "[]", "{}", "[1]", "['']", "['  ']", "[['嵌套']]",
                         "['第一步', None]", "('第一步',)", "第一步：计算", "['未闭合'"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                Planner._parse_plan(response)
        with patch("os.system") as system, self.assertRaises(ValueError):
            Planner._parse_plan("__import__('os').system('echo unexpected')")
        system.assert_not_called()

    def test_default_flow_and_step_context(self):
        llm = FakeLLM([PLAN, "30", "25", "70"])
        agent = MyPlanAndSolveAgent("默认助手", llm)
        self.assertEqual(agent.run(QUESTION), "70")
        self.assertEqual(len(llm.calls), 4)
        prompts = [messages[-1]["content"] for messages, _ in llm.calls]
        self.assertIn("结果: 30", prompts[2])
        self.assertIn("结果: 30", prompts[3])
        self.assertIn("结果: 25", prompts[3])
        self.assertIn("求三天总和", prompts[3])
        self.assertEqual([(m.role, m.content) for m in agent.get_history()],
                         [("user", QUESTION), ("assistant", "70")])

    def test_custom_prompts(self):
        prompts = {
            "planner": "数学规划：{question}\npython\n[\"步骤\"]",
            "executor": "数学执行：{question}\n{plan}\n{history}\n{current_step}",
        }
        llm = FakeLLM([f"python\n{PLAN}", "30", "25", "70"])
        agent = MyPlanAndSolveAgent("数学助手", llm, custom_prompts=prompts)
        self.assertEqual(agent.run(QUESTION), "70")
        self.assertTrue(llm.calls[0][0][-1]["content"].startswith("数学规划："))
        self.assertTrue(llm.calls[1][0][-1]["content"].startswith("数学执行："))
        self.assertIn("结果: 25", llm.calls[-1][0][-1]["content"])

    def test_partial_customization_and_system_prompt(self):
        llm = FakeLLM(['["回答"]', "70"])
        config = Config()
        prompts = {"planner": "规划：{question}"}
        agent = MyPlanAndSolveAgent("配置助手", llm, system_prompt="简洁回答",
                                   config=config, custom_prompts=prompts)
        self.assertEqual(agent.run(QUESTION, temperature=0.1, max_tokens=128), "70")
        self.assertIs(agent.config, config)
        self.assertEqual(prompts, {"planner": "规划：{question}"})
        for messages, kwargs in llm.calls:
            self.assertEqual(messages[0], {"role": "system", "content": "简洁回答"})
            self.assertEqual(kwargs, {"temperature": 0.1, "max_tokens": 128})

    def test_invalid_plan_stops_before_executor(self):
        llm = FakeLLM(["['有效步骤', 42]"])
        agent = MyPlanAndSolveAgent("无效计划", llm)
        result = agent.run(QUESTION)
        self.assertIn("规划失败", result)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(agent.get_history()[-1].content, result)

    def test_missing_model_output_stops_task(self):
        for missing in (None, "", "  ", 42):
            with self.subTest(missing=missing):
                llm = FakeLLM([missing])
                self.assertIn("规划失败", MyPlanAndSolveAgent("规划", llm).run(QUESTION))
                llm = FakeLLM([PLAN, missing])
                result = MyPlanAndSolveAgent("执行", llm).run(QUESTION)
                self.assertIn("第 1 步执行失败", result)
                self.assertEqual(len(llm.calls), 2)

    def test_model_errors_report_stage_and_stop(self):
        llm = FakeLLM([RuntimeError("unavailable")])
        self.assertIn("规划失败", MyPlanAndSolveAgent("规划", llm).run(QUESTION))
        llm = FakeLLM([PLAN, "30", RuntimeError("unavailable")])
        agent = MyPlanAndSolveAgent("执行", llm)
        result = agent.run(QUESTION)
        self.assertIn("第 2 步执行失败", result)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len(agent.get_history()), 2)

    def test_bad_prompt_placeholders_report_failure(self):
        llm = FakeLLM([])
        agent = MyPlanAndSolveAgent("规划", llm, custom_prompts={"planner": "{unknown}"})
        self.assertIn("规划器提示词格式错误", agent.run(QUESTION))
        self.assertEqual(len(llm.calls), 0)
        llm = FakeLLM([PLAN])
        agent = MyPlanAndSolveAgent("执行", llm, custom_prompts={"executor": "{unknown}"})
        self.assertIn("第 1 步执行失败", agent.run(QUESTION))
        self.assertEqual(len(llm.calls), 1)

    def test_empty_question_and_invalid_prompt_configuration(self):
        llm = FakeLLM([])
        self.assertIn("问题不能为空", MyPlanAndSolveAgent("空问题", llm).run("  "))
        self.assertEqual(len(llm.calls), 0)
        with self.assertRaises(ValueError):
            MyPlanAndSolveAgent("错误配置", llm, custom_prompts={"planner": ""})

    def test_repeated_runs_have_independent_execution_context(self):
        llm = FakeLLM(['["回答第一题"]', "旧答案", '["回答第二题"]', "新答案"])
        agent = MyPlanAndSolveAgent("多次调用", llm)
        agent.run("第一题")
        self.assertEqual(agent.run("第二题"), "新答案")
        self.assertNotIn("旧答案", llm.calls[-1][0][-1]["content"])
        self.assertEqual(len(agent.get_history()), 4)

    def test_demo_import_has_no_model_requests(self):
        source = Path(__file__).with_name("test_plan_solve_agent.py")
        with patch("hello_agents.core.llm.HelloAgentsLLM") as constructor:
            imported = runpy.run_path(str(source), run_name="imported_demo")
        constructor.assert_not_called()
        self.assertTrue(callable(imported["main"]))

    def test_demo_runs_default_and_math_examples(self):
        llm = FakeLLM([PLAN, "30", "25", "70", f"python\n{PLAN}", "30", "25", "70"])
        source = Path(__file__).with_name("test_plan_solve_agent.py")
        with patch("hello_agents.core.llm.HelloAgentsLLM", return_value=llm), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(source), run_name="__main__")
        self.assertIn("最终结果: 70", output.getvalue())
        self.assertIn("数学专用Agent结果: 70", output.getvalue())
        self.assertEqual(len(llm.calls), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
