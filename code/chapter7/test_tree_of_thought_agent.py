"""可见输出的 ToT 测试；设置 TOT_LIVE_TEST=1 可额外运行真实模型测试。"""

import os
from pathlib import Path
import unittest
from unittest.mock import Mock

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

if __package__:
    from .my_tree_of_thought_agent import TreeOfThoughtAgent
else:
    from my_tree_of_thought_agent import TreeOfThoughtAgent


class TreeOfThoughtAgentTests(unittest.TestCase):
    """默认使用预设响应验证分支选择，同时将结果打印到控制台。"""

    @staticmethod
    def show_result(label: str, task: str, agent: TreeOfThoughtAgent, result: str) -> None:
        print(f"\n=== {label} ===\n任务：{task}")
        for step in agent.search_trace:
            print(f"\n第 {step['depth']} 层候选方案：")
            for index, (candidate, score) in enumerate(zip(step["candidates"], step["scores"])):
                selected = " ← 选中" if index == step["selected_index"] else ""
                print(f"  {index + 1}. [{score} 分] {candidate}{selected}")
        print("\n选定路径：")
        for index, step in enumerate(agent.selected_path, start=1):
            print(f"  {index}. {step}")
        print(f"\n最终答案：\n{result}", flush=True)

    def test_math_result(self):
        task = "使用 3、3、8、8 和四则运算得到 24，每个数恰好使用一次。"
        llm = Mock(spec=HelloAgentsLLM)
        llm.invoke.side_effect = [
            '["直接将四个数相加", "构造分母 1/3，再用 8 除以它", "先将两个 8 相加"]',
            "[20, 95, 40]",
            '["计算 3 - 8/3 = 1/3", "计算 8 - 3 = 5", "计算 3 + 8 = 11"]',
            "[99, 40, 30]",
            '["计算 8 * (3 - 8/3)", "计算 8 / (3 - 8/3)", "计算 (8 + 8) + (3 + 3)"]',
            "[15, 100, 10]",
            "8 / (3 - 8/3) = 8 / (1/3) = 24，每个数恰好使用一次。",
        ]
        agent = TreeOfThoughtAgent("数学 ToT 助手", llm, max_depth=3, branching_factor=3)
        result = agent.run(task)
        self.show_result("模拟模型：24 点问题", task, agent, result)

        self.assertEqual(result, "8 / (3 - 8/3) = 8 / (1/3) = 24，每个数恰好使用一次。")
        self.assertEqual(agent.selected_path, [
            "构造分母 1/3，再用 8 除以它",
            "计算 3 - 8/3 = 1/3",
            "计算 8 / (3 - 8/3)",
        ])
        self.assertEqual(llm.invoke.call_count, 7)
        self.assertEqual(agent.get_history()[-1].content, result)
        next_prompt = llm.invoke.call_args_list[2].args[0][-1]["content"]
        self.assertIn(agent.selected_path[0], next_prompt)
        self.assertNotIn("直接将四个数相加", next_prompt)

    def test_learning_plan_result(self):
        task = "安排一天的 Python 学习，包括基础语法、编程练习和复盘。"
        llm = Mock(spec=HelloAgentsLLM)
        llm.invoke.side_effect = [
            '["全天只看视频", "上午学语法，下午做练习，晚上复盘", "直接开发复杂项目"]',
            "[40, 95, 60]",
            '["每阶段结束时核验学习成果", "跳过编程练习", "取消复盘"]',
            "[98, 20, 30]",
            "上午：学习变量、条件和循环，并完成小练习。\n"
            "下午：编写一个小程序，并检查运行结果。\n"
            "晚上：整理错误和知识点，列出后续学习任务。",
        ]
        agent = TreeOfThoughtAgent("学习规划 ToT 助手", llm, max_depth=2, branching_factor=3)
        result = agent.run(task)
        self.show_result("模拟模型：学习规划", task, agent, result)

        self.assertEqual(agent.selected_path, [
            "上午学语法，下午做练习，晚上复盘", "每阶段结束时核验学习成果",
        ])
        self.assertEqual(llm.invoke.call_count, 5)
        self.assertIn("上午：", result)
        self.assertIn("下午：", result)
        self.assertIn("晚上：", result)
        self.assertEqual(len(agent.get_history()), 2)

    @unittest.skipUnless(os.getenv("TOT_LIVE_TEST") == "1", "设置 TOT_LIVE_TEST=1 启用真实模型测试")
    def test_live_model_result(self):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        llm = HelloAgentsLLM(timeout=int(os.getenv("LLM_TIMEOUT", "180")))
        task = "安排一天的 Python 学习，包括基础语法、编程练习和复盘。"
        agent = TreeOfThoughtAgent("真实模型 ToT 助手", llm, max_depth=2, branching_factor=3)
        print(f"\n正在调用真实模型：{llm.model}；请求超时配置：{llm.timeout} 秒", flush=True)
        result = agent.run(task)
        self.show_result("真实模型：学习规划", task, agent, result)
        self.assertTrue(result.strip())
        self.assertEqual(len(agent.selected_path), 2)
        self.assertEqual(len(agent.search_trace), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
