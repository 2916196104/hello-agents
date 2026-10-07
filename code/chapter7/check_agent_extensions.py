"""评分反思与 Tree-of-Thought 的离线检查：无需密钥或网络。"""

import contextlib
import io
import unittest

from hello_agents import Config, HelloAgentsLLM
from hello_agents.core.agent import Agent

# unittest 将文件作为 chapter7 下的模块加载；直接运行脚本时没有包名。
if __package__:
    from .my_reflection_agent import MyReflectionAgent
    from .my_tree_of_thought_agent import TreeOfThoughtAgent
else:
    from my_reflection_agent import MyReflectionAgent
    from my_tree_of_thought_agent import TreeOfThoughtAgent


class FakeLLM(HelloAgentsLLM):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.provider = "test"

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class AgentExtensionChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_reflection_stops_at_or_above_threshold(self):
        for score in ("85", "95.5"):
            with self.subTest(score=score):
                llm = FakeLLM(["初稿", "有改进空间", score])
                agent = MyReflectionAgent("反思", llm, quality_threshold=85)
                self.assertEqual(agent.run("任务"), "初稿")
                self.assertEqual(len(llm.calls), 3)
                self.assertEqual(agent.scores, [float(score)])
                self.assertEqual([r["type"] for r in agent.memory.records],
                                 ["execution", "reflection", "score"])
                self.assertEqual([(m.role, m.content) for m in agent.get_history()],
                                 [("user", "任务"), ("assistant", "初稿")])

    def test_reflection_low_score_overrides_stop_marker_and_scores_new_version(self):
        llm = FakeLLM(["初稿", "无需改进", "60", "新版", "准确完整", "90"])
        config = Config()
        agent = MyReflectionAgent("反思", llm, system_prompt="简洁回答", config=config)
        self.assertEqual(agent.run("任务", temperature=0.1), "新版")
        self.assertEqual(agent.scores, [60, 90])
        self.assertIs(agent.config, config)
        self.assertIn("新版", llm.calls[-1][0][-1]["content"])
        self.assertIn("无需改进", llm.calls[3][0][-1]["content"])
        for messages, kwargs in llm.calls:
            self.assertEqual(messages[0], {"role": "system", "content": "简洁回答"})
            self.assertEqual(kwargs, {"temperature": 0.1})

    def test_reflection_iteration_limit_and_zero_refinements(self):
        for limit, responses, expected, scores in (
            (0, ["初稿", "需改进", "20"], "初稿", [20]),
            (1, ["初稿", "需改进", "20", "新版", "仍需改进", "30"],
             "新版", [20, 30]),
        ):
            with self.subTest(limit=limit):
                llm = FakeLLM(responses)
                agent = MyReflectionAgent("反思", llm, max_iterations=limit)
                self.assertEqual(agent.run("任务"), expected)
                self.assertEqual(agent.scores, scores)
                self.assertEqual(len(llm.calls), len(responses))

    def test_reflection_custom_score_prompt_and_repeated_runs(self):
        llm = FakeLLM(["初稿一", "意见一", "90", "初稿二", "意见二", "95"])
        prompts = {"score": "评分：{task}|{content}|{feedback}"}
        agent = MyReflectionAgent("反思", llm, custom_prompts=prompts)
        agent.run("任务一")
        self.assertEqual(agent.run("任务二"), "初稿二")
        self.assertEqual(agent.scores, [95])
        self.assertEqual(len(agent.memory.records), 3)
        self.assertEqual(len(agent.get_history()), 4)
        self.assertEqual(llm.calls[-1][0][-1]["content"], "评分：任务二|初稿二|意见二")
        self.assertEqual(prompts, {"score": "评分：{task}|{content}|{feedback}"})

    def test_reflection_rejects_invalid_scores_without_refining(self):
        for score in ("NaN", "inf", "-1", "101", "80/100", "优秀", ""):
            with self.subTest(score=score), self.assertRaises(ValueError):
                llm = FakeLLM(["初稿", "意见", score])
                agent = MyReflectionAgent("反思", llm)
                try:
                    agent.run("任务")
                finally:
                    self.assertEqual(len(llm.calls), 3)
                    self.assertEqual(agent.get_history(), [])

    def test_tree_selects_best_branches_and_carries_only_selected_path(self):
        llm = FakeLLM([
            '["失败方案", "最佳方案"]', "[20, 90]",
            '["完成方案", "无关方案"]', "[95, 10]", "最终答案",
        ])
        config = Config()
        agent = TreeOfThoughtAgent("树搜索", llm, system_prompt="准确回答", config=config,
                                   max_depth=2, branching_factor=2)
        self.assertIsInstance(agent, Agent)
        self.assertIs(agent.config, config)
        self.assertEqual(agent.run("任务", max_tokens=512), "最终答案")
        self.assertEqual(agent.selected_path, ["最佳方案", "完成方案"])
        self.assertEqual([s["selected_index"] for s in agent.search_trace], [1, 0])
        second_step = llm.calls[2][0][-1]["content"]
        final_prompt = llm.calls[-1][0][-1]["content"]
        self.assertIn("最佳方案", second_step)
        self.assertNotIn("失败方案", second_step)
        self.assertIn("完成方案", final_prompt)
        self.assertNotIn("无关方案", final_prompt)
        self.assertEqual(len(agent.get_history()), 2)
        for messages, kwargs in llm.calls:
            self.assertEqual(messages[0], {"role": "system", "content": "准确回答"})
            self.assertEqual(kwargs, {"max_tokens": 512})

    def test_tree_ties_are_stable_and_trace_resets(self):
        llm = FakeLLM(['["A", "B"]', "[0, 0]", "第一答",
                       '["C", "D"]', "[100, 90]", "第二答"])
        agent = TreeOfThoughtAgent("树搜索", llm, max_depth=1, branching_factor=2)
        self.assertEqual(agent.run("任务一"), "第一答")
        self.assertEqual(agent.selected_path, ["A"])
        self.assertEqual(agent.run("任务二"), "第二答")
        self.assertEqual(agent.selected_path, ["C"])
        self.assertEqual(len(agent.search_trace), 1)
        self.assertEqual(len(agent.get_history()), 4)

    def test_tree_rejects_invalid_candidates_before_evaluation(self):
        for text in ('["A"]', '["A", " "]', '["A", 1]', '[" A", "A "]',
                     "{}", "不是JSON", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                llm = FakeLLM([text])
                agent = TreeOfThoughtAgent("树搜索", llm, branching_factor=2)
                try:
                    agent.run("任务")
                finally:
                    self.assertEqual(len(llm.calls), 1)
                    self.assertEqual(agent.get_history(), [])

    def test_tree_rejects_invalid_scores_before_selection(self):
        for scores in ("[90]", "[true, 90]", '["90", 80]', "[NaN, 80]",
                       "[Infinity, 80]", "[-1, 80]", "[101, 80]", "{}"):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                llm = FakeLLM(['["A", "B"]', scores])
                agent = TreeOfThoughtAgent("树搜索", llm, branching_factor=2)
                try:
                    agent.run("任务")
                finally:
                    self.assertEqual(agent.selected_path, [])
                    self.assertEqual(len(llm.calls), 2)

    def test_invalid_configuration_inputs_and_model_failures(self):
        llm = FakeLLM([])
        for threshold in (-1, 101, float("nan"), float("inf"), True, "85"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                MyReflectionAgent("反思", llm, quality_threshold=threshold)
        for limit in (-1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                MyReflectionAgent("反思", llm, max_iterations=limit)
        for options in ({"max_depth": 0}, {"max_depth": True},
                        {"branching_factor": 1}, {"branching_factor": 2.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TreeOfThoughtAgent("树搜索", llm, **options)
        for agent_class in (MyReflectionAgent, TreeOfThoughtAgent):
            for text in ("", " ", None):
                with self.subTest(agent=agent_class, text=text), self.assertRaises(ValueError):
                    agent_class("测试", llm).run(text)
            for response, error in ((None, ValueError), (42, ValueError),
                                    (RuntimeError("模型不可用"), RuntimeError)):
                with self.subTest(agent=agent_class, response=response):
                    agent = agent_class("测试", FakeLLM([response]))
                    with self.assertRaises(error):
                        agent.run("任务")
                    self.assertEqual(agent.get_history(), [])


if __name__ == "__main__":
    unittest.main()
