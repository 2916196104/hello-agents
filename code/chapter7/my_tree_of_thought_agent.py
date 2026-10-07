"""Tree-of-Thought：生成多个候选方案，评分并择优扩展。"""

import json
import math
from typing import Optional

from hello_agents import Config, HelloAgentsLLM, Message
from hello_agents.core.agent import Agent


class TreeOfThoughtAgent(Agent):
    """继承 Agent，按深度限制执行贪心分支搜索，再生成最终答案。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_depth: int = 3,
        branching_factor: int = 3,
    ):
        if type(max_depth) is not int or max_depth < 1:
            raise ValueError("max_depth 必须是正整数")
        if type(branching_factor) is not int or branching_factor < 2:
            raise ValueError("branching_factor 必须是至少为 2 的整数")
        super().__init__(name, llm, system_prompt, config)
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.selected_path: list[str] = []
        self.search_trace: list[dict] = []

    def _ask(self, prompt: str, **kwargs) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.llm.invoke(messages, **kwargs)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("模型未返回有效文本")
        return response.strip()

    @staticmethod
    def _parse_list(text: str) -> list:
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ValueError("模型必须返回合法的 JSON 数组，不要附加 Markdown") from exc
        if not isinstance(value, list):
            raise ValueError("模型必须返回 JSON 数组")
        return value

    def run(self, input_text: str, **kwargs) -> str:
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("任务必须是非空字符串")
        self.selected_path.clear()
        self.search_trace.clear()

        for depth in range(self.max_depth):
            context = json.dumps(self.selected_path, ensure_ascii=False)
            candidates = self._parse_list(self._ask(
                f"任务：{input_text}\n已选择的方案路径：{context}\n"
                f"请生成恰好 {self.branching_factor} 个不同的下一步候选方案。"
                "每项简洁描述一个可执行步骤或部分解答，不要展开冗长推理。"
                "若已经解决任务，可提出核验或完善方案。"
                '只返回 JSON 字符串数组，例如 ["方案A", "方案B"]。',
                **kwargs,
            ))
            if (len(candidates) != self.branching_factor
                    or any(not isinstance(item, str) or not item.strip()
                           for item in candidates)):
                raise ValueError("候选方案数量不正确，或包含空白/非文本方案")
            candidates = [item.strip() for item in candidates]
            if len(set(candidates)) != len(candidates):
                raise ValueError("候选方案必须互不重复")

            scores = self._parse_list(self._ask(
                f"任务：{input_text}\n已选择的方案路径：{context}\n"
                f"下一步候选方案：{json.dumps(candidates, ensure_ascii=False)}\n"
                "根据正确性、可行性和对任务完成的贡献，给每个候选方案打 0–100 分。"
                "只返回与候选方案顺序一致、长度相同的 JSON 数字数组。",
                **kwargs,
            ))
            if (len(scores) != len(candidates)
                    or any(isinstance(score, bool)
                           or not isinstance(score, (int, float))
                           or not 0 <= score <= 100
                           or not math.isfinite(score) for score in scores)):
                raise ValueError("候选评分必须是数量匹配的 0–100 有限数字")

            # 每轮只保留最高分分支；同分时选择先生成的候选。
            selected = max(range(len(scores)), key=scores.__getitem__)
            self.selected_path.append(candidates[selected])
            self.search_trace.append({
                "depth": depth + 1,
                "candidates": candidates,
                "scores": scores,
                "selected_index": selected,
            })

        result = self._ask(
            f"原始任务：{input_text}\n"
            f"选定的方案路径：{json.dumps(self.selected_path, ensure_ascii=False)}\n"
            "请核验并整合以上方案，直接给出完整的最终答案。",
            **kwargs,
        )
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(result, "assistant"))
        return result
