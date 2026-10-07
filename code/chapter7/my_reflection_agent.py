"""第七章：生成初稿，再通过反思和改进迭代完成任务。"""

import math
from typing import Optional

from hello_agents import Config, HelloAgentsLLM, Message, ReflectionAgent


DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请指出不足之处，并提出具体的改进建议。
如果回答已经很好，请仅回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的完整回答。
""",
    "score": """
请评价当前回答对原始任务的完成质量，综合准确性、完整性和相关性给出 0–100 分。
任务: {task}
当前回答: {content}
反思意见: {feedback}
只输出一个数字，不要附加解释、单位或 Markdown。
""",

}

class MyReflectionAgent(ReflectionAgent):
    """每次反思后评分，仅在低于质量阈值时继续优化。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[dict[str, str]] = None,
        quality_threshold: float = 85,
    ):
        if type(max_iterations) is not int or max_iterations < 0:
            raise ValueError("max_iterations 必须是非负整数")
        if (isinstance(quality_threshold, bool)
                or not isinstance(quality_threshold, (int, float))
                or not 0 <= quality_threshold <= 100):
            raise ValueError("quality_threshold 必须是 0–100 的有限数字")
        self.quality_threshold = quality_threshold
        self.scores: list[float] = []

        # 每个实例保存独立模板，也允许只替换某一个阶段的提示词。
        prompts = DEFAULT_PROMPTS.copy()
        if custom_prompts:
            prompts.update(custom_prompts)

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            max_iterations=max_iterations,
            custom_prompts=prompts,
        )

    def run(self, input_text: str, **kwargs) -> str:
        """最多优化 max_iterations 次；最后一个版本也会反思并评分。"""
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("任务必须是非空字符串")
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 当前任务的执行轨迹重新开始，对话历史由 Agent 基类独立保存。
        self.memory.records.clear()
        self.scores.clear()
        # 根据原始任务生成初稿。
        initial_prompt = self.prompts["initial"].format(task=input_text)
        result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", result)
        # 初稿和每个优化版本都接受审查，避免返回未经评分的新版本。
        for iteration in range(self.max_iterations + 1):
            print(f"\n--- 第 {iteration + 1} 次反思 ---")
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text, content=result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            score_prompt = self.prompts["score"].format(
                task=input_text, content=result, feedback=feedback
            )
            score_text = self._get_llm_response(score_prompt, **kwargs)
            try:
                score = float(score_text)
            except ValueError as exc:
                raise ValueError("质量评分必须是 0–100 的数字") from exc
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError("质量评分必须是 0–100 的有限数字")
            self.scores.append(score)
            self.memory.add_record("score", str(score))
            print(f"⭐ 当前版本评分: {score}；阈值: {self.quality_threshold}")
            if score >= self.quality_threshold:
                print("\n✅ 达到质量阈值，提前结束。")
                break
            if iteration == self.max_iterations:
                print("\n已达到优化次数上限，返回当前版本。")
                break
            # 根据反馈生成新版本。
            refine_prompt = self.prompts["refine"].format(
                task=input_text, last_attempt=result, feedback=feedback
            )
            result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", result)
        # 存最终结果，返回调用方。
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(result, "assistant"))
        print(f"\n--- 任务完成 ---\n最终结果:\n{result}")
        return result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """通过框架的统一接口调用模型，并传入可选系统提示词。"""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.llm.invoke(messages, **kwargs)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("模型未返回有效文本")
        return response.strip()
