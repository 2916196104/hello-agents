"""第七章：生成初稿，再通过反思和改进迭代完成任务。"""

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
}

class MyReflectionAgent(ReflectionAgent):
    """支持通用任务和自定义提示词的反思智能体。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        if max_iterations < 0:
            raise ValueError("max_iterations 不能小于 0")

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
        """生成初稿，最多执行 max_iterations 轮反思与改进。"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 当前任务的执行轨迹重新开始，对话历史由 Agent 基类独立保存。
        self.memory.records.clear()
        # 根据原始任务生成初稿。
        initial_prompt = self.prompts["initial"].format(task=input_text)
        result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", result)
       # 把当前版本交给模型审查。
        for iteration in range(self.max_iterations):
            print(f"\n--- 第 {iteration + 1}/{self.max_iterations} 轮反思 ---")
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text, content=result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # 仅将明确的结束标记视为通过，避免误判引用标记的批评意见。
            verdict = feedback.strip().strip('"\'“”‘’').rstrip("。.!！").casefold()
            if verdict in ("无需改进", "no need for improvement"):
                print("\n✅ 无需改进，任务完成。")
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
        return self.llm.invoke(messages, **kwargs) or ""
