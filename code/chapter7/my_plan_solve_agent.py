"""第七章：先生成结构化计划，再携带历史结果逐步执行。"""

import ast
import re
from typing import Optional

from hello_agents import Config, HelloAgentsLLM, Message, PlanAndSolveAgent
from hello_agents.agents.plan_solve_agent import Executor as BaseExecutor
from hello_agents.agents.plan_solve_agent import Planner as BasePlanner


DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。请将复杂问题分解成有逻辑顺序、可执行的子任务。
最后一个步骤必须给出原始问题的最终答案。
你的输出必须是一个Python列表，其中每个元素都是描述子任务的字符串。

问题: {question}

请严格按照以下格式输出计划，不要添加解释：
```python
["步骤1", "步骤2", "给出最终答案"]
```
"""

DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。请严格按照计划，一步步解决问题。
请使用已经完成的步骤与结果，专注于当前步骤，仅输出该步骤的最终答案。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对当前步骤的回答：
"""


def _get_llm_response(
    llm: HelloAgentsLLM,
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> str:
    """统一模型调用，并拒绝空响应，避免后续步骤使用缺失的结果。"""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try: # 发出模型请求，取得回答
        response = llm.invoke(messages, **kwargs)
    except Exception as exc:
        raise RuntimeError("模型调用失败，请检查模型配置和网络连接。") from exc
    if not isinstance(response, str) or not response.strip():
        raise ValueError("模型未返回有效文本。")
    return response.strip()


class Planner(BasePlanner):
    """生成计划，并将模型文本转换为经过校验的步骤列表。"""

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        prompt_template: str = DEFAULT_PLANNER_PROMPT,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(llm_client, prompt_template)
        self.system_prompt = system_prompt

    @staticmethod
    def _parse_plan(response: str) -> list[str]:
        text = response.strip()
        # 支持 python/json/无语言标记的代码块，以及直接输出的列表。
        code_block = re.search(
            r"```(?:python|json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE
        )
        if code_block:
            text = code_block.group(1).strip()
        else:
            lines = text.splitlines()
            if lines and lines[0].strip().lower() in ("python", "json"):
                text = "\n".join(lines[1:]).strip()

        try:
            # 只解析字面量，不执行模型返回的代码。
            plan = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("计划必须是Python字符串列表，例如 ['步骤1', '步骤2']。") from exc

        if not isinstance(plan, list) or not plan:
            raise ValueError("计划必须是非空列表。")
        if any(not isinstance(step, str) or not step.strip() for step in plan):
            raise ValueError("计划中的每一步都必须是非空字符串。")
        return [step.strip() for step in plan]

    def plan(self, question: str, **kwargs) -> list[str]:
        try: # 将原始问题填入规划提示词
            prompt = self.prompt_template.format(question=question)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError("规划器提示词格式错误，请使用 {question} 占位符。") from exc

        print("\n--- 正在生成计划 ---")
        # 请求模型生成计划文本
        response = _get_llm_response(
            self.llm_client, prompt, self.system_prompt, **kwargs
        )
        # 将文本转换为经过检查的 Python 列表
        plan = self._parse_plan(response)
        for index, step in enumerate(plan, 1):
            print(f"{index}. {step}")
        return plan


class Executor(BaseExecutor):
    """逐步调用模型，把前面步骤的结果加入后续步骤的上下文。"""

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        prompt_template: str = DEFAULT_EXECUTOR_PROMPT,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(llm_client, prompt_template)
        self.system_prompt = system_prompt

    def execute(self, question: str, plan: list[str], **kwargs) -> str:
        if not plan:
            raise ValueError("没有可执行的计划。")

        history: list[str] = []
        final_answer = ""
        for index, step in enumerate(plan, 1):
            print(f"\n--- 执行步骤 {index}/{len(plan)}: {step} ---")
            try:
                prompt = self.prompt_template.format(
                    question=question,
                    plan=plan,
                    history="\n\n".join(history) if history else "无",
                    current_step=step,
                )
                final_answer = _get_llm_response(
                    self.llm_client, prompt, self.system_prompt, **kwargs
                )
            except (KeyError, IndexError, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"第 {index} 步执行失败：{exc}") from exc

            history.append(f"步骤 {index}: {step}\n结果: {final_answer}")
            print(f"结果: {final_answer}")
        return final_answer


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """支持自定义规划、执行提示词及失败反馈的 Plan-and-Solve 智能体。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        prompts = {
            "planner": DEFAULT_PLANNER_PROMPT,
            "executor": DEFAULT_EXECUTOR_PROMPT,
        }
        if custom_prompts:
            prompts.update(custom_prompts)
        for key in ("planner", "executor"):
            if not isinstance(prompts[key], str) or not prompts[key].strip():
                raise ValueError(f"{key} 提示词必须是非空字符串。")

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            custom_prompts=prompts,
        )
        self.planner = Planner(llm, prompts["planner"], system_prompt)
        self.executor = Executor(llm, prompts["executor"], system_prompt)

    def run(self, input_text: str, **kwargs) -> str:
        """返回最后一步的答案；规划或执行失败时返回明确的终止原因。"""
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        try:
            if not input_text.strip():
                raise ValueError("问题不能为空。")
            plan = self.planner.plan(input_text, **kwargs)
        except (ValueError, RuntimeError) as exc:
            final_answer = f"规划失败，任务终止：{exc}"
        else:
            try:
                final_answer = self.executor.execute(input_text, plan, **kwargs)
            except (ValueError, RuntimeError) as exc:
                final_answer = f"执行失败，任务终止：{exc}"

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
