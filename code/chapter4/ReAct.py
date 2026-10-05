import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search, calculator

# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

遇到数学计算时，必须先调用 Calculator 获取结果，再根据 Observation 给出最终答案。
工具调用失败时，根据 Observation 的错误原因修正工具名称或参数；不要把错误信息当作成功结果。
如果缺少必要信息且无法继续，可用 Finish[无法完成：请补充所需信息] 如实说明。

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5, max_tool_failures: int = 3):
        if not isinstance(max_tool_failures, int) or max_tool_failures < 1:
            raise ValueError("max_tool_failures 必须是正整数。")
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.max_tool_failures = max_tool_failures
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0
        consecutive_failures = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                return None

            thought, action = self._parse_output(response_text)
            if thought: print(f"🤔 思考: {thought}")
            tool_name, tool_input = self._parse_action(action or "")
            if tool_name == "Finish" and tool_input.strip():
                # 如果是Finish指令，提取最终答案并结束
                final_answer = tool_input
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            if not tool_name or tool_name == "Finish":
                success, observation = False, "错误：Action 格式无效或最终答案为空。"
            else:
                print(f"🎬 行动: {tool_name}[{tool_input}]")
                success, observation = self.tool_executor.executeTool(tool_name, tool_input)

            consecutive_failures = 0 if success else consecutive_failures + 1
            if not success:
                observation += (
                    f"\n纠正提示：连续失败 {consecutive_failures}/{self.max_tool_failures} 次。"
                    "检查工具用途、名称和参数；下一轮只输出一个 Action: 工具名[非空参数]。"
                    f"\n可用工具及参数说明：\n{tools_desc}"
                )
                if consecutive_failures >= 2:
                    observation += "\n请重新选择合适工具或修改参数，不要重复失败的调用；缺少信息时请明确说明。"
            
            print(f"👀 观察: {observation}")
            # 保存执行记录，供下一轮使用
            self.history.append(f"Action: {action or '（缺失）'}")
            self.history.append(f"Observation: {observation}")
            if consecutive_failures >= self.max_tool_failures:
                message = f"工具调用已连续失败 {consecutive_failures} 次，停止执行；请检查工具配置或补充参数后重试。"
                self.history.append(f"Observation: {message}")
                print(message)
                return None

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.fullmatch(r"(\w+)\[(.*)\]", action_text.strip(), re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""

if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    tool_executor.registerTool("Calculator", "计算数学表达式，支持括号、小数、负数和四则运算（+、-、*、/，也支持×、÷）。输入仅包含表达式，例如 Calculator[(123 + 456) * 789 / 12]。", calculator)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "计算 (123 + 456) × 789 / 12 = ?"
    agent.run(question)
