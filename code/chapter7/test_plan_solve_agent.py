"""直接运行，比较默认提示词与数学专用提示词的规划执行结果。"""

from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    from hello_agents.core.llm import HelloAgentsLLM
    from my_plan_solve_agent import MyPlanAndSolveAgent

    llm = HelloAgentsLLM()
    question = (
        "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。"
        "周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    )

    print("=== 示例一：默认提示词 ===")
    agent = MyPlanAndSolveAgent(name="我的规划执行助手", llm=llm)
    result = agent.run(question)
    print(f"\n最终结果: {result}")
    print(f"对话历史: {len(agent.get_history())} 条消息")

    math_prompts = {
        "planner": """
你是数学问题规划专家。请将数学问题分解为计算步骤，最后一步求最终答案：

问题: {question}

输出格式：
python
["计算步骤1", "计算步骤2", "求总和"]
""",
        "executor": """
你是数学计算专家。请计算当前步骤：

问题: {question}
计划: {plan}
历史: {history}
当前步骤: {current_step}

请只输出数值结果：
""",
    }
    print("\n=== 示例二：数学专用提示词 ===")
    math_agent = MyPlanAndSolveAgent(
        name="数学计算助手", llm=llm, custom_prompts=math_prompts
    )
    math_result = math_agent.run(question)
    print(f"\n数学专用Agent结果: {math_result}")
    print(f"对话历史: {len(math_agent.get_history())} 条消息")


if __name__ == "__main__":
    main()
