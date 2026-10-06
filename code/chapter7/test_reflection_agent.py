"""直接运行本文件，体验通用写作和自定义代码生成的反思流程。"""

from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    # 固定读取项目根目录的 .env，从不同工作目录启动也能找到配置。
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    # 加载配置后再导入框架；导入本测试文件时不会创建模型或发送请求。
    from hello_agents import HelloAgentsLLM
    from my_reflection_agent import MyReflectionAgent

    llm = HelloAgentsLLM()

    print("=== 示例一：使用默认提示词写文章 ===")
    general_agent = MyReflectionAgent(
        name="我的反思助手",
        llm=llm,
        max_iterations=2,
    )
    general_agent.run("写一篇关于人工智能发展历程的简短文章")
    print(f"对话历史: {len(general_agent.get_history())} 条消息\n")

    print("=== 示例二：使用自定义提示词生成代码 ===")
    code_prompts = {
        "initial": (
            "你是Python专家，请根据任务编写函数：{task}\n"
            "代码应包含函数签名和文档字符串，请直接输出完整代码。"
        ),
        "reflect": (
            "请审查代码的正确性、算法效率和边界情况。\n"
            "任务：{task}\n代码：{content}\n"
            "请提出具体改进建议；如果已经满足要求，请仅回答‘无需改进’。"
        ),
        "refine": (
            "请根据反馈优化代码，并直接输出改进后的完整代码。\n"
            "任务：{task}\n上一轮代码：{last_attempt}\n反馈：{feedback}"
        ),
    }
    code_agent = MyReflectionAgent(
        name="我的代码生成助手",
        llm=llm,
        max_iterations=2,
        custom_prompts=code_prompts,
    )
    code_agent.run(
        "编写 find_primes(n: int) -> list[int]，返回 1 到 n（含 n）之间"
        "的所有素数；n 小于 2 时返回空列表。"
    )
    print(f"对话历史: {len(code_agent.get_history())} 条消息")


if __name__ == "__main__":
    main()
