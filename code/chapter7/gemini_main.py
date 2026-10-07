"""7.2.1 扩展练习：从 .env 自动检测 Gemini，并调用 SimpleAgent。"""

from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    # 固定读取仓库根目录 .env，先加载配置，再导入框架。
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    from hello_agents import SimpleAgent
    from my_llm import MyLLM

    llm = MyLLM()  # 不指定 provider；自动检测 Gemini 的环境变量。
    if llm.provider != "gemini":
        raise ValueError("请先在仓库根目录 .env 中填写 GEMINI_API_KEY 或 GOOGLE_API_KEY。")

    agent = SimpleAgent("Gemini助手", llm, system_prompt="请使用中文简洁回答。")
    print(f"供应商：{llm.provider}；模型：{llm.model}")
    print(agent.run("你好，请用两句话介绍什么是 Agent。"))


if __name__ == "__main__":
    main()
