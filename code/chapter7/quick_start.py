"""第七章入门：体验基础对话与历史记录。"""

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, SimpleAgent


def main() -> None:
    # 自动查找 .env；本项目会读取根目录中已有的配置。
    load_dotenv()

    llm = HelloAgentsLLM()
    agent = SimpleAgent(
        name="AI助手",
        llm=llm,
        system_prompt="你是一个有用的AI助手，请用中文简洁回答。",
    )

    print("第一次对话：")
    response = agent.run("你好！请介绍一下自己")
    print(response)

    # 这里仍是让模型回答，并没有调用计算器工具。
    print("\n第二次对话：")
    response = agent.run("请帮我计算 2 + 3 * 4")
    print(response)

    # 每次对话保存一条用户消息和一条助手消息；两轮共四条。
    print(f"\n历史消息数: {len(agent.get_history())}")


if __name__ == "__main__":
    main()
