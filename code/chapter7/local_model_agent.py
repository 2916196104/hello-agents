"""第七章：通过 HelloAgentsLLM 调用本地 Ollama 或 vLLM 服务。"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="调用已经启动的本地模型服务")
    parser.add_argument("--provider", choices=("ollama", "vllm"), default="ollama")
    parser.add_argument("--model", help="模型名称，必须与服务中已加载的模型一致")
    parser.add_argument("--base-url", help="本地模型服务的 OpenAI 兼容接口地址")
    args = parser.parse_args()

    from hello_agents import HelloAgentsLLM, SimpleAgent

    if args.provider == "ollama":
        model = args.model or "llama3"
        base_url = args.base_url or "http://localhost:11434/v1"
    else:
        model = args.model or "Qwen/Qwen1.5-0.5B-Chat"
        base_url = args.base_url or "http://localhost:8000/v1"

    # 显式传入本地配置；服务需要事先启动，模型需要事先下载。
    # 下列占位密钥用于本地未启用密钥验证的服务。
    llm = HelloAgentsLLM(
        provider=args.provider,
        model=model,
        base_url=base_url,
        api_key=args.provider,
        timeout=300,
        max_tokens=256,
    )
    agent = SimpleAgent(
        name="本地AI助手",
        llm=llm,
        system_prompt="你是一个有用的AI助手，请用中文简洁回答。",
    )

    print(f"正在通过 {args.provider} 调用模型：{model}")
    print(agent.run("你好！请用两句话介绍什么是 Agent。"))
    print(f"历史消息数: {len(agent.get_history())}")


if __name__ == "__main__":
    main()
