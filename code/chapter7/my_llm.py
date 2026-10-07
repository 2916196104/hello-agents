# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """通过继承扩展 Gemini，保留 ModelScope 和父类的其他供应商。"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # 与 Google SDK 一致：两个密钥同时存在时 GOOGLE_API_KEY 优先。
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if provider in (None, "auto"):
            provider = "gemini" if gemini_key else None

        if provider == "gemini":
            resolved_key = (api_key or gemini_key or "").strip()
            if not resolved_key:
                raise ValueError("请设置 GEMINI_API_KEY / GOOGLE_API_KEY，或传入 api_key。")

            # Gemini 的 OpenAI 兼容接口允许直接继承 invoke/think/stream_invoke。
            # 使用供应商专属配置，避免仓库的 LLM_* 指向其他平台。
            super().__init__(
                model=model or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash",
                api_key=resolved_key,
                base_url=base_url or os.getenv("GEMINI_BASE_URL")
                or "https://generativelanguage.googleapis.com/v1beta/openai/",
                provider="auto",
                **kwargs,
            )
            self.provider = "gemini"

        elif provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
            
            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"
            
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            
            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)
