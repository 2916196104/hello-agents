"""离线检查：不需要真实密钥，不发送网络请求。"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hello_agents import HelloAgentsLLM
from my_llm import MyLLM


class GeminiChecks(unittest.TestCase):
    def setUp(self):
        self.env = self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.client_factory = self.enterContext(patch("hello_agents.core.llm.OpenAI"))
        self.client = self.client_factory.return_value
        self.enterContext(patch("my_llm.OpenAI", return_value=self.client))

    def test_auto_detection_and_provider_specific_configuration(self):
        for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            with self.subTest(key_name=key_name), patch.dict(os.environ, {
                key_name: "test-key",
                "GEMINI_MODEL": "test-gemini-model",
                "GEMINI_BASE_URL": "https://gemini.example/v1/",
                "OPENAI_API_KEY": "other-key",
                "LLM_MODEL_ID": "other-model",
                "LLM_BASE_URL": "https://other.example/v1/",
            }, clear=True):
                llm = MyLLM()
                self.assertIsInstance(llm, HelloAgentsLLM)
                self.assertEqual((llm.provider, llm.api_key, llm.model, llm.base_url),
                                 ("gemini", "test-key", "test-gemini-model",
                                  "https://gemini.example/v1/"))

    def test_explicit_parameters_and_google_key_precedence(self):
        os.environ.update(GOOGLE_API_KEY="google-key", GEMINI_API_KEY="gemini-key",
                          GEMINI_MODEL="env-model")
        self.assertEqual(MyLLM(provider=None).api_key, "google-key")
        llm = MyLLM(provider="gemini", api_key="explicit-key", model="explicit-model",
                    base_url="https://explicit.example/v1/", temperature=0.2,
                    max_tokens=128, timeout=20)
        self.assertEqual((llm.api_key, llm.model, llm.base_url),
                         ("explicit-key", "explicit-model", "https://explicit.example/v1/"))
        self.assertEqual((llm.temperature, llm.max_tokens, llm.timeout), (0.2, 128, 20))
        self.client_factory.assert_called_with(api_key="explicit-key",
                                               base_url="https://explicit.example/v1/",
                                               timeout=20)

    def test_missing_key_is_rejected_before_client_creation(self):
        for key in (None, " "):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                MyLLM(provider="gemini", api_key=key)
        self.client_factory.assert_not_called()

    def test_existing_providers_are_preserved(self):
        os.environ.update(GEMINI_API_KEY="gemini-key", OPENAI_API_KEY="openai-key")
        self.assertEqual(MyLLM(provider="openai", model="test-model").provider, "openai")
        llm = MyLLM(provider="modelscope", api_key="ms-test", model="test-model")
        self.assertEqual((llm.provider, llm.api_key), ("modelscope", "ms-test"))
        del os.environ["GEMINI_API_KEY"]
        self.assertEqual(MyLLM().provider, "openai")

    def test_inherited_invocation_and_streaming(self):
        llm = MyLLM(provider="gemini", api_key="test-key")
        messages = [{"role": "user", "content": "你好"}]
        create = self.client.chat.completions.create
        create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content="你好！"))
        ])
        self.assertEqual(llm.invoke(messages), "你好！")
        self.assertEqual(create.call_args.kwargs["model"], llm.model)
        create.return_value = [SimpleNamespace(choices=[
            SimpleNamespace(delta=SimpleNamespace(content=text))
        ]) for text in ("你", "", "好")]
        self.assertEqual("".join(llm.stream_invoke(messages)), "你好")
        self.assertTrue(create.call_args.kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
