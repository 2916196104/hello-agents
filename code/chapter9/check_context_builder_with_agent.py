"""离线验证字符串响应和 ModelScope 的代理排除设置。"""

from contextlib import redirect_stdout
import importlib.util
import inspect
from io import StringIO
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from hello_agents import HelloAgentsLLM, SimpleAgent


spec = importlib.util.spec_from_file_location(
    "context_agent_example", Path(__file__).with_name("02_context_builder_with_agent.py")
)
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
# 动态加载文件没有可供 IDE 索引的模块名，通过其全局字典访问成员。
app_globals = vars(app)
context_agent_class = app_globals["ContextAwareAgent"]


class ContextAgentChecks(unittest.TestCase):
    def test_string_responses_and_conversation_history(self):
        llm = Mock(spec=HelloAgentsLLM)
        llm.invoke.side_effect = ["第一轮回答", "第二轮回答"]
        builder = Mock()
        builder.return_value.build.return_value = "优化后的上下文"
        with patch.dict(app_globals, {"ContextBuilder": builder}):
            agent = context_agent_class(name="测试助手", llm=llm)
            self.assertEqual(agent.run(input_text="第一个问题", max_tool_iterations=1, temperature=0.2), "第一轮回答")
            self.assertEqual(llm.invoke.call_args.kwargs, {"temperature": 0.2})
            self.assertEqual(agent.run("第二个问题"), "第二轮回答")
        self.assertEqual(
            [message.role for message in agent.conversation_history],
            ["user", "assistant", "user", "assistant"],
        )
        self.assertEqual(agent.conversation_history[-1].content, "第二轮回答")

    def test_run_signature_matches_base_class(self):
        base_parameters = inspect.signature(SimpleAgent.run).parameters
        own_parameters = inspect.signature(context_agent_class.run).parameters
        self.assertEqual(list(own_parameters), list(base_parameters))
        for name, base_parameter in base_parameters.items():
            self.assertEqual(own_parameters[name].kind, base_parameter.kind)
            self.assertEqual(own_parameters[name].default, base_parameter.default)

    def test_proxy_is_bypassed_only_for_modelscope(self):
        for host in ("api-inference.modelscope.cn", "api.example.com"):
            with self.subTest(host=host), patch.dict(os.environ, {
                "LLM_BASE_URL": f"https://{host}/v1",
                "HTTP_PROXY": "http://127.0.0.1:7897",
                "HTTPS_PROXY": "http://127.0.0.1:7897",
                "NO_PROXY": "localhost,127.0.0.1",
                "no_proxy": "localhost,127.0.0.1",
            }), patch.dict(app_globals, {
                "HelloAgentsLLM": Mock(), "ContextAwareAgent": Mock(),
            }), redirect_stdout(StringIO()):
                app_globals["main"]()
                expected = "localhost,127.0.0.1"
                if host == "api-inference.modelscope.cn":
                    expected += ",api-inference.modelscope.cn"
                self.assertEqual(os.environ["NO_PROXY"], expected)
                self.assertEqual(os.environ["no_proxy"], expected)
                self.assertEqual(os.environ["HTTPS_PROXY"], "http://127.0.0.1:7897")


if __name__ == "__main__":
    unittest.main()
