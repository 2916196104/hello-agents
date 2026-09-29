"""离线检查最终答案解析；运行 python check_finish.py，不调用外部 API。"""
import io
import os
import runpy
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def check(actions, expected):
    client = Mock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=f"Thought: 整理结果。\nAction: {action}"
        ))])
        for action in actions
    ]
    env = {"OPENAI_API_KEY": "offline-test", "OPENAI_BASE_URL": "https://example.invalid/v1",
           "MODEL_NAME": "offline-test"}
    with patch.dict(os.environ, env), patch("openai.OpenAI", return_value=client), \
            patch("dotenv.load_dotenv"), patch("builtins.input", side_effect=["请推荐景点", "退出"]), \
            redirect_stdout(io.StringIO()) as output:
        runpy.run_path(str(Path(__file__).with_name("FirstAgentTest.py")), run_name="__main__")
    assert f"任务完成，最终答案: {expected}" in output.getvalue()
    assert client.chat.completions.create.call_count == len(actions)
    if len(actions) > 1:
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "结束格式无效" in prompt


if __name__ == "__main__":
    check(["Finish[推荐西湖]"], "推荐西湖")
    answer = "杭州小阵雨，24℃。\n\n1. 西湖\n2. 杭州博物馆\n\n提示：携带雨具。"
    check([f"Finish[{answer}]"], answer)
    check(["Finish[缺少右括号", "Finish[已修正]"], "已修正")
    check(["Finish[]", "Finish[已补充答案]"], "已补充答案")
    check(["Finish[参考资料：\nObservation: [原始说明]\n结束]"], "参考资料：\nObservation: [原始说明]\n结束")
    print("Finish 解析检查通过：单行、多行、格式错误后的恢复。")
