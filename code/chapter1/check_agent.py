"""离线检查记忆、工具选择、反馈传递和对话控制：python check_agent.py。"""
import io
import os
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

with patch("dotenv.load_dotenv"):
    import FirstAgentTest as agent


def check():
    history = []
    llm = Mock()
    llm.generate.side_effect = [
        'Action: get_weather(city="杭州")',
        'Action: get_attraction(city="杭州", weather="小雨", preferences="带老人，历史文化")',
        'Action: Finish[推荐杭州博物馆，符合历史文化兴趣。]',
        'Action: get_attraction(city="杭州", weather="小雨", preferences="带老人，历史文化，少走路", exclude="博物馆")',
        'Action: Finish[已排除博物馆，改为茶馆休息；无障碍条件待确认。]',
        'Action: get_weekday(date_str="2026-10-01")',
        'Action: Finish[2026-10-01是星期四。]',
    ]
    weather = Mock(return_value="杭州：小雨，24℃")
    search = Mock()
    search.search.side_effect = [{"answer": "杭州博物馆"}, {"answer": "茶馆；无障碍条件待确认"}]
    with patch.dict(os.environ, {"TAVILY_API_KEY": "offline-test"}), \
            patch.dict(agent.available_tools, {"get_weather": weather}), \
            patch.object(agent, "TavilyClient", return_value=search):
        agent.run_turn(llm, "查询今天杭州天气；带老人，喜欢历史文化，推荐景点。", history)
        answer = agent.run_turn(llm, "不要博物馆，尽量少走路。", history)
        assert "排除博物馆" in answer
        feedback_prompt = llm.generate.call_args_list[3].args[0]
        assert all(text in feedback_prompt for text in ("带老人", "历史文化", "杭州博物馆", "不要博物馆", "少走路"))
        query = search.search.call_args.kwargs["query"]
        assert "用户要求：带老人，历史文化，少走路" in query
        assert "请排除这些景点或类型：博物馆" in query
        assert search.search.call_count == 2
        weather.assert_called_once_with(city="杭州")
        agent.run_turn(llm, "另查2026年10月1日是星期几。", history)
        assert "Observation: 2026-10-01是星期四" in history
        assert search.search.call_count == 2  # 查星期不应执行搜索。

    # 模型生成的调用只能访问注册工具，不能执行表达式或偷偷丢弃错误参数。
    assert agent.execute_tool("get_weekday(date_str='2026-10-01')") == "2026-10-01是星期四"
    for action in ("get_weekday()", "get_weekday(date_str=42)", "unknown()",
                   'get_weekday(**{"date_str": "2026-10-01"})',
                   'get_weekday(date_str="2026-10-01", extra="x")',
                   'get_weekday(date_str="a", date_str="b")', "get_weekday("):
        assert agent.execute_tool(action).startswith("错误："), action
    with patch("os.system") as system:
        assert agent.execute_tool('__import__("os").system("echo unexpected")').startswith("错误：")
        system.assert_not_called()

    repair = Mock()
    repair.generate.side_effect = ['Action: get_weekday()', 'Action: get_weekday(date_str="2026-10-01")', 'Action: Finish[星期四]']
    assert agent.run_turn(repair, "查星期", []) == "星期四"
    assert "工具调用格式无效" in repair.generate.call_args_list[1].args[0]
    failed = Mock()
    failed.generate.return_value = None
    assert "调用失败" in agent.run_turn(failed, "你好", [])
    failed.generate.assert_called_once()
    limited = Mock()
    limited.generate.return_value = "没有Action"
    assert "尚未完成" in agent.run_turn(limited, "测试", [], max_steps=2)
    assert limited.generate.call_count == 2

    # 实际终端入口：空输入不调用模型；Finish回到输入；记忆可查看和清除。
    console_llm = Mock()
    console_llm.generate.side_effect = ['Action: Finish[已记住喜欢历史文化。]', 'Action: Finish[请提供城市。]']
    inputs = ["", "我喜欢历史文化", "查看记忆", "清空记忆", "查看记忆", "推荐景点", "退出"]
    env = {"OPENAI_API_KEY": "offline-test", "OPENAI_BASE_URL": "https://example.invalid/v1", "MODEL_NAME": "offline-test"}
    with patch.dict(os.environ, env), patch.object(agent, "OpenAICompatibleClient", return_value=console_llm), \
            patch("builtins.input", side_effect=inputs), redirect_stdout(io.StringIO()) as output:
        agent.main()
    assert "用户请求: 我喜欢历史文化" in output.getvalue()
    assert "会话记忆已清空" in output.getvalue() and "暂无会话记忆" in output.getvalue()
    assert console_llm.generate.call_count == 2
    assert "历史文化" not in console_llm.generate.call_args.args[0]
    with patch.dict(os.environ, env), patch.object(agent, "OpenAICompatibleClient"), \
            patch("builtins.input", side_effect=EOFError):
        agent.main()


if __name__ == "__main__":
    with redirect_stdout(io.StringIO()):
        check()
    print("会话检查通过：历史记忆、工具分派、偏好和排除项传递、清空记忆、异常与轮数上限。")
