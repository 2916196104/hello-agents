from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

import ast
import math
import operator
import os
from serpapi import SerpApiClient
from typing import Dict, Any

def calculator(expression: str) -> str:
    """只计算数字、括号和四则运算，不执行输入中的代码。"""
    expression = expression.strip().replace("×", "*").replace("÷", "/")
    if not expression or len(expression) > 200:
        return "错误：请输入不超过200个字符的算术表达式。"
    operators = {ast.Add: operator.add, ast.Sub: operator.sub,
                 ast.Mult: operator.mul, ast.Div: operator.truediv}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("只支持数字、括号以及 +、-、*、/ 运算")

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("表达式过于复杂，请拆分计算")
        # ponytail: 使用浮点除法，金额或精确小数场景改用 Decimal。
        result = evaluate(tree.body)
        if not math.isfinite(result):
            raise ValueError("计算结果超出支持的数值范围")
        return str(result)
    except ZeroDivisionError:
        return "错误：除数不能为0。"
    except (SyntaxError, ValueError, OverflowError) as error:
        return f"错误：无法计算表达式（{error}）。"

def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"
    
from typing import Dict, Any

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")
        
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def executeTool(self, name: str, tool_input: str) -> tuple[bool, str]:
        """统一检查并执行工具，返回是否成功及可反馈给模型的结果。"""
        tool = self.getTool(name)
        if tool is None:
            return False, f"错误：未找到工具 '{name}'。请从已注册工具中选择，名称区分大小写。"
        if not isinstance(tool_input, str) or not tool_input.strip():
            return False, "错误：工具参数必须是非空字符串，请根据工具说明补充参数。"
        try:
            result = tool(tool_input)
        except Exception as error:
            return False, f"错误：工具 '{name}' 执行失败（{type(error).__name__}: {error}）。"
        if result is None or not str(result).strip():
            return False, f"错误：工具 '{name}' 未返回有效结果。"
        observation = str(result)
        # ponytail: 当前工具用文本表示错误，新增工具较多时改为结构化结果。
        failed = observation.lstrip().startswith(("错误：", "搜索时发生错误:"))
        return not failed, observation

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    success, observation = toolExecutor.executeTool(tool_name, tool_input)
    print("--- 观察 (Observation) ---")
    print(observation)
