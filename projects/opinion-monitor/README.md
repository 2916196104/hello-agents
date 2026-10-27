# 声量 · 商品口碑与风险监控 Agent

导入商品评论，识别观点维度，比较问题变化，生成带原文依据的报告。

首版面向消费品牌／电商商品，使用 Python + SQLite + Streamlit。自带 **756 条明确标注的虚构评论**，覆盖三款商品、21 天，可以直接体验历史回放。

规则分析无需模型密钥，适合验证流程。模型观点提取和 Agent 工具调用需要配置自己的模型服务。当前尚未连接电商或社交平台的实时采集接口。

## 立即运行

在项目目录执行（Python 3.10+；Windows PowerShell）：

```powershell
cd E:\githubRecours\hello-agent\projects\opinion-monitor
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

本次开发已经安装了独立环境，当前工作区可以直接执行最后一行。打开终端显示的本机地址，默认是 `http://127.0.0.1:8501`。

1. 保持左侧“模拟演示”，选“演示-无线耳机”，截止日期 `2026-09-21`，窗口 7 天。
2. 查看每日评论、负向占比、增长最快的问题及商品对比。
3. 点击“检查并记录告警”，会出现“使用效果”和“客服售后”两个待核查问题。
4. 在“原文证据”中筛选问题，查看评论全文、观点片段和来源定位。
5. 将截止日期切换为 `2026-09-17`，可以看到问题集中出现前的情况。
6. 在“Agent 问答与报告”下载 Markdown 统计报告；配置模型后可以提问。

模拟数据中，最后三天耳机的“断连／杂音”和“客服回复慢”集中增加。在截至 09-21 的 7 天窗口，使用效果问题从上期 0/84 增至本期 24/84，增加 **28.57 个百分点**。这是合成场景，不代表真实品牌。

## 导入自己的评论

切换左侧“我的数据”，上传 CSV 或 JSONL，再点击“执行观点分析”。上传文件须为 UTF-8，支持 BOM；单文件最多 10 MB、10,000 条。建议先取 3—5 款商品的少量评论。

标准格式参见 [sample_reviews.csv](sample_reviews.csv)。此文件也是虚构示例。建议替换为商家授权导出的评价、售后记录，或已准备好的公开数据集。

| 字段 | 要求 | 说明 |
| --- | --- | --- |
| `source` | 建议填写 | 来源名称；缺省为 `import`，同一来源保持名称稳定 |
| `source_record_id` | 必填 | 来源内唯一 ID，用于重复导入去重 |
| `product_id` | 必填 | 商品标识；不同来源的同款商品需提前统一 |
| `text` | 必填 | 评论正文，最多 12,000 字符 |
| `rating` | 可选 | 原始 1—5 星评分；留空表示未知，评分不直接决定文本情感 |
| `published_at` | 必填 | ISO 日期或时间；无时区时按 Asia/Taipei 处理 |
| `evidence_ref` | 建议填写 | 原文链接或可回查定位；缺省为文件名与记录位置 |

JSONL 每行一个对象，例如：

```json
{"source":"shop-export","source_record_id":"review-001","product_id":"headphones-001","text":"音质清晰，但是物流慢。","rating":3,"published_at":"2026-09-20T10:30:00+08:00","evidence_ref":"export-20260920.jsonl:line=1"}
```

也支持 Amazon Reviews 2023 评论 JSONL 中的 `parent_asin`、`text`、`rating`、`timestamp`（毫秒）和 `user_id`：自动转换日期，生成稳定记录指纹。先在外部筛选小文件再导入，不会在本项目中下载整个数据集。数据集位置是原文依据；不会伪造公开评论链接。

同一 `source + source_record_id` 再次导入时跳过。若 ID 相同但正文、商品、评分或发布时间变化，会计为冲突并保留旧记录，避免旧分析被错误套用。文件中任意一条记录不合法，整批导入回滚。

原始 JSON、正文、发布时间、获取时间和来源定位保存在 SQLite。请保留原导入文件，便于根据记录位置回查。

## 启用模型分析和 Agent

复制本项目配置：

```powershell
Copy-Item .env.example .env
```

在 `.env` 填入 `MONITOR_API_KEY`、`MONITOR_MODEL`；使用其他兼容服务时修改 `MONITOR_BASE_URL`。默认地址为 OpenAI API。**需要所选模型同时支持 Chat Completions、function calling 和 JSON object 输出。** 不会读取仓库根目录的 `.env`。

模型协议参考 [OpenAI 官方 function calling 文档](https://developers.openai.com/api/docs/guides/function-calling)。本项目使用其中的 Chat Completions 接入方式；仅支持 Responses 工具调用的模型需要另行适配。

观点分析会将每条评论拆为多个“维度＋情感＋原文片段”。固定维度为商品质量、使用效果、价格、包装、物流、客服售后、其他；情感为正向、负向、中性、未知。使用 Pydantic 校验结构，并验证片段逐字存在于原文。失败记录保留待分析状态，成功记录逐条保存。

Agent 通过白名单调用四个只读工具：

| 工具 | 功能 |
| --- | --- |
| `get_trend` | 每日评论量、负向占比、待分析及未知数量 |
| `get_top_issues` | 本期与前一个等长窗口的问题数量及占比变化 |
| `get_evidence` | 原文、观点和可回查的证据编号 |
| `compare_products` | 同期各商品的统计对比 |

可以提问：“这款商品最近哪些问题增长最快？给出原文依据和优先核查建议。”商品、截止日期与窗口以界面选择为准；每次提问独立执行。最多 7 轮，每轮最多 6 次工具调用，不提供任意 SQL 或代码执行。

Agent 最终引用只能使用本次查到的证据编号。界面同时展示原始工具调用结果，便于人工核对。引用存在不代表模型解读一定正确，原因和处置建议仍需验证。

评论文本会发送给你配置的模型服务；规则模式与统计报告在本机执行。模型调用按条发生，建议先测试 10—20 条再扩大批量。

## 统计与告警口径

- 日期按评论发布时间归入 Asia/Taipei，采集时间单独保存。查询排除截止日之后的评论。
- 一条评论可含多个情感及维度；出现任意负向观点就计为一条负向评论。
- 同一评论在同一问题维度只计一次；各问题的数量相加可能大于负向评论数。
- 占比分母为已分析评论。待分析和全部观点未知的数量独立展示，未知不等于满意。
- 本期为所选窗口，上期为紧邻本期的同长度窗口。按占比变化的百分点排序，避免只按绝对数量判断。
- 告警要求两期分别至少 20 条、全部分析完成且分析方法一致；该问题本期至少 5 条、占比增长至少 15 个百分点。
- 同一商品同一问题冷却 3 天；重复检查不重复入库。历史回放不展示未来日期的告警。

告警只是“待核查”记录，保存在本机数据库。没有配置邮件、企业微信或其他外部发送渠道。

## 使用命令行持续处理

命令均在本项目目录执行，`--db` 是子命令之前的全局参数。

```powershell
# 生成独立演示库并分析，重复运行会去重
.\.venv\Scripts\python.exe cli.py --db data/demo.sqlite3 demo

# 导入真实文件，分析待处理评论（不加 --model 时为规则模式）
.\.venv\Scripts\python.exe cli.py import sample_reviews.csv
.\.venv\Scripts\python.exe cli.py analyze --model --limit 20

# 检查告警并输出带原文的报告
.\.venv\Scripts\python.exe cli.py --db data/demo.sqlite3 check --product "演示-无线耳机" --as-of 2026-09-21
.\.venv\Scripts\python.exe cli.py --db data/demo.sqlite3 report --product "演示-无线耳机" --as-of 2026-09-21 --out reports/demo-report.md

# 模型问答
.\.venv\Scripts\python.exe cli.py --db data/demo.sqlite3 ask --product "演示-无线耳机" --as-of 2026-09-21 "哪些问题增长最快？"
```

可以在 Windows 任务计划程序中按顺序运行 `import → analyze → check → report`，设置工作目录为本项目。未传 `--as-of` 时，按当前 Asia/Taipei 日期处理。当前尚未安装系统定时任务；持续采集需由外部数据源持续更新导入文件。

分析数量超过批次上限时可重复执行 `analyze`，只处理待分析记录。`--reanalyze` 会替换库中最早的前 `--limit` 条分析，适合对小批次比较规则和模型效果；它不带翻页功能。

## 文件结构与验证

| 文件 | 职责 |
| --- | --- |
| `monitor.py` | 导入、清洗、规则分析、SQLite、统计、证据、告警与固定报告 |
| `agent.py` | 可选模型观点分析、工具调用循环、引用校验 |
| `app.py` | Streamlit 看板与交互 |
| `cli.py` | 批量导入、分析、检查和报告命令 |
| `demo.py` | 可重复生成的虚构评论 |
| `tests/` | 统计、导入、告警、历史回放和模拟模型调用测试 |

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

模拟数据和真实导入分别使用 `data/demo.sqlite3` 与 `data/reviews.sqlite3`，不会在界面中混合。数据库、密钥和生成报告已加入项目 `.gitignore`。

## 当前边界

这是可运行的本地首版，尚未接入真实商家账号、平台接口、采集失败告警、语义聚类或外部通知。规则词表仅供演示，不适合直接评价真实品牌风险。模型准确性仍需要人工标注集评估；当前没有完成真实模型服务的联网验收。

历史回放限制的是评论时间范围；分析标签可能由当前模型生成，不代表当时已经获得的预测结果。告警阈值只是首版规则，需要用真实业务样本校准。
