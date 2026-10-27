"""Commands suitable for manual execution or an operating-system scheduler."""

import argparse
import json
from datetime import date
from pathlib import Path

from demo import demo_records
from monitor import Monitor, Tools, report, today

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="商品口碑与风险监控 Agent")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "reviews.sqlite3")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="导入固定日期的模拟数据并执行规则分析")
    load = commands.add_parser("import", help="导入 CSV/JSONL，保留原始记录")
    load.add_argument("file", type=Path)
    analyze = commands.add_parser("analyze", help="分析待处理评论")
    analyze.add_argument("--model", action="store_true", help="使用 .env 配置的模型")
    analyze.add_argument("--limit", type=int, default=500)
    analyze.add_argument("--reanalyze", action="store_true", help="重新分析前 limit 条，替换原分析")
    for command in ("report", "check", "ask"):
        sub = commands.add_parser(command)
        sub.add_argument("--product", required=True)
        sub.add_argument("--as-of", type=date.fromisoformat, default=today())
        sub.add_argument("--days", type=int, default=7)
        if command == "report":
            sub.add_argument("--out", type=Path)
        if command == "ask":
            sub.add_argument("question")
    args = parser.parse_args()
    with Monitor(args.db) as db:
        if args.command == "demo":
            print(json.dumps(db.import_records(demo_records()), ensure_ascii=False))
            print(json.dumps(db.analyze_pending(limit=1000), ensure_ascii=False))
        elif args.command == "import":
            print(json.dumps(db.import_file(args.file.name, args.file.read_bytes()), ensure_ascii=False))
        elif args.command == "analyze":
            kwargs = {"limit": args.limit, "reanalyze": args.reanalyze}
            if args.model:
                from agent import ModelAnalyzer
                analyzer = ModelAnalyzer()
                kwargs.update(analyzer=analyzer, label=analyzer.label)
            result = db.analyze_pending(**kwargs)
            print(json.dumps(result, ensure_ascii=False))
            if result["failed"]:
                raise SystemExit(1)
        else:
            if args.product not in db.products():
                parser.error("商品不存在，请先导入数据或检查 --product")
            tools = Tools(db, args.product, args.as_of, args.days)
            if args.command == "check":
                print(json.dumps(tools.check_alerts(), ensure_ascii=False, indent=2))
            elif args.command == "ask":
                from agent import ask, render_answer
                print(render_answer(ask(tools, args.question)))
            else:
                content = report(tools)
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(content, encoding="utf-8")
                    print(str(args.out.resolve()))
                else:
                    print(content)


if __name__ == "__main__":
    main()
