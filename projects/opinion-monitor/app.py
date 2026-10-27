from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from agent import ModelAnalyzer, ask, configured, render_answer
from demo import demo_records
from monitor import ASPECTS, Monitor, Tools, report, today

ROOT = Path(__file__).parent
st.set_page_config(page_title="声量 · 商品口碑监控", page_icon="📡", layout="wide")


def main():
    st.title("📡 声量 · 商品口碑监控")
    st.caption("从评论发现问题变化，让每个结论都有原文依据。")
    if "notice" in st.session_state:
        st.success(st.session_state.pop("notice"))
    with st.sidebar:
        st.header("监控范围")
        mode = st.radio("数据空间", ["模拟演示", "我的数据"])
        is_demo = mode == "模拟演示"
        db_path = ROOT / "data" / ("demo.sqlite3" if is_demo else "reviews.sqlite3")
        if is_demo:
            st.info("模拟数据：2026-09-01 至 09-21，3 款虚构商品。可将截止日期调到 09-18 前后观察变化。")
        else:
            st.caption("导入商家授权导出的评论，或准备好的公开数据集。")
        st.caption(f"模型问答：{'已配置' if configured() else '未配置'}")

    with Monitor(db_path) as db:
        if is_demo:
            db.import_records(demo_records())
            db.analyze_pending(limit=1000)
        products = db.products()
        if not products:
            st.info("先在下方导入评论。导入后选择商品并分析。")
            import_panel(db)
            return
        with st.sidebar:
            product = st.selectbox("商品", products, index=products.index("演示-无线耳机") if is_demo else 0)
            as_of = st.date_input("统计截止日期", date(2026, 9, 21) if is_demo else today(), max_value=today())
            days = st.selectbox("分析窗口（天）", [3, 7, 14, 30], index=1)
            st.caption("上期为紧邻本期、长度相同的窗口。时间按 Asia/Taipei 归日。")
        tools = Tools(db, product, as_of, days)
        trend = tools.get_trend()
        summary = trend["summary"]
        columns = st.columns(4)
        columns[0].metric("评论量", summary["total"])
        columns[1].metric("已分析", summary["analyzed"])
        columns[2].metric("含负向观点", summary["negative"])
        columns[3].metric("负向评论占比", f"{summary['negative_rate']:.1%}" if summary["negative_rate"] is not None else "—")
        st.caption(f"统计期 {trend['start']} 至 {trend['end']} · 待分析 {summary['pending']} 条 · 全部观点未知 {summary['unknown']} 条。占比分母为已分析评论，每条评论可涉及多个问题。")
        if summary["pending"]:
            st.warning("存在待分析评论，当前比例可能偏离完整样本；请先完成分析。该窗口暂不触发风险告警。")
        if "rules-v1" in summary["analyzers"]:
            st.warning("当前包含规则分析结果，仅供流程演示；否定、反讽和隐含观点可能识别错误。")
        if len(summary["analyzers"]) > 1:
            st.warning("当前混合了不同分析方法，比较结果时请注意口径差异。")

        dashboard, evidence_tab, agent_tab, data_tab = st.tabs(["趋势与告警", "原文证据", "Agent 问答与报告", "导入与分析"])
        with dashboard:
            st.subheader("每日讨论变化")
            daily = pd.DataFrame(trend["daily"]).set_index("day")
            st.line_chart(daily[["total", "negative"]].rename(columns={"total": "评论量", "negative": "含负向观点"}))
            st.line_chart((daily[["negative_rate"]] * 100).rename(columns={"negative_rate": "负向占比（%）"}))
            st.caption("空白比例表示当日没有已分析评论；评论量为零也可能是数据未采集到。")
            st.subheader("哪些问题增长最快")
            issues = tools.get_top_issues()
            if issues["issues"]:
                frame = pd.DataFrame(issues["issues"]).rename(columns={"aspect": "问题", "count": "本期数量", "previous_count": "上期数量", "rate": "本期占比", "previous_rate": "上期占比", "change_pp": "变化（百分点）"})
                st.dataframe(frame, hide_index=True, width="stretch",
                             column_config={"本期占比": st.column_config.NumberColumn(format="percent"),
                                            "上期占比": st.column_config.NumberColumn(format="percent")})
            else:
                st.info("当前窗口没有识别到负向问题。")
            st.caption(f"本期／上期已分析样本：{issues['current_samples']}／{issues['previous_samples']}。")
            st.subheader("商品对比")
            comparison = pd.DataFrame(tools.compare_products())
            st.bar_chart(comparison.set_index("product_id")[["negative"]].rename(columns={"negative": "含负向观点的评论"}))
            st.subheader("待核查告警")
            st.caption("两期各至少 20 条、全部完成且分析方法一致；问题至少 5 条、占比增加至少 15 个百分点；同商品同问题冷却 3 天。告警保存在本机。")
            if st.button("检查并记录告警"):
                added = tools.check_alerts()
                st.success(f"新增 {len(added)} 条告警。未触发可能是变化或样本不足、分析未完成或方法混合，或处于冷却期。")
            for alert in tools.alert_history():
                with st.expander(f"{alert['as_of']} · {alert['aspect']} · +{alert['change_pp']:.2f} 个百分点 · 待核查"):
                    st.write(f"本期 {alert['count']}/{alert['current_samples']}，上期 {alert['previous_count']}/{alert['previous_samples']}，窗口 {alert['window_days']} 天。")
                    st.json(alert["evidence"])

        with evidence_tab:
            aspect = st.selectbox("筛选负向问题", ["全部"] + list(ASPECTS))
            records = tools.get_evidence(None if aspect == "全部" else aspect, 20)
            if not records:
                st.info("没有匹配的已分析评论。")
            st.caption("展示最近最多 20 条；筛选后的原文是样本，不能据此计算总体比例。")
            for item in records:
                with st.expander(f"[{item['evidence_id']}] {item['day']} · {item['source']} · {item['source_record_id']}"):
                    st.text(item["text"])
                    st.dataframe(item["opinions"], hide_index=True, width="stretch")
                    ref = item["evidence_ref"]
                    parsed = urlparse(ref)
                    if parsed.scheme in ("https", "http") and parsed.netloc:
                        st.link_button("打开原文", ref)
                    else:
                        st.code(ref, language=None)
                    st.caption(f"分析方法：{item['analyzer']}")

        with agent_tab:
            st.subheader("带证据的分析报告")
            content = report(tools)
            st.download_button("下载统计报告（Markdown）", content, "opinion-report.md", "text/markdown")
            with st.expander("查看统计报告"):
                st.markdown(content)
            st.subheader("向 Agent 提问")
            st.caption("模型可调用趋势、问题、证据、商品对比四个工具。问答使用上方选定的商品和日期，每次提问独立执行。")
            question = st.text_area("分析问题", "最近一周哪些问题增长最快？给出原文依据和优先核查建议。", max_chars=4000)
            if not configured():
                st.info("配置项目 .env 后启用真正的模型工具调用；上方统计报告现在即可使用。")
            if st.button("开始分析", disabled=not configured()):
                with st.spinner("Agent 正在查询统计与原文…"):
                    try:
                        result = ask(tools, question)
                        st.markdown(render_answer(result))
                        with st.expander("工具调用记录"):
                            st.json(result["trace"])
                    except Exception as exc:
                        st.error(f"分析失败（{type(exc).__name__}）。检查模型配置、工具调用支持和网络；本次未生成结论。")
        with data_tab:
            if is_demo:
                st.info("模拟空间使用固定规则分析。切换到“我的数据”导入并分析真实评论。")
            else:
                import_panel(db)


def import_panel(db):
    st.subheader("导入评论")
    st.caption("UTF-8 CSV / JSONL，单文件最多 10 MB、10,000 条。支持统一字段及 Amazon Reviews 2023 评论 JSONL。")
    st.code("source,source_record_id,product_id,text,rating,published_at,evidence_ref", language=None)
    file = st.file_uploader("选择评论文件", type=["csv", "jsonl"])
    if st.button("导入文件", disabled=file is None):
        try:
            stats = db.import_file(file.name, file.getvalue())
            st.session_state["notice"] = f"新增 {stats['inserted']} 条，重复 {stats['duplicates']} 条，ID 冲突 {stats['conflicts']} 条（保留旧记录）。"
            st.rerun()
        except (ValueError, TypeError, UnicodeError) as exc:
            st.error(f"导入失败，整批已回滚：{exc}")
    st.subheader("分析待处理评论")
    use_model = st.checkbox("使用模型提取观点（评论文本将发给你配置的模型服务）", disabled=not configured())
    limit = st.number_input("本批最多分析条数", min_value=1, max_value=1000, value=100 if use_model else 500)
    reanalyze = st.checkbox("重新分析库中最早的一批评论，替换已有分析")
    st.caption("规则模式只用于演示。模型调用逐条执行；分析失败保留待处理状态，成功记录立即保存，下次可继续。")
    if st.button("执行观点分析"):
        with st.spinner("正在分析评论…"):
            try:
                kwargs = {"limit": int(limit), "reanalyze": reanalyze}
                if use_model:
                    analyzer = ModelAnalyzer()
                    kwargs.update(analyzer=analyzer, label=analyzer.label)
                result = db.analyze_pending(**kwargs)
                st.success(f"成功 {result['analyzed']} 条，失败 {len(result['failed'])} 条。")
                if result["failed"]:
                    st.json(result["failed"])
                else:
                    st.session_state["notice"] = f"已分析 {result['analyzed']} 条评论。"
                    st.rerun()
            except Exception as exc:
                st.error(f"无法开始分析（{type(exc).__name__}），请检查模型配置。")


if __name__ == "__main__":
    main()
