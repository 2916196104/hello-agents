"""Import, aspect analysis, time-bounded statistics and local alert storage."""

import csv
import hashlib
import io
import json
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TAIPEI = timezone(timedelta(hours=8))
ASPECTS = ("商品质量", "使用效果", "价格", "包装", "物流", "客服售后", "其他")


class Opinion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect: Literal["商品质量", "使用效果", "价格", "包装", "物流", "客服售后", "其他"]
    sentiment: Literal["正向", "负向", "中性", "未知"]
    quote: str = Field(min_length=1, max_length=12000)


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opinions: list[Opinion] = Field(max_length=20)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    source: str = Field(min_length=1, max_length=100)
    source_record_id: str = Field(min_length=1, max_length=300)
    product_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=12000)
    rating: float | None = Field(default=None, ge=1, le=5, allow_inf_nan=False)
    published_at: str
    evidence_ref: str = Field(min_length=1, max_length=2000)


def today() -> date:
    return datetime.now(TAIPEI).date()


def normalize(raw: dict, location: str) -> Review:
    """Accept the common schema, or an Amazon Reviews 2023 JSONL record."""
    if not isinstance(raw, dict):
        raise ValueError("每条记录必须是 JSON 对象")
    amazon = "parent_asin" in raw and "timestamp" in raw
    text = raw.get("text", "")
    product = raw.get("product_id") or raw.get("parent_asin")
    published = raw.get("published_at")
    if amazon and not published:
        published = datetime.fromtimestamp(float(raw["timestamp"]) / 1000, TAIPEI).isoformat()
    if not isinstance(published, str):
        raise ValueError("published_at 必须是 ISO 日期或时间")
    moment = datetime.fromisoformat(published.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=TAIPEI)
    record_id = raw.get("source_record_id")
    if amazon and not record_id:
        # Amazon has no review ID; the stable fingerprint survives file reordering.
        identity = [product, raw.get("user_id"), raw["timestamp"], text]
        record_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
    return Review(
        source=raw.get("source") or ("amazon-2023" if amazon else "import"),
        source_record_id=record_id,
        product_id=product,
        text=text,
        rating=None if raw.get("rating") in (None, "") else raw["rating"],
        published_at=moment.astimezone(TAIPEI).isoformat(),
        evidence_ref=raw.get("evidence_ref") or location,
    )


# shortcut: rules only recognize a small vocabulary; use model analysis for real monitoring.
RULES = {
    "商品质量": (("坏了", "断裂", "漏液", "故障", "破损", "开裂"), ("质量很好", "结实", "耐用")),
    "使用效果": (("没效果", "不好用", "杂音", "断连", "续航差", "刺激"), ("好用", "效果很好", "音质清晰")),
    "价格": (("涨价", "太贵", "不值", "乱收费"), ("实惠", "划算", "便宜")),
    "包装": (("包装破", "包装简陋", "盒子压坏"), ("包装完好", "包装精美")),
    "物流": (("物流慢", "没收到", "延迟", "丢件", "配送慢"), ("物流快", "配送快", "准时送达")),
    "客服售后": (("不退款", "退款难", "客服慢", "回复慢", "不理人", "售后差"), ("客服耐心", "退款及时", "服务很好")),
}


def rule_analyze(text: str) -> Analysis:
    opinions = []
    for clause in re.split(r"[，。！？；,;!?\n]|但是|不过|就是", text):
        clause = clause.strip()
        if not clause:
            continue
        for aspect, (negative, positive) in RULES.items():
            if any(word in clause for word in negative):
                opinions.append(Opinion(aspect=aspect, sentiment="负向", quote=clause))
            elif any(word in clause for word in positive):
                opinions.append(Opinion(aspect=aspect, sentiment="正向", quote=clause))
    if not opinions:
        opinions = [Opinion(aspect="其他", sentiment="未知", quote=text)]
    return Analysis(opinions=opinions[:20])


class Monitor:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=15)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY, source TEXT NOT NULL,
                source_record_id TEXT NOT NULL, product_id TEXT NOT NULL,
                text TEXT NOT NULL, rating REAL, published_at TEXT NOT NULL,
                day TEXT NOT NULL, fetched_at TEXT NOT NULL,
                evidence_ref TEXT NOT NULL, raw_json TEXT NOT NULL,
                analysis_json TEXT, analyzer TEXT, analyzed_at TEXT,
                UNIQUE(source, source_record_id)
            );
            CREATE INDEX IF NOT EXISTS reviews_product_day ON reviews(product_id, day);
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY, product_id TEXT NOT NULL, aspect TEXT NOT NULL,
                as_of TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(product_id, aspect, as_of)
            );
        """)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.conn.close()

    def import_records(self, records) -> dict:
        stats = {"inserted": 0, "duplicates": 0, "conflicts": 0}
        with self.conn:
            for index, (raw, location) in enumerate(records, 1):
                if index > 10000:
                    raise ValueError("每次最多导入 10,000 条，请拆分文件")
                try:
                    review = normalize(raw, location)
                except (ValueError, TypeError, OverflowError, OSError) as exc:
                    raise ValueError(f"第 {index} 条记录无效：{exc}") from exc
                values = review.model_dump()
                old = self.conn.execute(
                    "SELECT * FROM reviews WHERE source=? AND source_record_id=?",
                    (review.source, review.source_record_id),
                ).fetchone()
                if old:
                    changed = any(old[key] != values[key] for key in ("product_id", "text", "rating", "published_at"))
                    stats["conflicts" if changed else "duplicates"] += 1
                    continue
                self.conn.execute("""
                    INSERT INTO reviews(source, source_record_id, product_id, text, rating,
                        published_at, day, fetched_at, evidence_ref, raw_json)
                    VALUES(:source, :source_record_id, :product_id, :text, :rating,
                        :published_at, :day, :fetched_at, :evidence_ref, :raw_json)
                """, {**values, "day": review.published_at[:10],
                      "fetched_at": datetime.now(TAIPEI).isoformat(),
                      "raw_json": json.dumps(raw, ensure_ascii=False)})
                stats["inserted"] += 1
        return stats

    def import_file(self, name: str, content: bytes) -> dict:
        if len(content) > 10 * 1024 * 1024:
            raise ValueError("文件不得超过 10 MB，请先选取少量商品或拆分文件")
        text = content.decode("utf-8-sig")
        suffix = Path(name).suffix.lower()
        if suffix == ".csv":
            records = ((row, f"{name}:record={i}") for i, row in enumerate(csv.DictReader(io.StringIO(text)), 1))
        elif suffix == ".jsonl":
            records = ((json.loads(line), f"{name}:line={i}") for i, line in enumerate(text.splitlines(), 1) if line.strip())
        else:
            raise ValueError("只支持 UTF-8 CSV 和 JSONL 文件")
        return self.import_records(records)

    def products(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT product_id FROM reviews ORDER BY product_id")]

    def rows(self, product_id: str | None, start: date, end: date) -> list[dict]:
        if end < start:
            raise ValueError("结束日期不得早于开始日期")
        sql = "SELECT * FROM reviews WHERE day BETWEEN ? AND ?"
        args = [start.isoformat(), end.isoformat()]
        if product_id:
            sql += " AND product_id=?"
            args.append(product_id)
        return [dict(row) for row in self.conn.execute(sql + " ORDER BY day, id", args)]

    def analyze_pending(self, analyzer=rule_analyze, label="rules-v1", limit=500, reanalyze=False) -> dict:
        if not 1 <= limit <= 1000:
            raise ValueError("单批分析数量须为 1—1000")
        where = "" if reanalyze else "WHERE analysis_json IS NULL"
        rows = self.conn.execute(f"SELECT id, text FROM reviews {where} ORDER BY id LIMIT ?", (limit,)).fetchall()
        done, errors = 0, []
        for row in rows:
            try:
                result = Analysis.model_validate(analyzer(row["text"]))
                if not result.opinions or any(o.quote not in row["text"] for o in result.opinions):
                    raise ValueError("模型证据片段必须逐字出现在原文，且观点不能为空")
                with self.conn:
                    self.conn.execute(
                        "UPDATE reviews SET analysis_json=?, analyzer=?, analyzed_at=? WHERE id=?",
                        (result.model_dump_json(), label, datetime.now(TAIPEI).isoformat(), row["id"]),
                    )
                done += 1
            except Exception as exc:
                # Keep failed records pending; never silently substitute rule labels.
                errors.append({"id": row["id"], "error": type(exc).__name__})
        return {"analyzed": done, "failed": errors, "attempted": len(rows)}


def negatives(row: dict) -> set[str]:
    analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else {"opinions": []}
    return {o["aspect"] for o in analysis["opinions"] if o["sentiment"] == "负向"}


def summarize(rows: list[dict]) -> dict:
    analyzed = [r for r in rows if r["analysis_json"]]
    negative = sum(bool(negatives(r)) for r in analyzed)
    unknown = sum(all(o["sentiment"] == "未知" for o in json.loads(r["analysis_json"])["opinions"]) for r in analyzed)
    return {"total": len(rows), "analyzed": len(analyzed), "pending": len(rows) - len(analyzed),
            "negative": negative, "negative_rate": negative / len(analyzed) if analyzed else None,
            "unknown": unknown, "coverage": len(analyzed) / len(rows) if rows else None,
            "analyzers": dict(Counter(r["analyzer"] for r in analyzed))}


class Tools:
    """A query context pins all model tools to the selected product and replay date."""
    def __init__(self, db: Monitor, product_id: str, as_of: date, days=7):
        if not 1 <= days <= 90:
            raise ValueError("分析窗口须为 1—90 天")
        self.db, self.product_id, self.as_of, self.days = db, product_id, as_of, days
        self.start = as_of - timedelta(days=days - 1)

    def get_trend(self) -> dict:
        rows = self.db.rows(self.product_id, self.start, self.as_of)
        daily = []
        for offset in range(self.days):
            day = (self.start + timedelta(days=offset)).isoformat()
            daily.append({"day": day, **summarize([r for r in rows if r["day"] == day])})
        return {"product_id": self.product_id, "start": self.start.isoformat(),
                "end": self.as_of.isoformat(), "summary": summarize(rows), "daily": daily}

    def get_top_issues(self) -> dict:
        current = self.db.rows(self.product_id, self.start, self.as_of)
        previous = self.db.rows(self.product_id, self.start - timedelta(days=self.days), self.start - timedelta(days=1))
        n, baseline_n = summarize(current)["analyzed"], summarize(previous)["analyzed"]
        counts = Counter(a for row in current for a in negatives(row))
        old = Counter(a for row in previous for a in negatives(row))
        issues = []
        for aspect in counts.keys() | old.keys():
            rate, baseline = counts[aspect] / n if n else None, old[aspect] / baseline_n if baseline_n else None
            issues.append({"aspect": aspect, "count": counts[aspect], "previous_count": old[aspect],
                           "rate": rate, "previous_rate": baseline,
                           "change_pp": round((rate - baseline) * 100, 2) if rate is not None and baseline is not None else None})
        issues.sort(key=lambda x: (x["change_pp"] if x["change_pp"] is not None else -101, x["count"]), reverse=True)
        return {"current_samples": n, "previous_samples": baseline_n,
                "current_coverage": summarize(current)["coverage"],
                "previous_coverage": summarize(previous)["coverage"], "issues": issues}

    def get_evidence(self, aspect: str | None = None, limit: int = 5) -> list[dict]:
        if aspect is not None and aspect not in ASPECTS:
            raise ValueError("未知问题维度")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("证据数量须为 1—20")
        rows = reversed(self.db.rows(self.product_id, self.start, self.as_of))
        result = []
        for row in rows:
            if not row["analysis_json"] or (aspect and aspect not in negatives(row)):
                continue
            result.append({"evidence_id": f"R{row['id']}", "source": row["source"],
                           "source_record_id": row["source_record_id"], "day": row["day"],
                           "text": row["text"], "evidence_ref": row["evidence_ref"],
                           "analyzer": row["analyzer"], "opinions": json.loads(row["analysis_json"])["opinions"]})
            if len(result) >= limit:
                break
        return result

    def compare_products(self) -> list[dict]:
        return [{"product_id": product, **summarize(self.db.rows(product, self.start, self.as_of))}
                for product in self.db.products()]

    def check_alerts(self, min_samples=20, min_mentions=5, min_increase_pp=15, cooldown_days=3) -> list[dict]:
        comparison = self.get_top_issues()
        # An incomplete analysis batch can change apparent rates enough to fabricate a spike.
        if min(comparison["current_samples"], comparison["previous_samples"]) < min_samples:
            return []
        if comparison["current_coverage"] != 1 or comparison["previous_coverage"] != 1:
            return []
        labels = {r["analyzer"] for r in self.db.rows(self.product_id, self.start - timedelta(days=self.days), self.as_of)}
        if len(labels) != 1:
            return []
        added = []
        with self.db.conn:
            for issue in comparison["issues"]:
                if issue["count"] < min_mentions or issue["change_pp"] is None or issue["change_pp"] < min_increase_pp:
                    continue
                recent = self.db.conn.execute(
                    "SELECT 1 FROM alerts WHERE product_id=? AND aspect=? AND as_of BETWEEN ? AND ?",
                    (self.product_id, issue["aspect"], (self.as_of - timedelta(days=cooldown_days - 1)).isoformat(), self.as_of.isoformat()),
                ).fetchone()
                if recent:
                    continue
                payload = {"product_id": self.product_id, "as_of": self.as_of.isoformat(),
                           "window_days": self.days, **issue, "current_samples": comparison["current_samples"],
                           "previous_samples": comparison["previous_samples"],
                           "evidence": self.get_evidence(issue["aspect"], 3), "status": "待核查"}
                self.db.conn.execute(
                    "INSERT INTO alerts(product_id,aspect,as_of,payload,created_at) VALUES(?,?,?,?,?)",
                    (self.product_id, issue["aspect"], self.as_of.isoformat(), json.dumps(payload, ensure_ascii=False), datetime.now(TAIPEI).isoformat()),
                )
                added.append(payload)
        return added

    def alert_history(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.conn.execute(
            "SELECT payload FROM alerts WHERE product_id=? AND as_of<=? ORDER BY as_of DESC, id DESC LIMIT 50",
            (self.product_id, self.as_of.isoformat()))]


def report(tools: Tools) -> str:
    trend, issues = tools.get_trend(), tools.get_top_issues()
    summary = trend["summary"]
    rate = f"{summary['negative_rate']:.1%}" if summary["negative_rate"] is not None else "无已分析样本"
    lines = [f"# {tools.product_id} 口碑分析", "", f"统计期：{trend['start']} 至 {trend['end']}（Asia/Taipei）。",
             f"评论 {summary['total']} 条，已分析 {summary['analyzed']} 条，待分析 {summary['pending']} 条；包含负向观点的评论占已分析评论的 {rate}。",
             f"全部观点未知的评论 {summary['unknown']} 条。分析方法：{json.dumps(summary['analyzers'], ensure_ascii=False)}。", "",
             "## 问题变化", "", f"本期／上期已分析样本：{issues['current_samples']}／{issues['previous_samples']}。"]
    evidence_by_id = {}
    for issue in issues["issues"][:3]:
        change = f"{issue['change_pp']:+.2f} 个百分点" if issue["change_pp"] is not None else "基线不足，无法比较"
        evidence = tools.get_evidence(issue["aspect"], 2)
        evidence_by_id.update({item["evidence_id"]: item for item in evidence})
        refs = " ".join(f"[{item['evidence_id']}]" for item in evidence)
        lines.append(f"- {issue['aspect']}：本期 {issue['count']} 条、上期 {issue['previous_count']} 条，占比变化 {change}。{refs}")
    if not issues["issues"]:
        lines.append("没有识别到负向问题；请结合待分析和未知评论数量判断覆盖情况。")
    lines += ["", "## 原文证据", ""]
    for item in evidence_by_id.values():
        lines += [f"[{item['evidence_id']}] {item['day']} · {item['source']} · {item['source_record_id']}",
                  f"> {item['text'].replace(chr(10), ' ')}", f"来源：{item['evidence_ref']}", ""]
    lines += ["## 处理建议与局限", "", "先核查增长问题对应的订单、批次与售后记录，再决定处理顺序。占比变化仅描述相关性，不能证明投诉属实或判断具体原因。",
              "统计只代表已导入的数据；未提及某个问题不等于满意。规则模式无法可靠处理否定、反讽和隐含观点。"]
    return "\n".join(lines)
