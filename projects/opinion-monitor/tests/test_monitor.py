import json
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import ask
from demo import demo_records
from monitor import Monitor, Tools, normalize, report, rule_analyze, summarize


def record(id="one", text="产品好用，就是物流慢。", published="2026-09-20", **kwargs):
    return {"source": "test", "source_record_id": id, "product_id": "test-product",
            "text": text, "published_at": published, **kwargs}


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.db = Monitor(":memory:")

    def tearDown(self):
        self.db.conn.close()

    def add(self, records):
        return self.db.import_records((r, "fixture") for r in records)

    def demo(self):
        self.db.import_records(demo_records())
        self.db.analyze_pending(limit=1000)
        return Tools(self.db, "演示-无线耳机", date(2026, 9, 21))

    def test_import_deduplicates_and_preserves_conflicting_original(self):
        self.assertEqual(self.add([record()])["inserted"], 1)
        self.assertEqual(self.add([record()])["duplicates"], 1)
        self.assertEqual(self.add([record(text="新文本")])["conflicts"], 1)
        row = self.db.rows("test-product", date(2026, 9, 1), date(2026, 9, 30))[0]
        self.assertEqual(row["text"], "产品好用，就是物流慢。")
        self.assertEqual(json.loads(row["raw_json"])["text"], row["text"])

    def test_import_is_atomic_when_later_record_is_invalid(self):
        content = (json.dumps(record()) + '\n{"text":"missing required fields"}').encode()
        with self.assertRaises(ValueError):
            self.db.import_file("reviews.jsonl", content)
        self.assertEqual(self.db.products(), [])
        with self.assertRaises(ValueError):
            self.db.import_file("reviews.jsonl", (json.dumps(record()) + "\n{broken").encode())
        self.assertEqual(self.db.products(), [])

    def test_csv_bom_and_blank_rating(self):
        content = "\ufeffsource,source_record_id,product_id,text,rating,published_at\nx,1,p,好用,,2026-09-01\n".encode("utf-8")
        self.assertEqual(self.db.import_file("reviews.csv", content)["inserted"], 1)

    def test_date_conversion_and_amazon_stable_identity(self):
        normalized = normalize(record(published="2026-09-01T23:00:00Z"), "file")
        self.assertTrue(normalized.published_at.startswith("2026-09-02T07:00"))
        amazon = {"parent_asin": "A", "text": "Good", "timestamp": 1660000000000, "user_id": "u", "rating": 4}
        self.assertEqual(normalize(amazon, "file1").source_record_id, normalize(amazon, "file2").source_record_id)
        with self.assertRaises(ValueError):
            normalize([], "bad")

    def test_mixed_sentiments_and_unique_aspect_counts(self):
        self.add([record(text="产品好用，物流慢，配送慢，客服回复慢。")])
        self.db.analyze_pending()
        tools = Tools(self.db, "test-product", date(2026, 9, 21))
        opinions = tools.get_evidence()[0]["opinions"]
        self.assertIn(("使用效果", "正向"), [(o["aspect"], o["sentiment"]) for o in opinions])
        counts = {i["aspect"]: i["count"] for i in tools.get_top_issues()["issues"]}
        self.assertEqual(counts["物流"], 1)
        self.assertEqual(counts["客服售后"], 1)

    def test_unknown_is_visible_and_pending_excluded_from_denominator(self):
        self.add([record(), record(id="two", text="今天收到了"), record(id="three")])
        self.db.analyze_pending(limit=2)
        summary = Tools(self.db, "test-product", date(2026, 9, 21)).get_trend()["summary"]
        self.assertEqual((summary["analyzed"], summary["pending"], summary["unknown"]), (2, 1, 1))
        self.assertEqual(summary["negative_rate"], 0.5)

    def test_hallucinated_quote_rejected_and_record_remains_pending(self):
        self.add([record()])
        def bad_analyzer(_):
            return {"opinions": [{"aspect": "物流", "sentiment": "负向", "quote": "不存在的原文"}]}
        result = self.db.analyze_pending(bad_analyzer, "bad-model")
        self.assertEqual(result["analyzed"], 0)
        self.assertEqual(result["failed"][0]["error"], "ValueError")
        self.assertEqual(self.db.analyze_pending()["analyzed"], 1)

    def test_replay_never_sees_future_reviews_or_alerts(self):
        tools = self.demo()
        tools.check_alerts()
        earlier = Tools(self.db, "演示-无线耳机", date(2026, 9, 17))
        self.assertEqual(earlier.get_trend()["summary"]["negative"], 7)
        self.assertEqual(earlier.alert_history(), [])
        self.assertTrue(all(r["day"] <= "2026-09-17" for r in earlier.get_evidence(limit=20)))
        self.assertEqual(earlier.check_alerts(), [])

    def test_demo_spike_alerts_have_evidence_and_cooldown(self):
        tools = self.demo()
        issues = {i["aspect"]: i for i in tools.get_top_issues()["issues"]}
        self.assertEqual(issues["使用效果"]["count"], 24)
        self.assertAlmostEqual(issues["使用效果"]["change_pp"], 28.57)
        alerts = tools.check_alerts()
        self.assertEqual({a["aspect"] for a in alerts}, {"使用效果", "客服售后"})
        self.assertTrue(all(a["evidence"] for a in alerts))
        self.assertEqual(tools.check_alerts(), [])
        next_day = Tools(self.db, tools.product_id, date(2026, 9, 22))
        self.assertEqual(next_day.check_alerts(), [])
        self.assertIn("[R", report(tools))

    def test_volume_growth_with_unchanged_issue_rate_does_not_alert(self):
        rows = [record(id=f"old-{i}", text="物流慢" if i < 10 else "好用", published="2026-09-10") for i in range(40)]
        rows += [record(id=f"new-{i}", text="物流慢" if i < 20 else "好用", published="2026-09-20") for i in range(80)]
        self.add(rows)
        self.db.analyze_pending()
        self.assertEqual(Tools(self.db, "test-product", date(2026, 9, 21)).check_alerts(), [])

    def test_incomplete_or_mixed_analysis_suppresses_alerts(self):
        tools = self.demo()
        with self.db.conn:
            self.db.conn.execute("UPDATE reviews SET analysis_json=NULL WHERE id=1")
        # id=1 falls outside the comparison; choose a record inside the previous window.
        with self.db.conn:
            self.db.conn.execute("UPDATE reviews SET analysis_json=NULL WHERE product_id=? AND day='2026-09-10'", (tools.product_id,))
        self.assertEqual(tools.check_alerts(), [])
        self.db.analyze_pending()
        with self.db.conn:
            self.db.conn.execute("UPDATE reviews SET analyzer='model-v1:other' WHERE product_id=? AND day='2026-09-20'", (tools.product_id,))
        self.assertEqual(tools.check_alerts(), [])

    def test_insufficient_baseline_does_not_alert(self):
        self.add([record(id=str(i), text="物流慢") for i in range(40)])
        self.db.analyze_pending()
        tools = Tools(self.db, "test-product", date(2026, 9, 21))
        self.assertIsNone(tools.get_top_issues()["issues"][0]["change_pp"])
        self.assertEqual(tools.check_alerts(), [])

    def fake_client(self, evidence_id):
        from openai.types.chat import ChatCompletion
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "get_trend", "arguments": "{}"}},
                {"id": "t2", "type": "function", "function": {"name": "get_evidence", "arguments": '{"aspect":"物流","limit":2}'}},
            ]},
            {"role": "assistant", "content": json.dumps({"findings": [{"text": "有物流慢反馈。", "evidence_ids": [evidence_id]}],
                                                         "next_steps": ["核查物流订单"], "limitations": "样本有限"})},
        ]
        def create(**kwargs):
            return ChatCompletion(id="fake", created=0, model="fake", object="chat.completion",
                                  choices=[{"index": 0, "finish_reason": "stop", "message": messages.pop(0)}])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_agent_executes_tools_and_validates_citations(self):
        self.add([record()])
        self.db.analyze_pending()
        tools = Tools(self.db, "test-product", date(2026, 9, 21))
        result = ask(tools, "有什么问题？", self.fake_client("R1"), "fake")
        self.assertEqual([t["tool"] for t in result["trace"]], ["get_trend", "get_evidence"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "R1")
        with self.assertRaisesRegex(ValueError, "未查询到的证据"):
            ask(tools, "有什么问题？", self.fake_client("R99999"), "fake")


if __name__ == "__main__":
    unittest.main()
