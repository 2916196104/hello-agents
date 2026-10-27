"""Deterministic synthetic reviews; never represented as real consumer evidence."""

from datetime import date, timedelta


def demo_records():
    start = date(2026, 9, 1)
    products = ("演示-无线耳机", "演示-保温杯", "演示-洗面奶")
    positive = ("质量很好，包装完好。", "非常好用，物流快。", "价格实惠，客服耐心。")
    for day_index in range(21):
        day = start + timedelta(days=day_index)
        for product_index, product in enumerate(products):
            for i in range(12):
                text, rating = positive[i % len(positive)], 5
                if i == 0:
                    text, rating = "产品好用，就是物流慢。", 3
                if product_index == 0 and day_index >= 18 and 1 <= i <= 8:
                    text, rating = "包装完好，但是耳机断连还有杂音，客服回复慢。", 2
                record_id = f"synthetic-{day_index}-{product_index}-{i}"
                raw = {"source": "synthetic-demo", "source_record_id": record_id,
                       "product_id": product, "text": text, "rating": rating,
                       "published_at": f"{day}T12:00:00+08:00",
                       "evidence_ref": f"synthetic-demo:{record_id}"}
                yield raw, raw["evidence_ref"]
