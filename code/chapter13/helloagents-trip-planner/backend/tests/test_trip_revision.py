"""无需外部模型密钥的行程修改回归测试。"""

import json
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.trip_revision_agent import (
    RevisionGenerationError,
    RevisionValidationError,
    TripRevisionAgent,
)
from app.api.main import app
from app.models.schemas import TripPlan, TripRevisionRequest


def sample_plan() -> TripPlan:
    return TripPlan.model_validate({
        "city": "北京", "start_date": "2026-10-20", "end_date": "2026-10-22",
        "overall_suggestions": "按原计划游览，提前预约。",
        "weather_info": [{"date": "2026-10-21", "day_weather": "晴", "day_temp": 20}],
        "days": [{
            "date": f"2026-10-{20 + day}", "day_index": day,
            "description": f"第{day + 1}天游览", "transportation": "公共交通",
            "accommodation": "经济型酒店",
            "hotel": {"name": f"酒店{day + 1}", "estimated_cost": 300},
            "attractions": [{
                "name": f"景点{day + 1}-{index + 1}", "address": "北京市",
                "location": {"longitude": 116.4 + index * 0.01, "latitude": 39.9},
                "visit_duration": 120, "description": "原景点资料", "ticket_price": 20,
            } for index in range(3)],
            "meals": [{"type": kind, "name": kind, "estimated_cost": 30}
                      for kind in ("breakfast", "lunch", "dinner")],
        } for day in range(3)],
        "budget": {"total_attractions": 180, "total_hotels": 900, "total_meals": 270,
                   "total_transportation": 120, "total": 1470},
    })


class TripRevisionTests(unittest.TestCase):
    def setUp(self):
        self.plan = sample_plan()
        self.request = TripRevisionRequest(
            trip_plan=self.plan, instruction="第二天太累了，减少一个景点，酒店不变",
        )
        self.changed = self.plan.days[1].model_copy(deep=True)
        self.changed.attractions.pop()
        self.changed.description = "减少一个景点，留出休息时间"
        self.llm = Mock()
        self.mock_llm = patch("app.agents.trip_revision_agent.get_llm", return_value=self.llm)
        self.mock_llm.start()
        self.addCleanup(self.mock_llm.stop)
        self.respond([self.changed])

    def respond(self, days):
        self.llm.invoke.return_value = "```json\n" + json.dumps({
            "days": [day.model_dump() for day in days], "summary": "第二天减少一个景点，酒店不变。",
        }, ensure_ascii=False) + "\n```"

    def test_reduces_second_day_and_preserves_other_days_hotels_and_metadata(self):
        before = self.plan.model_dump()
        result = TripRevisionAgent().revise_trip(self.request)
        self.assertTrue(result.success)
        self.assertEqual(result.changed_days, [1])
        self.assertEqual(len(result.data.days[1].attractions), 2)
        for index in (0, 2):
            self.assertEqual(result.data.days[index], self.plan.days[index])
        for index in range(3):
            self.assertEqual(result.data.days[index].hotel, self.plan.days[index].hotel)
        self.assertEqual(result.data.weather_info, self.plan.weather_info)
        self.assertEqual(result.data.overall_suggestions, self.plan.overall_suggestions)
        self.assertEqual(self.plan.model_dump(), before)
        self.assertEqual(result.data.budget.total_attractions, 160)
        self.assertEqual(result.data.budget.total_hotels, 900)
        self.assertEqual(result.data.budget.total_transportation, 120)
        self.assertEqual(result.data.budget.total, 1450)
        messages = self.llm.invoke.call_args.args[0]
        prompt = json.loads(messages[-1]["content"])
        self.assertEqual(prompt["trip_plan"], before)
        self.assertEqual(prompt["editable_day_indices"], [1])

    def test_rejects_out_of_scope_day(self):
        self.changed.day_index = 0
        self.respond([self.changed])
        with self.assertRaisesRegex(RevisionValidationError, "范围外"):
            TripRevisionAgent().revise_trip(self.request)

    def test_rejects_changed_date(self):
        self.changed.date = "2026-11-01"
        self.respond([self.changed])
        with self.assertRaisesRegex(RevisionValidationError, "日期"):
            TripRevisionAgent().revise_trip(self.request)

    def test_rejects_locked_hotel_or_accommodation(self):
        self.changed.hotel.name = "另一家酒店"
        self.respond([self.changed])
        with self.assertRaisesRegex(RevisionValidationError, "酒店或住宿"):
            TripRevisionAgent().revise_trip(self.request)

    def test_instruction_locks_hotels_even_when_checkbox_is_off(self):
        self.request.keep_hotels = False
        self.test_rejects_locked_hotel_or_accommodation()

    def test_can_change_hotel_when_explicitly_unlocked(self):
        self.request.instruction = "第二天改住第一天的酒店"
        self.request.keep_hotels = False
        self.changed.hotel = self.plan.days[0].hotel.model_copy(deep=True)
        self.respond([self.changed])
        result = TripRevisionAgent().revise_trip(self.request)
        self.assertEqual(result.data.days[1].hotel, self.plan.days[0].hotel)

    def test_requires_exact_reduction_count(self):
        self.changed.attractions.pop()
        self.respond([self.changed])
        with self.assertRaisesRegex(RevisionValidationError, "数量"):
            TripRevisionAgent().revise_trip(self.request)

    def test_rejects_empty_day_duplicate_day_or_new_place(self):
        cases = ("empty", "duplicate", "new_place", "negative_price", "invalid_location")
        for case in cases:
            with self.subTest(case=case):
                day = self.changed.model_copy(deep=True)
                days = [day]
                if case == "empty":
                    day.attractions = []
                elif case == "duplicate":
                    days.append(day)
                elif case == "new_place":
                    day.attractions[0].name = "未经查询的新景点"
                elif case == "negative_price":
                    day.attractions[0].ticket_price = -1
                else:
                    day.attractions[0].location.longitude = 300
                self.respond(days)
                with self.assertRaises(RevisionValidationError):
                    TripRevisionAgent().revise_trip(self.request)

    def test_rejects_noop_or_insufficient_information(self):
        for days in ([self.plan.days[1]], []):
            with self.subTest(days=len(days)):
                self.respond(days)
                with self.assertRaises(RevisionValidationError):
                    TripRevisionAgent().revise_trip(self.request)

    def test_bad_json_and_provider_failure_do_not_mutate_original(self):
        before = self.plan.model_dump()
        self.llm.invoke.return_value = "这不是有效的 JSON"
        with self.assertRaises(RevisionGenerationError):
            TripRevisionAgent().revise_trip(self.request)
        self.llm.invoke.side_effect = RuntimeError("provider unavailable")
        with self.assertRaises(RevisionGenerationError):
            TripRevisionAgent().revise_trip(self.request)
        self.assertEqual(self.plan.model_dump(), before)

    def test_rejects_invalid_input_before_calling_model(self):
        for instruction in ("第十天减少一个景点", "第0天减少一个景点"):
            self.request.instruction = instruction
            with self.assertRaises(RevisionValidationError):
                TripRevisionAgent().revise_trip(self.request)
        self.llm.invoke.assert_not_called()
        with self.assertRaises(ValidationError):
            TripRevisionRequest(trip_plan=self.plan, instruction="  \n  ")

    def test_provider_rate_limit_and_authentication_errors_are_actionable(self):
        client = TestClient(app)
        for status, expected_status, message in ((429, 429, "限流"), (401, 502, "认证失败")):
            with self.subTest(status=status):
                cause = RuntimeError("provider error containing private details")
                cause.status_code = status
                error = RuntimeError("wrapped model error")
                error.__cause__ = cause
                self.llm.invoke.side_effect = error
                response = client.post("/api/trip/revise", json=self.request.model_dump())
                self.assertEqual(response.status_code, expected_status)
                self.assertIn(message, response.json()["detail"])
                self.assertNotIn("private", response.json()["detail"])

    def test_requests_do_not_share_agent_history(self):
        agent = TripRevisionAgent()
        agent.revise_trip(self.request)
        agent.revise_trip(self.request)
        self.assertEqual([len(call.args[0]) for call in self.llm.invoke.call_args_list], [2, 2])

    def test_api_returns_updated_plan_and_validation_errors(self):
        client = TestClient(app)
        response = client.post("/api/trip/revise", json=self.request.model_dump())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["changed_days"], [1])
        self.assertEqual(len(response.json()["data"]["days"][1]["attractions"]), 2)
        self.changed.hotel.name = "错误酒店"
        self.respond([self.changed])
        response = client.post("/api/trip/revise", json=self.request.model_dump())
        self.assertEqual(response.status_code, 422)
        self.assertIn("酒店", response.json()["detail"])
        self.llm.invoke.return_value = "invalid JSON"
        response = client.post("/api/trip/revise", json=self.request.model_dump())
        self.assertEqual(response.status_code, 502)
        self.assertEqual(client.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
