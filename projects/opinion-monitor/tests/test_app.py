"""Streamlit's official test runner exercises the same script as the browser."""

import unittest
from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_demo_replay_and_empty_import_workspace(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "84")
        self.assertEqual(app.metric[2].value, "31")
        app.date_input[0].set_value(date(2026, 9, 17)).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[2].value, "7")
        app.radio[0].set_value("我的数据").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.file_uploader[0].label, "选择评论文件")


if __name__ == "__main__":
    unittest.main()
