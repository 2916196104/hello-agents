"""离线回归检查：使用真实网页组件，替换需要外部服务的记忆和检索工具。"""

import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pdf_rag_tool


spec = importlib.util.spec_from_file_location(
    "pdf_learning_assistant", Path(__file__).with_name("11_Q&A_Assistant.py")
)
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class PDFLearningAssistantChecks(unittest.TestCase):
    def setUp(self):
        memory_patch = patch.object(app, "MemoryTool")
        rag_patch = patch.object(app, "PDFRAGTool")
        self.memory = memory_patch.start().return_value
        self.rag = rag_patch.start().return_value
        self.addCleanup(memory_patch.stop)
        self.addCleanup(rag_patch.stop)
        self.memory.run.return_value = "学习记录"
        self.rag.run.return_value = "✅ 完成"

    def test_duration_and_report(self):
        assistant = app.PDFLearningAssistant()
        assistant.add_note("学习笔记", "attention")
        stats = assistant.get_stats()
        self.assertEqual(stats["学习笔记"], 1)
        self.assertTrue(stats["会话时长"].endswith("秒"))
        report = assistant.generate_report(save_to_file=False)
        self.assertGreaterEqual(report["session_info"]["duration_seconds"], 0)
        self.assertEqual(
            report["session_info"]["start_time"],
            assistant.stats["session_start"].isoformat(),
        )
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            report_path = Path(directory) / "report.json"
            with patch.object(app, "open", create=True) as open_report:
                with report_path.open("w", encoding="utf-8") as output:
                    open_report.return_value.__enter__.return_value = output
                    assistant.generate_report()
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["learning_metrics"]["concepts_learned"], 1)

    def test_failed_document_does_not_count_as_loaded(self):
        assistant = app.PDFLearningAssistant()
        self.rag.run.return_value = "❌ 添加文档失败"
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            pdf_path = Path(directory) / "sample.pdf"
            pdf_path.touch()
            result = assistant.load_document(str(pdf_path))
        self.assertFalse(result["success"])
        self.assertEqual(assistant.stats["documents_loaded"], 0)
        self.assertIsNone(assistant.current_document)
        self.assertFalse(assistant.loading_document)
        self.memory.run.assert_not_called()

    def test_loading_blocks_questions_and_duplicate_loads(self):
        assistant = app.PDFLearningAssistant()
        assistant.loading_document = True
        self.assertIn("正在加载", assistant.ask("问题"))
        result = assistant.load_document(str(Path(__file__)))
        self.assertFalse(result["success"])
        self.rag.run.assert_not_called()

    def test_empty_document_warning_is_not_success(self):
        assistant = app.PDFLearningAssistant()
        self.rag.run.return_value = "⚠️ 未能从文件解析内容"
        result = assistant.load_document(str(Path(__file__)))
        self.assertFalse(result["success"])
        self.assertIsNone(assistant.current_document)
        self.assertFalse(assistant.loading_document)

    def test_rate_limit_does_not_record_successful_learning(self):
        assistant = app.PDFLearningAssistant()
        assistant.current_document = "sample.pdf"
        self.rag.run.return_value = "❌ 智能问答失败: 429 Too Many Requests"
        self.assertIn("限流或额度不足", assistant.ask("问题"))
        self.assertFalse(self.rag.run.call_args.args[0]["enable_advanced_search"])
        self.assertEqual(assistant.stats["questions_asked"], 0)
        self.assertEqual(self.memory.run.call_count, 1)

    def test_web_callbacks_and_filepath_upload(self):
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}):
            demo = app.create_gradio_ui()
        self.addCleanup(demo.close)
        self.assertIsInstance(demo.get_api_info(), dict)
        callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values() if fn.fn}
        self.assertIn("请先初始化", callbacks["load_pdf"](None))
        self.assertIn("请先初始化", callbacks["add_note_ui"]("笔记", ""))
        self.assertIn("请先初始化", callbacks["get_stats_ui"]())
        self.assertIn("请先初始化", callbacks["generate_report_ui"]())
        _, history = callbacks["chat"]("问题", [])
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertIn("请先初始化", history[-1]["content"])
        callbacks["init_assistant"]("")
        self.assertIn("请上传", callbacks["load_pdf"](None))
        self.assertIn("不能为空", callbacks["add_note_ui"]("  ", ""))
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            pdf_path = Path(directory) / "sample.pdf"
            pdf_path.touch()
            self.assertIn("加载成功", callbacks["load_pdf"](str(pdf_path)))
            self.assertEqual(self.rag.run.call_args.args[0]["file_path"], str(pdf_path))
        _, history = callbacks["chat"]("技术问题", [])
        self.assertIn("回答", history[-1]["content"])
        chatbot = next(block for block in demo.blocks.values() if isinstance(block, app.gr.Chatbot))
        self.assertIsNotNone(chatbot.postprocess(history))
        self.assertEqual(self.rag.run.call_args.args[0]["action"], "ask")
        _, history = callbacks["chat"]("回顾学习", history)
        self.assertIn("学习回顾", history[-1]["content"])
        self.assertEqual(len(history), 4)
        self.assertEqual(callbacks["chat"]("  ", history), ("", history))
        self.assertIn("已保存", callbacks["add_note_ui"]("笔记", ""))
        self.assertIn("提问次数", callbacks["get_stats_ui"]())

    def test_launch_parameters_match_installed_version(self):
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}), patch.object(
            app.gr.Blocks, "launch", autospec=True
        ) as launch:
            app.main()
        self.assertTrue(launch.call_args.kwargs["inbrowser"])
        self.assertIsNone(launch.call_args.kwargs["server_port"])
        launch.call_args.args[0].close()


class PDFIngestionChecks(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.pdf_path = Path(self.directory.name) / "sample.pdf"
        self.pdf_path.write_bytes(b"%PDF-1.4 test fixture")
        self.reader = SimpleNamespace(
            is_encrypted=False,
            pages=[SimpleNamespace(extract_text=lambda: "中文知识" * 3200)],
        )
        self.store = Mock(vector_size=3)
        self.store.add_vectors.return_value = True
        self.embedder = Mock()
        self.embedder.encode.side_effect = lambda texts: [[1.0, 0.5, 0.25] for _ in texts]
        self.tool = pdf_rag_tool.PDFRAGTool.__new__(pdf_rag_tool.PDFRAGTool)
        self.tool.rag_namespace = "pdf_test"
        self.tool.initialized = True
        self.tool._pipelines = {"pdf_test": {"store": self.store}}

    def ingest(self, **kwargs):
        with patch.object(pdf_rag_tool, "PdfReader", return_value=self.reader), patch.object(
            pdf_rag_tool, "get_text_embedder", return_value=self.embedder
        ):
            return self.tool.run({
                "action": "add_document",
                "file_path": str(self.pdf_path),
                "namespace": "pdf_test",
                **kwargs,
            })

    def test_long_paragraphs_have_bounded_chunks_batches_and_progress(self):
        updates = []
        result = self.ingest(progress=lambda value, description: updates.append((value, description)))
        self.assertTrue(result.startswith("✅"))
        batches = [call.args[0] for call in self.embedder.encode.call_args_list]
        self.assertGreater(len(batches), 1)
        self.assertTrue(all(1 <= len(batch) <= 10 for batch in batches))
        texts = [text for batch in batches for text in batch]
        self.assertTrue(all(0 < len(text) <= 1000 for text in texts))
        metadata = self.store.add_vectors.call_args.kwargs["metadata"]
        self.assertEqual(len(metadata), len(texts))
        self.assertTrue(all(item["rag_namespace"] == "pdf_test" for item in metadata))
        self.assertTrue(all(item["page"] == 1 for item in metadata))
        values = [value for value, _ in updates]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[-1], 0.99)

    def test_api_failure_stops_without_retry_or_database_write(self):
        self.embedder.encode.side_effect = RuntimeError("Embedding REST 调用失败: 400")
        result = self.ingest()
        self.assertTrue(result.startswith("❌"))
        self.assertIn("已停止加载", result)
        self.assertEqual(self.embedder.encode.call_count, 1)
        self.store.add_vectors.assert_not_called()

    def test_invalid_vectors_are_never_written(self):
        for vector in ([0.0, 0.0, 0.0], [1.0], [float("nan"), 1.0, 1.0]):
            with self.subTest(vector=vector):
                self.embedder.encode.side_effect = lambda texts: [vector for _ in texts]
                self.assertTrue(self.ingest().startswith("❌"))
                self.store.add_vectors.assert_not_called()

    def test_empty_pdf_and_failed_storage_report_failure(self):
        self.reader.pages = [SimpleNamespace(extract_text=lambda: "")]
        result = self.ingest()
        self.assertIn("OCR", result)
        self.embedder.encode.assert_not_called()
        self.reader.pages = [SimpleNamespace(extract_text=lambda: "有效的文档文字")]
        self.store.add_vectors.return_value = False
        self.assertIn("写入失败", self.ingest())


if __name__ == "__main__":
    unittest.main()
