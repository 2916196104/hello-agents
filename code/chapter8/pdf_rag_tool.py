"""PDF 入库适配：真实解析、受限分块、严格向量校验和进度反馈。"""

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid5

from hello_agents.memory.embedding import get_text_embedder
from hello_agents.tools import RAGTool
from pypdf import PdfReader


ProgressCallback = Callable[[float, str], None]


class PDFRAGTool(RAGTool):
    """保留 RAGTool 的检索和问答，替换有错误兜底的 PDF 入库流程。"""

    def _add_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        namespace: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        progress: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> str:
        if not 0 <= chunk_overlap < chunk_size <= 1000:
            raise ValueError("分块大小应为 1–1000，相邻重叠必须小于分块大小")
        pdf_path = Path(file_path)
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError("请选择有效的 PDF 文件")

        def notify(value: float, description: str) -> None:
            if progress is not None:
                progress(value, description)

        notify(0, "正在读取 PDF…")
        reader = PdfReader(pdf_path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF 已加密，请先解密后上传")
        page_count = len(reader.pages)
        doc_id = document_id or hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        target_namespace = namespace or self.rag_namespace
        chunks: List[Dict[str, Any]] = []

        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").replace("\x00", "").strip()
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                content = text[start:end].strip()
                if content:
                    chunks.append({
                        "id": str(uuid5(NAMESPACE_URL, f"{target_namespace}/{doc_id}/{page_number}/{start}")),
                        "content": content,
                        "source_path": str(pdf_path),
                        "doc_id": doc_id,
                        "page": page_number,
                        "start": start,
                        "end": end,
                    })
                if end == len(text):
                    break
                start = end - chunk_overlap
            notify(0.2 * page_number / max(page_count, 1), f"解析 PDF：{page_number}/{page_count} 页")

        if not chunks:
            raise ValueError("PDF 未提取到文字；如果是扫描件，请先进行 OCR 文字识别")
        print(f"[PDF] 解析完成：{page_count} 页，{len(chunks)} 个文本片段", flush=True)
        pipeline = self._get_pipeline(target_namespace)
        store = pipeline["store"]
        embedder = get_text_embedder()
        dimension = store.vector_size
        vectors: List[List[float]] = []

        # 先完成全部向量生成，任何错误都阻止入库，避免写入零向量。
        for start in range(0, len(chunks), 10):
            batch = chunks[start:start + 10]
            notify(0.2 + 0.65 * start / len(chunks), f"生成向量：{start}/{len(chunks)} 个片段")
            try:
                encoded = embedder.encode([chunk["content"] for chunk in batch])
                if len(encoded) != len(batch):
                    raise ValueError("向量接口返回数量与文本数量不一致")
                for vector in encoded:
                    values = [float(value) for value in vector]
                    if len(values) != dimension:
                        raise ValueError(f"向量维度不匹配：知识库需要 {dimension}，接口返回 {len(values)}")
                    if not all(math.isfinite(value) for value in values) or not any(values):
                        raise ValueError("向量接口返回了无效向量或零向量")
                    vectors.append(values)
            except Exception as error:
                raise RuntimeError(f"第 {start + 1} 个片段起的向量生成失败，已停止加载：{error}") from error
            print(f"[PDF] 向量生成：{len(vectors)}/{len(chunks)}", flush=True)

        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            metadata = [
                {
                    **{key: value for key, value in chunk.items() if key != "id"},
                    "memory_id": chunk["id"],
                    "user_id": "rag_user",
                    "memory_type": "rag_chunk",
                    "rag_namespace": target_namespace,
                    "namespace": target_namespace,
                    "is_rag_data": True,
                    "data_source": "rag_pipeline",
                }
                for chunk in batch
            ]
            notify(0.85 + 0.14 * start / len(chunks), f"写入知识库：{start}/{len(chunks)} 个片段")
            if not store.add_vectors(
                vectors=vectors[start:start + 64],
                metadata=metadata,
                ids=[chunk["id"] for chunk in batch],
            ):
                raise RuntimeError("知识库写入失败，请检查 Qdrant 连接和服务状态")

        notify(0.99, "文档入库完成，正在记录学习历程…")
        return f"✅ 文档已添加到知识库：{pdf_path.name}（{page_count} 页，{len(chunks)} 个片段）"
