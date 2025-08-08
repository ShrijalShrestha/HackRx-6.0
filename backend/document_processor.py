import logging
import time
import hashlib
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # We use PyMuPDF (fitz) for PDFs and community loaders for other formats
        self._fitz_available = False
        try:
            import fitz  # PyMuPDF
            self._fitz_available = True
        except Exception as e:
            logger.warning("PyMuPDF (fitz) not available: %s. PDF parsing will fall back to community loaders.", e)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info("Document processor initialized. Using PyMuPDF for PDFs: %s", self._fitz_available)

    def load_documents(self, directory: str = None) -> List[Document]:
        base = Path(directory or config.DATA_DIRECTORY)
        if not base.exists():
            logger.error("Directory not found: %s", base)
            return []

        files = list(f for f in base.iterdir() if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS)
        if not files:
            logger.warning("No supported files in %s", base)
            return []

        docs = []
        for file in files:
            try:
                docs.extend(self._process_file(file))
            except Exception as e:
                logger.error("Error processing %s: %s", file.name, e)
        logger.info("Loaded %d documents", len(docs))
        return docs

    def _process_file(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()
        if ext == ".pdf" and self._fitz_available:
            return self._parse_pdf_with_pymupdf(file_path)
        # For other types or if fitz isn't available, use community loaders
        return self._parse_with_fallback(file_path)
    def _parse_pdf_with_pymupdf(self, file_path: Path) -> List[Document]:
        """Parse a PDF file using PyMuPDF (fitz)."""
        start = time.time()
        try:
            import fitz  # type: ignore

            text_parts: List[str] = []
            with fitz.open(str(file_path)) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    # Use text with layout; if empty, fallback to simple text
                    page_text = page.get_text("text") or page.get_text()
                    # Add a simple page header to keep some structure
                    text_parts.append(f"\n\n--- Page {page_num + 1} ---\n{page_text.strip()}")

            content = "".join(text_parts).strip()
            if not content:
                logger.warning("Empty text extracted with PyMuPDF for %s; falling back to community loader.", file_path.name)
                return self._parse_with_fallback(file_path)

            doc_obj = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "parser": "pymupdf",
                    "processing_time": round(time.time() - start, 3),
                },
            )
            logger.info("Parsed PDF with PyMuPDF: %s in %.2fs", file_path.name, time.time() - start)
            return [doc_obj]
        except Exception as e:
            logger.error("PyMuPDF parsing failed for %s: %s", file_path.name, e)
            return self._parse_with_fallback(file_path)


    def _parse_with_fallback(self, file_path: Path) -> List[Document]:
        try:
            from langchain_community.document_loaders import (
                PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredEmailLoader
            )
            loader = {
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".eml": UnstructuredEmailLoader,
                ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
                ".md": lambda p: TextLoader(str(p), encoding="utf-8")
            }.get(file_path.suffix.lower())

            if not loader:
                logger.error("Unsupported file type: %s", file_path.suffix.lower())
                return []

            docs = loader(str(file_path)).load()
            for d in docs:
                d.metadata.update({
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "parser": "fallback"
                })
            return [d for d in docs if d.page_content.strip()]
        except Exception as e:
            logger.error("Fallback parsing failed for %s: %s", file_path.name, e)
            return []

    # Backward-compatible alias for callers using the older method name
    def load_documents_from_directory(self, directory: str = None) -> List[Document]:
        return self.load_documents(directory)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for idx, doc in enumerate(documents):
            for ci, text in enumerate(self.text_splitter.split_text(doc.page_content)):
                if not text.strip():
                    continue
                chunk_id = f"{hashlib.md5((doc.metadata.get('file_path','')+str(idx)).encode()).hexdigest()[:8]}_{ci}"
                chunks.append(Document(
                    page_content=text,
                    metadata={**doc.metadata, "chunk_id": chunk_id, "chunk_index": ci}
                ))
        logger.info("Created %d chunks", len(chunks))
        return chunks

    def get_stats(self, documents: List[Document]) -> dict:
        total = len(documents)
        size = sum(len(d.page_content) for d in documents)
        return {"total_chunks": total, "total_size": size, "average_chunk_size": size / total if total else 0}
