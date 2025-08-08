"""
Minimal FastAPI app exposing a single endpoint:

POST /hackrx/run
Authorization: Bearer <api_key>
Content-Type: application/json

Payload:
{
  "documents": "https://example.com/policy.pdf" | ["http://...", "..."],
  "questions": ["Question 1", "Question 2"]
}

Response:
{ "answers": ["Answer 1", "Answer 2"] }
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Union
import mimetypes
import logging
import time
import requests
from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field

from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from query_processor import QueryProcessor


logger = logging.getLogger(__name__)

app = FastAPI(title="HackRx Runner", version="1.0.0")
_START_TIME = time.time()


# Initialize core components once
try:
    _doc_processor = DocumentProcessor()
    _vector_manager = VectorStoreManager()
    _query_processor = QueryProcessor(_vector_manager)
except Exception as e:
    logger.exception("Failed to initialize core components: %s", e)
    raise


class RunRequest(BaseModel):
    documents: Union[str, List[str]] = Field(..., description="URL or list of URLs to documents")
    questions: List[str] = Field(..., min_items=1, description="List of questions to ask")


class RunResponse(BaseModel):
    answers: List[str]


def _require_bearer(token_header: Union[str, None]) -> None:
    """Simple bearer token validation using env var HACKRX_API_KEY.
    If HACKRX_API_KEY not set, the check is skipped (dev mode).
    """
    expected = os.getenv("HACKRX_API_KEY", "").strip()
    if not expected:
        return  # no auth configured; allow
    if not token_header or not token_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    provided = token_header.split(" ", 1)[1].strip()
    if provided != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def _download_to(temp_dir: Path, url: str) -> Path:
    """Download a single file URL into temp_dir and return the local path."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        # Try to derive filename and extension
        name_from_url = Path(url.split("?")[0]).name or "document"
        suffix = Path(name_from_url).suffix
        if not suffix:
            # Guess from content-type
            ctype = resp.headers.get("Content-Type", "application/octet-stream")
            guessed = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ".pdf"
            name_from_url += guessed
        local_path = temp_dir / name_from_url
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download {url}: {e}")


@app.post("/hackrx/run", response_model=RunResponse)
def run_hackrx(req: RunRequest, authorization: Union[str, None] = Header(default=None)):
    # Auth
    _require_bearer(authorization)

    # Normalize documents to list
    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]

    # Prepare a temp workspace for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix="hackrx_"))
    try:
        # Download all docs
        local_files: List[Path] = []
        for url in doc_urls:
            if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
                local_files.append(_download_to(temp_dir, url))
            else:
                # Treat as local file path
                p = Path(url)
                if not p.exists():
                    raise HTTPException(status_code=400, detail=f"File not found: {url}")
                # Copy into temp_dir for consistent processing
                target = temp_dir / p.name
                shutil.copy2(p, target)
                local_files.append(target)

        # Process documents -> chunks
        documents = _doc_processor.load_documents(str(temp_dir))
        if not documents:
            raise HTTPException(status_code=400, detail="No readable content extracted from provided documents")
        chunks = _doc_processor.chunk_documents(documents)
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to create chunks from documents")

        # Build vector store for this run
        if not _vector_manager.create_vector_store(chunks):
            raise HTTPException(status_code=500, detail="Failed to create vector store for documents")
        time.sleep(5)
        # Answer questions
        answers: List[str] = []
        for q in req.questions:
            result = _query_processor.process_query(q, max_results=5, score_threshold=0.1)
            answers.append(result.get("answer", ""))

        return RunResponse(answers=answers)
    finally:
        # Clean up downloaded files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# Optional root route for quick sanity
@app.get("/")
def root():
    return {"status": "ok", "service": "hackrx-runner"}


@app.get("/health")
def health():
    """Basic health endpoint for quick checks."""
    try:
        vs_ok = _vector_manager.check_vector_store_health()
    except Exception:
        vs_ok = False
    try:
        llm_ok = bool(getattr(_query_processor, "llm", None))
    except Exception:
        llm_ok = False
    return {
        "status": "healthy",
        "service": "hackrx-runner",
        "system_ready": vs_ok and llm_ok,
        "components": {
            "vector_store": vs_ok,
            "llm": llm_ok,
            "document_processor": True,
        },
        "uptime_seconds": round(time.time() - _START_TIME, 2),
    }
